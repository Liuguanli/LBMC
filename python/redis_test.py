# load data and queries
import argparse
import bisect
import copy
import csv
import json
import math
import os
import os.path
# from mpl_toolkits.axisartist.axislines import SubplotZero
# import matplotlib.patches as patches
import random
import sys
import time
from math import remainder
from time import perf_counter
from random import shuffle

import matplotlib.pyplot as plt
# from utils import Point
# from utils import Window
# from utils import Config
import numpy as np
import psycopg2
import psycopg2.extras
import redis
from app_query_gen import (data_set_ratios, gen_2d_query_osm, gen_3d_query_nyc,
                           gen_3d_query_tpch, read_windows, write_windows)
from ddpg import DeepDeterministicPolicyGradient
from dqn import DeepQNetwork
from global_cost import GlobalCost
from hilbertcurve.hilbertcurve import HilbertCurve
from local_cost import LocalCost
from psycopg2 import sql
# from app1 import datasets
# import utils
from utils import Point, Window, ratio_to_pattern

np.random.seed(1)
random.seed(1)


bit_letters = ["A", "B", "C", "D", "E"]

bit_letters_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

iteration = 20000
MEMORY_CAPACITY = 500
train_gap = 10

class RLconfig:
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            json_content = json.load(f)
        print("json_path", json_path)
        self.initial_BMC = json_content["init"]
        self.is_reward_zero = json_content["is_reward_zero"] == "1"
        self.iteration = int(json_content["iteration"])
        self.memory_capacity = int(json_content["memory_size"])
        self.cost = json_content["cost"]
        # self.result = json_content["result"]

    def is_both(self):
        return self.cost == "both"
    
    def is_product(self):
        return self.cost == "product"
    
    def is_sum(self):
        return self.cost == "sum"

    def is_local(self):
        return self.cost == "local"

    def is_global(self):
        return self.cost == "global"


def choose_RL(name, length):
    if name == 'dqn':
        RL = DeepQNetwork(length, length, width=64, reward_decay=0.9,
                          e_greedy=0.8, learning_rate=0.01, memory_size=MEMORY_CAPACITY,
                          batch_size=32, e_greedy_increment=float(train_gap) / iteration)
    elif name == 'ddpg':
        REPLACEMENT = [
            dict(name='soft', tau=0.01),
            dict(name='hard', rep_iter_a=600, rep_iter_c=500)
        ][0]
        LR_A = 0.01
        LR_C = 0.01
        GAMMA = 0.9
        EPSILON = 0.9
        VAR_DECAY = .9995
        RL = DeepDeterministicPolicyGradient(length, length, 1, LR_A, LR_C, REPLACEMENT, GAMMA,
                                             EPSILON)
    return RL


def exchange_bit(string, position):
    if position >= len(string) - 1:
        return False, string
    left = string[position]
    right = string[position + 1]
    exchange = True
    if left == right:
        exchange = False
    return exchange, string[:position] + right + left + string[position+2:]


def bit_distribution_to_array(bit_distribution, dim, encoding="one_hot"):
    res = []
    if encoding is None or encoding == "no":
        for s in bit_distribution:
            res.append(bit_letters.index(s))
        return res
    if encoding == "normalized":
        for s in bit_distribution:
            res.append((bit_letters.index(s) + 1.0) / dim)
        return res
    if encoding == "one_hot":
        one_hot = {2: [[0, 1], [1, 0]], 3: [[0, 0, 1], [0, 1, 0], [1, 0, 0]]}
        for s in bit_distribution:
            res.extend(one_hot[dim][bit_letters.index(s)])
        return res


def get_learned_results(LC: LocalCost, GC: GlobalCost, config: RLconfig, bits_nums, encoding="one_hot"):
    pre_global_cost, _ = GC.global_cost(config.initial_BMC)
    pre_local_cost, _ = LC.local_cost(config.initial_BMC)
    initial_global_cost = pre_global_cost
    initial_local_cost = pre_local_cost
    iteration = config.iteration
    MEMORY_CAPACITY = config.memory_capacity

    print("learning cost:" + config.cost)

    dim = len(bits_nums)
    length = sum(bits_nums)
    if encoding is None:
        scale = 1
    else:
        scale = dim if encoding == "one_hot" else 1
    RL = choose_RL("dqn", length * scale)
    step = 0
    M = 2
    reward_decreased_num = 0
    result_BMC = config.initial_BMC
    for i in range(M):
        BMC = config.initial_BMC
        while step < iteration:
            state = bit_distribution_to_array(BMC, dim, encoding)
            force_random = step < MEMORY_CAPACITY
            action = RL.choose_action(np.array(state), force_random)
            action = int(action / scale)
            exchange, new_BMC = exchange_bit(BMC, action)
            gc, _ = GC.global_cost(new_BMC)
            lc, _ = LC.local_cost(new_BMC)
            r_g = 0
            r_l = 0
            if gc < pre_global_cost:
                # pre_global_cost = gc
                r_g = 0.5
            elif gc > pre_global_cost:
                r_g = -0.5

            if lc < pre_local_cost:
                # pre_local_cost = lc
                r_l = 0.5
            elif lc > pre_local_cost:
                r_l = -0.5

            if config.is_both():
                reward = r_g + r_l
            elif config.is_global():
                temp_reward = gc
                pre_reward = pre_global_cost
                reward = 1 - temp_reward / pre_reward
            elif config.is_local():
                temp_reward = lc
                pre_reward = pre_local_cost
                reward = 1 - temp_reward / pre_reward
            elif config.is_product():
                temp_reward = gc * lc
                pre_reward = pre_global_cost * pre_local_cost
                reward = 1 - temp_reward / pre_reward
            elif config.is_sum():
                temp_reward = lc / initial_local_cost + gc / initial_global_cost
                pre_reward = pre_local_cost / initial_local_cost + pre_global_cost / initial_global_cost
                reward = 1 - temp_reward / pre_reward
            # print("BMC:", BMC, "->action:", action, "->new_BMC:", new_BMC, "reward:", reward)

            BMC = new_BMC
            if reward > 0 or (config.is_reward_zero and reward == 0):
                reward_decreased_num += 1
                result_BMC = new_BMC
            # if exchange is True:
            current_state = bit_distribution_to_array(new_BMC, dim, encoding)
            RL.store_transition(state, action, reward, current_state)
            if (step > MEMORY_CAPACITY) and (step % train_gap == 0):
                RL.learn()
                # print("reward:", reward)

            pre_global_cost = gc
            pre_local_cost = lc
            step += 1
    print("reward_decreased_num:", reward_decreased_num)
    print("result_BMC:", result_BMC)
    return result_BMC


def load_queries(file_name="OSM_1000_64_262144_dim2_norm", dim_scalar=[pow(2, 16) - 1, pow(2, 16) - 1], bit_gap=0):
    print("load queries")
    query_path = "./Learned-BMTree/query"
    dim = 2
    windows = []
    print(os.path.join(query_path, file_name + ".json"))
    if not os.path.exists(os.path.join(query_path, file_name + ".json")):
        return windows
    with open(os.path.join(query_path, file_name + ".json"), 'r') as f:
        queries = json.load(f)

        for i, row in enumerate(queries):
            dimension_low = []
            dimension_high = []
            dimension_low_raw = []
            dimension_high_raw = []
            for j in range(dim):
                dim_l = float(row[j])
                dim_h = float(row[j + dim])
                dimension_low_raw.append(dim_l)
                dimension_high_raw.append(dim_h)
                dimension_low.append(dim_l * dim_scalar[j])
                dimension_high.append(dim_h * dim_scalar[j])

            window = Window(dimension_low, dimension_high,
                            dimension_low_raw, dimension_high_raw)
            windows.append(window)
    return windows


def quilts_curve_design_multi(windows, frequent_window, dim, bits_nums):
    bits_nums_copy = copy.deepcopy(bits_nums)
    l_i = copy.deepcopy(bits_nums)
    u_i = [0 for i in range(dim)]
    d_i = []
    for i in range(dim):
        dim_len = max(frequent_window.dimension_high[i] -
                      frequent_window.dimension_low[i], 1)
        bit_num = math.ceil(math.log(dim_len) / math.log(2))
        d_i.append(bit_num)
    for window in windows:
        for i in range(dim):
            dim_len = max(
                window.dimension_high[i] - window.dimension_low[i], 1)
            bit_num = math.ceil(math.log(dim_len) / math.log(2))
            l_i[i] = min(l_i[i], bit_num)
            u_i[i] = max(u_i[i], bit_num)
    most_sig_res = ""
    for i in range(dim):
        u_i[i] = u_i[i] - d_i[i]
        d_i[i] = d_i[i] - l_i[i]

    length = sum(l_i)
    while length > 0:
        for i in range(dim):
            if l_i[i] > 0:
                bits_nums_copy[i] -= 1
                l_i[i] -= 1
                # most_sig_res = bit_letters[i] + most_sig_res
                most_sig_res += bit_letters[i]
                length -= 1
    middle_sig_res = ""
    length = sum(d_i)
    while length > 0:
        for i in range(dim):
            if d_i[i] > 0:
                bits_nums_copy[i] -= 1
                d_i[i] -= 1
                # middle_sig_res = bit_letters[i] + middle_sig_res
                middle_sig_res += bit_letters[i]
                length -= 1
    length = sum(u_i)
    least_sig_res = ""
    C_least_sig_res = ""
    Z_least_sig_res = ""
    for i, bit_num in enumerate(bits_nums_copy):
        for j in range(bit_num):
            C_least_sig_res = C_least_sig_res + bit_letters[i]
    C_res = C_least_sig_res + least_sig_res + middle_sig_res + most_sig_res
    length = sum(bits_nums_copy)

    while length > 0:
        for i in range(dim):
            if bits_nums_copy[i] > 0:
                bits_nums_copy[i] -= 1
                Z_least_sig_res = Z_least_sig_res + bit_letters[i]
                length -= 1

    Z_res = Z_least_sig_res + least_sig_res + middle_sig_res + most_sig_res
    return C_res, Z_res

def get_optimal_quilts_result(LC:LocalCost, GC:GlobalCost, windows:list, dim, bits_nums):
    BMCs = []
    for i in range(10):
        C_res, Z_res = quilts_curve_design_multi(windows, windows[i], dim, bits_nums)
        BMCs.append(C_res)
        BMCs.append(Z_res)
    minimal_total_cost = 0
    opt_BMC = ""
    for i, BMC in enumerate(BMCs):
        # local_cost = LC.naive_local_cost(BMC)
        # global_cost = GC.naive_global_cost(BMC)
        local_cost = LC.local_cost(BMC)[0]
        global_cost = GC.global_cost(BMC)[0]
        total_cost = global_cost * local_cost
        if i == 0:
            minimal_total_cost = total_cost
            opt_BMC = BMC
            continue
        if total_cost < minimal_total_cost:
            minimal_total_cost = total_cost
            opt_BMC = BMC
    return opt_BMC

def get_bit_distribution(windows, name, dim=2, bits_nums=[16, 16]):

    res = ""
    if name == "LC":
        for i in range(dim):
            res = res + bit_letters[i] * bits_nums[i]
    elif name == "ZC":
        total_bits = sum(bits_nums)
        while total_bits > 0:
            for i in range(dim):
                if (bits_nums[i] > 0):
                    res = res + bit_letters[i]
                    bits_nums[i] -= 1
                    total_bits -= 1
    elif name == "QUILTS":
        start_time = time.time()
        GC = GlobalCost(windows, bits_nums)
        LC = LocalCost(windows, bits_nums)
        end_time = time.time()
        print("cost initition time usage: {} s".format((end_time - start_time)))
        start_time = time.time()
        # _, res = quilts_curve_design_multi(windows, windows[0], dim, bits_nums)
        res = get_optimal_quilts_result(LC, GC, windows, dim, bits_nums)
        end_time = time.time()
        print("curve design time usage: {} s".format((end_time - start_time)))
        # print("QUILTS:", res)
    elif name == "LBMC":
        GC = GlobalCost(windows, bits_nums)
        LC = LocalCost(windows, bits_nums)
        print(args.width, args.height)
        if args.width == 0 and args.height == 0:
            config = RLconfig("./python/configs/config_mix.json")
        else:
            if args.bit_num == '20':
                config_path = f"./python/configs/config_{args.width}_{args.height}"
                if args.lbmc_cost != "":
                    config_path += f"_{args.lbmc_cost}"
            else:
                config_path = f"./python/configs/config_{args.width}_{args.height}_{args.bit_num}"
            config_path += ".json"

            config = RLconfig(config_path)
        # _, res = quilts_curve_design_multi(windows, windows[0], dim, bits_nums)
        # config.initial_BMC = res
        start_time = time.time()
        res = get_learned_results(LC, GC, config, bits_nums)
        end_time = time.time()
        print("time usage: {} s".format((end_time - start_time)))

    return res


def load_data(file_name="OSM_10000", bit_gap=0):
    data_path = "./Learned-BMTree/data"
    data = []
    if not os.path.exists(os.path.join(data_path, file_name + ".json")):
        return data
    with open(os.path.join(data_path, file_name + ".json"), 'r') as f:
        dataset = json.load(f)
        scale = 2 ** bit_gap
        for row in dataset:
            point = Point([int(row[0] * scale), int(row[1] * scale)])
            data.append(point)
    return data


def get_BMC_value(point, bit_distribution, dim=2):

    xs = point

    merged_value = 0
    idxs = [0 for i in range(dim)]
    shift_count = 0

    for char in reversed(bit_distribution):
        index = bit_letters_map[char]
        bit = (xs[index] >> idxs[index]) & 1
        idxs[index] += 1
        merged_value |= (bit << shift_count)

        shift_count += 1

    return merged_value


def order_data_points(data, name, bit_distribution, bit_num):

    if name == "HC":
        hilbert_curve = HilbertCurve(bit_num, 2)
        for point in data:
            point.value = hilbert_curve.distances_from_points(
                [[max(x - 1, 0) for x in point.xs]])[0]
        data.sort(key=lambda x: x.value)
    elif name == "RANDOM":
        shuffle(data)
        for i, point in enumerate(data):
            point.value = i
    else:
        for point in data:
            point.value = get_BMC_value(point.xs, bit_distribution)
        data.sort(key=lambda x: x.value)
    return data



def insert_data(r, data):
    r.delete("myzset")

    for score, value in data:
        r.zadd('myset', {value: score})

def range_query(r, rmin_score, max_score):
    return r.zrangebyscore('myset', rmin_score, max_score)

def filter_query(candidates, query_range, dim_names):
    res = []
    for candidate in candidates:
        candidate = json.loads(candidate)
        for i, dim_name in enumerate(dim_names):
            dim_val = candidate[dim_name]
            if dim_val < query_range.dimension_low[i] or dim_val > query_range.dimension_high[i]:
                break
        res.append(candidate)


    return res

def test_redis(data, queries, bit_distribution, db_password=123456, dim=2, bit_num=20, name="BMC"):
    print("test_redis")
    r = redis.Redis(host='localhost', port=6379, db=0)

    dim_names = [f'dim{i + 1}' for i in range(dim)]
    
    redis_data = []
    for item in data:
        value_dict = {dim_names[0]: item.xs[0], dim_names[1]: item.xs[1]}
        value = json.dumps(value_dict)
        redis_data.append((item.value, value))

    insert_data(r, redis_data)

    query_time_usage = 0
    filter_time_usage = 0
    number_of_records = 0

    print("bit_distribution:", bit_distribution)
    hilbert_curve = HilbertCurve(bit_num, 2)

    for i, query in enumerate(queries):

        if name == "RANDOM":
            curve_value_low = 0
            curve_value_high = len(data) - 1
        elif name == "HC":
            curve_value_low = hilbert_curve.distances_from_points(
                [[max(x - 1, 0) for x in query.dimension_low]])[0]
            curve_value_high = hilbert_curve.distances_from_points(
                [[max(x - 1, 0) for x in query.dimension_high]])[0]
        else:
            curve_value_low = get_BMC_value(
                query.dimension_low, bit_distribution)
            curve_value_high = get_BMC_value(
                query.dimension_high, bit_distribution)
        
        start_time = time.time()
        candidates = range_query(r, curve_value_low, curve_value_high)
        end_time = time.time()
        query_time_usage += end_time - start_time

        start_time = time.time()
        result = filter_query(candidates, query, dim_names)
        end_time = time.time()
        filter_time_usage += end_time - start_time

        # Get the number of records
        number_of_records += len(result)


    
    print('finish running, query time usage: {} ms'.format(
        query_time_usage * 1000 / len(queries)))
    print('finish running, filter time usage: {} ms'.format(
        filter_time_usage * 1000 / len(queries)))
    print('all_result_num: {}'.format(number_of_records))
    
    r.connection_pool.disconnect()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset name.")
    parser.add_argument("--test_query", help="")
    parser.add_argument("--training_query", help="")
    parser.add_argument("--name", help="")
    parser.add_argument("--db_password", help="")
    parser.add_argument("--width", help="", default=0)
    parser.add_argument("--height", help="", default=0)
    parser.add_argument("--lbmc_cost", help="")
    parser.add_argument("--bit_num", help="", default=20)

    # parser.add_argument("path", help="")
    args = parser.parse_args()

    '''redirect the std out to file'''
    result_save_path = f'result/redis/{args.name}/'
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
    
    bit_num = int(args.bit_num)
    bit_gap = bit_num - 20
    windows = load_queries(args.test_query, dim_scalar=[pow(2, bit_num) - 1, pow(2, bit_num) - 1], bit_gap=bit_gap)

    windows = load_queries(args.training_query, dim_scalar=[pow(2, bit_num) - 1, pow(2, bit_num) - 1], bit_gap=bit_gap)
    data = load_data(args.dataset, bit_gap=bit_gap)
    bit_distribution = get_bit_distribution(windows, args.name, dim=2, bits_nums=[bit_num, bit_num])
    data = order_data_points(data, args.name, bit_distribution, bit_num)

    f = open(result_save_path + '{}_{}_{}.txt'.format(args.dataset,
             args.test_query, args.name), 'a')
    sys.stdout = f
    print("\n")
    test_redis(data, windows, bit_distribution, args.db_password, bit_num=bit_num, name=args.name)