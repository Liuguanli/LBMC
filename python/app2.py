#!/usr/bin/env python
# coding: utf-8


# from threading import enumerate
from math import remainder
from app_query_gen import gen_3d_query_nyc, gen_3d_query_tpch
from app_query_gen import gen_2d_query_osm
from app_query_gen import data_set_ratios
import json
from app_query_gen import read_windows
from time import perf_counter
import matplotlib.pyplot as plt
# from mpl_toolkits.axisartist.axislines import SubplotZero
# import matplotlib.patches as patches
import random
import math
import copy
import time
import sys
from global_cost import GlobalCost
from local_cost import LocalCost
from dqn import DeepQNetwork
from ddpg import DeepDeterministicPolicyGradient
import os.path
from app_query_gen import write_windows
# from app1 import datasets
# import utils
from utils import ratio_to_pattern
# from utils import Point
# from utils import Window
# from utils import Config
import numpy as np
import os
import csv
from time import perf_counter

np.random.seed(1)
random.seed(10)
plt.rc('font', family='Times New Roman', size=20)
# plt.rc('lines', linewidth=3)


bit_letters = ["A", "B", "C", "D", "E"]
factor_letters = ["a", "b", "c", "d", "e"]
logger_print = True

class RLconfig:
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            json_content = json.load(f)
        self.initial_BMC = json_content["init"]
        self.is_reward_zero = json_content["is_reward_zero"] == "1"
        self.iteration = int(json_content["iteration"])
        self.memory_capacity = int(json_content["memory_size"])
        self.cost = json_content["cost"]
        # self.result = json_content["result"]
    
    def is_both(self):
        return self.cost == "both"
    
    def is_local(self):
        return self.cost == "local"

    def is_global(self):
        return self.cost == "global"

def quilts_curve_design_multi(windows, frequent_window, dim, bits_nums):
    bits_nums_copy = copy.deepcopy(bits_nums)
    l_i = copy.deepcopy(bits_nums)
    u_i = [0 for i in range(dim)]
    d_i = []
    for i in range(dim):
        dim_len = max(frequent_window.dimension_high[i] - \
            frequent_window.dimension_low[i], 1)
        bit_num = math.ceil(math.log(dim_len) / math.log(2))
        d_i.append(bit_num)
    for window in windows:
        for i in range(dim):
            dim_len = max(window.dimension_high[i] - window.dimension_low[i], 1)
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
        local_cost = LC.local_cost(BMC)
        global_cost = GC.global_cost(BMC)
        total_cost = global_cost * local_cost
        if i == 0:
            minimal_total_cost = total_cost
            opt_BMC = BMC
            continue
        if total_cost < minimal_total_cost:
            minimal_total_cost = total_cost
            opt_BMC = BMC
    return opt_BMC

iteration = 200000
MEMORY_CAPACITY = 500
train_gap = 1
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

def exchange_bit(string, position):
    if position >= len(string) - 1:
        return False, string
    left = string[position]
    right = string[position + 1]
    exchange = True
    if left == right:
        exchange = False
    return exchange, string[:position] + right + left + string[position+2:]

def get_learned_results(LC:LocalCost, GC:GlobalCost, config:RLconfig, bits_nums, encoding="one_hot"):
    pre_global_cost = GC.global_cost(config.initial_BMC)
    pre_local_cost = LC.local_cost(config.initial_BMC)
    iteration = config.iteration
    MEMORY_CAPACITY = config.memory_capacity

    dim = len(bits_nums)
    length = sum(bits_nums)
    if encoding is None:
        scale = 1
    else:
        scale = dim if encoding == "one_hot" else 1
    RL = choose_RL("dqn", length * scale)
    step = 0
    M = 1
    reward_decreased_num = 0
    result_BMC = config.initial_BMC
    for i in range(M):
        BMC = config.initial_BMC
        while step < iteration:
            state = bit_distribution_to_array(BMC, dim, encoding)
            force_random = step < MEMORY_CAPACITY
            action = RL.choose_action(np.array(state), force_random)
            exchange, new_BMC = exchange_bit(BMC, action)

            gc = GC.global_cost(new_BMC)
            lc = LC.local_cost(new_BMC)
            r_g = 0
            r_l = 0
            if gc < pre_global_cost:
                pre_global_cost = gc
                r_g = 0.5
            elif gc > pre_global_cost:
                r_g = -0.5

            if lc < pre_local_cost:
                pre_local_cost = lc
                r_l = 0.5
            elif lc > pre_local_cost:
                r_l = -0.5

            if config.is_both():
                reward = r_g + r_l
            elif config.is_global():
                reward = r_g
            elif config.is_local():
                reward = r_l
            BMC = new_BMC
            if reward > 0 or (config.is_reward_zero and reward == 0):
                reward_decreased_num += 1
                result_BMC = new_BMC
            # if exchange is True:
            current_state = bit_distribution_to_array(new_BMC, dim, encoding)
            RL.store_transition(state, action, reward, current_state)
            if (step > MEMORY_CAPACITY) and (step % train_gap == 0):
                RL.learn()
            step += 1
    print("reward_decreased_num", reward_decreased_num)
    return result_BMC, new_BMC


def save_BMC(json_path, data):
    with open(json_path, 'w') as outfile:
        json.dump(data, outfile)


def get_init_BMC(LC:LocalCost, GC:GlobalCost, dim:int):

    BMCs_2d = ["ABABABABABABABABABABABABABABABAB",
                "BABABABABABABABABABABABABABABABA",
                # "AAAAAAAAAAAAAAAABBBBBBBBBBBBBBBB",
                # "BBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAA"
                ]
    BMCs_3d = ["CBACBACBACBACBACBACBACBACBACBACBACBACBACBACBACBA",
                "BCABCABCABCABCABCABCABCABCABCABCABCABCABCABCABCA",
                "CABCABCABCABCABCABCABCABCABCABCABCABCABCABCABCAB",
                "ACBACBACBACBACBACBACBACBACBACBACBACBACBACBACBACB",
                "BACBACBACBACBACBACBACBACBACBACBACBACBACBACBACBAC",
                "ABCABCABCABCABCABCABCABCABCABCABCABCABCABCABCABC",
                # "AAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBCCCCCCCCCCCCCCCC",
                # "AAAAAAAAAAAAAAAACCCCCCCCCCCCCCCCBBBBBBBBBBBBBBBB",
                # "BBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAACCCCCCCCCCCCCCCC",
                # "BBBBBBBBBBBBBBBBCCCCCCCCCCCCCCCCAAAAAAAAAAAAAAAA",
                # "CCCCCCCCCCCCCCCCAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBB",
                # "CCCCCCCCCCCCCCCCBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAA"
                ]
    
    BMCs = BMCs_2d if dim == 2 else BMCs_3d
    minimal_total_cost = 0
    opt_BMC = ""
    for i, BMC in enumerate(BMCs):
        global_cost = GC.global_cost(BMC)
        local_cost = LC.local_cost(BMC)
        total_cost = global_cost * local_cost
        if i == 0:
            minimal_total_cost = total_cost
            opt_BMC = BMC
            continue
        if total_cost < minimal_total_cost:
            minimal_total_cost = total_cost
            opt_BMC = BMC
    return opt_BMC


def learn_mix_data(data_name: str):
    windows_path = "windows/app1/" + data_name + ".csv"
    config_path = "python/configs/" + data_name + ".json"

    bits_nums = [16, 16, 16]
    if data_name == "OSM":
        bits_nums = [16, 16]
    dim_scalar = [(pow(2, l) - 1) for l in bits_nums]
    windows = read_windows(windows_path, dim_scalar)
    config = RLconfig(config_path)

    dim = len(bits_nums)
    GC = GlobalCost(windows, bits_nums)
    LC = LocalCost(windows, bits_nums)

    QUILTS_BMC = get_optimal_quilts_result(LC, GC, windows, dim, bits_nums)
    # init_BMC = QUILTS_BMC
    # init_BMC = get_init_BMC(LC, GC, dim)
    # start_time = perf_counter()
    LBMC, current_BMC = get_learned_results(LC, GC, config, bits_nums)
    # end_time = perf_counter()
    # print('{0} costs {1:.2f}s'.format("learning time:", end_time - start_time))
    json_path = "windows/app1/" + data_name + ".json"
    print("LBMC      ", GC.global_cost(LBMC), LC.local_cost(LBMC))
    print("QUILTS_BMC", GC.global_cost(QUILTS_BMC), LC.local_cost(QUILTS_BMC))
    data = {'lbmc': LBMC , "quilts": QUILTS_BMC, "cbmc": current_BMC}
    save_BMC(json_path, data)


def learn_pattern_data(data_name: str):
    ratios = data_set_ratios[data_name]
    bits_nums = [16, 16, 16]
    if data_name == "OSM":
        bits_nums = [16, 16]
    dim_scalar = [(pow(2, l) - 1) for l in bits_nums]
    for i, ratio in enumerate(ratios):
        # if i != 1:
        #     continue
        windows_path = "windows/app2/" + data_name + "/" + ratio_to_pattern(ratio) + ".csv"
        config_path = "python/configs/" + data_name + "/" + ratio_to_pattern(ratio) + ".json"
        windows = read_windows(windows_path, dim_scalar)
        config = RLconfig(config_path)
        dim = len(bits_nums)
        GC = GlobalCost(windows, bits_nums)
        LC = LocalCost(windows, bits_nums)

        QUILTS_BMC = get_optimal_quilts_result(LC, GC, windows, dim, bits_nums)
        # init_BMC = "ABCABCABCABCABCABCABCABCABCABCABCABCABCABCABCABC"
        # if data_name == "TPCH":
        #     init_BMC = "CCCCCCCCCCCCCCCCAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBB"
        # if data_name == "OSM":
        #     config.initial_BMC = get_init_BMC(LC, GC, dim)
        LBMC, current_BMC = get_learned_results(LC, GC, config, bits_nums)
        json_path = "windows/app2/" + data_name + "/" + ratio_to_pattern(ratio) + ".json"
        print("LBMC      ", GC.global_cost(LBMC), LC.local_cost(LBMC))
        print("QUILTS_BMC", GC.global_cost(QUILTS_BMC), LC.local_cost(QUILTS_BMC))
        data = {'lbmc': LBMC, "quilts": QUILTS_BMC, "cbmc": current_BMC}
        save_BMC(json_path, data)
        # break

datasets = {"OSM": "OSM_100000000_1_2_.csv",
"NYC": "yellow_tripdata_2015_normalized_3d.csv",
"TPCH": "tpch_normalized_3d.csv"}
com_template = "./Main -s {dataset} -n BMC -c {scheme} -t {data_name} -d {dim} -w {window_path}"#.format(fname = "John", age = 36)

def run_mix(data_name: str, bit_key: str):
    dataset = datasets[data_name]
    json_path = "./windows/app1/" + data_name + ".json"
    window_path = "./windows/app1/" + data_name + ".csv"
    config_path = "python/configs/" + data_name + ".json"
    config = RLconfig(config_path)

    com_template = "./Main -s {dataset} -n BMC -c {scheme} -t {data_name} -d {dim} -w {window_path}"#.format(fname = "John", age = 36)
    dim = 2 if data_name == "OSM" else 3
    with open(json_path, 'r') as f:
        y = json.load(f)
        # bit_keys = ['lbmc', 'quilts']
        # for bit_key in bit_keys:
        command = com_template.format(dataset=dataset, scheme=y[bit_key], data_name=data_name, dim = dim, window_path=window_path)
        os.system(command)
            # print(command)



def run_pattern(data_name: str, bit_key: str):

    dataset = datasets[data_name]
    ratios = data_set_ratios[data_name]
    bits_nums = [16, 16] if data_name == "OSM" else [16, 16, 16]
    dim = 2 if data_name == "OSM" else 3
    for i, ratio in enumerate(ratios):
        # if i != 1:
        #     continue
        window_path = "./windows/app2/" + data_name + "/" + ratio_to_pattern(ratio) + ".csv"
        json_path = "./windows/app2/" + data_name + "/" + ratio_to_pattern(ratio) + ".json"
        config_path = "python/configs/" + data_name + "/" + ratio_to_pattern(ratio) + ".json"
        config = RLconfig(config_path)
        with open(json_path, 'r') as f:
            y = json.load(f)
            # bit_keys = ['lbmc', 'quilts']
            # for bit_key in bit_keys:
            command = com_template.format(dataset=dataset, scheme=y[bit_key], data_name=data_name, dim = dim, window_path=window_path)
            os.system(command)
                # print(command)
        # break


if __name__ == "__main__":
    test_dataset = sys.argv[1]
    # learn_mix_data(test_dataset)
    # run_mix(test_dataset, "quilts")
    # run_mix(test_dataset, "lbmc")
    # run_mix(test_dataset, "cbmc")

    learn_pattern_data(test_dataset)

    # run_pattern(test_dataset, "quilts")

    run_pattern(test_dataset, "lbmc")

    run_pattern(test_dataset, "cbmc")



# define config
# init, costs, iterations, learning_rate
# lbmc or bmc 