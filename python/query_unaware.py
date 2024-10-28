#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
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
from utils import bit_letters
from utils import ratio_to_pattern
from utils import Point
from utils import Window
from utils import Config
import csv

import numpy as np
import os
from operator import itemgetter, attrgetter
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(1)
random.seed(10)
plt.rc('font', family='Times New Roman', size=20)
# plt.rc('lines', linewidth=3)

# labelsize=20
# fontsize=20
# legend_fontsize=20
# lw = 4
# floder = "SIGMOD2023"
# bit_letters = ["A", "B", "C", "D", "E"]
# factor_letters = ["a", "b", "c", "d", "e"]
logger_print = True
random.seed(10)

# value_letters = ["x", "y", "z"]


class Config:
    def __init__(self, dim_length):
        self.dim_length = dim_length
        self.dim = len(dim_length)


class Point:
    def __init__(self, xs, value=0):
        self.xs = xs
        self.value = value
        self.dim = len(xs)

    def __str__(self):
        return "pos: " + " ".join(map(str, self.xs)) + " val: " + str(self.value) + "\n"

    def __repr__(self):
        return "pos: " + " ".join(map(str, self.xs)) + " val: " + str(self.value) + "\n"


class Window:
    def __init__(self, dimension_low, dimension_high, dimension_low_raw, dimension_high_raw):
        assert len(dimension_low) == len(
            dimension_high), "dimension_low and dimension_high should be same dimension"
        self.point_l = Point(dimension_low)
        self.point_h = Point(dimension_high)
        self.dimension_low = [int(_) for _ in dimension_low]
        self.dimension_high = [int(_) for _ in dimension_high]
        self.dimension_low_raw = dimension_low_raw
        self.dimension_high_raw = dimension_high_raw
        self.dim = len(dimension_low)
        self.ratio = 0

    def get_area(self):
        area = 1
        for high, low in zip(self.dimension_high, self.dimension_low):
            area *= (high - low + 1)
        return area

    def __str__(self):
        return "pl: " + str(self.point_l) + " ph:" + str(self.point_h)


def generate_a_window(unit_len, dim, ratio, dim_scalar):
    lengths = [unit_len * rat for rat in ratio]
    dimension_low = []
    dimension_high = []
    dimension_low_raw = []
    dimension_high_raw = []
#     random.seed(10)
    for i in range(dim):
        # set the random range [0, 1-dim_i_length]
        start_dim_i = random.random() * (1 - lengths[i])
        end_dim_i = start_dim_i + lengths[i]
        dimension_low.append(start_dim_i * dim_scalar[i])
        dimension_high.append(end_dim_i * dim_scalar[i])
        dimension_low_raw.append(start_dim_i)
        dimension_high_raw.append(end_dim_i)

    window = Window(dimension_low, dimension_high,
                    dimension_low_raw, dimension_high_raw)
    return window


def get_query_windows(unit_len, dim, ratios, nums, dim_scalar):
    windows = []
    for i in range(len(nums)):
        for j in range(nums[i]):
            windows.append(generate_a_window(
                unit_len, dim, ratios[i], dim_scalar))
    return windows


def get_curve_value_via_location(location, bit_distribution, bits_nums):
    res = 0
    bit_current_location = len(bit_distribution) - 1
    masks = [pow(2, bits_num - 1) for bits_num in bits_nums]
    for char in bit_distribution:
        bit_index = bit_letters.index(char)
        if masks[bit_index] & location[bit_index] != 0:
            res += pow(2, bit_current_location)
        masks[bit_index] = masks[bit_index] >> 1
        bit_current_location -= 1
    return res


def timer(fn):
    from time import perf_counter

    def inner(*args, **kwargs):
        start_time = perf_counter()
        to_execute = fn(*args, **kwargs)
        end_time = perf_counter()
        execution_time = end_time - start_time
        if logger_print:
            print('{0} costs {1:.8f}s'.format(fn.__name__, execution_time))
        return to_execute

    return inner



def random_bit_distribution(old_bit_nums=[8, 8]):
    bit_nums = copy.deepcopy(old_bit_nums)
    res = ""
    all_ = sum(old_bit_nums)
    i = all_
    dim = len(bit_nums)
    while len(res) == all_:
        for j in range(dim):
            if bit_nums[j] == 0:
                continue
            else:
                res = bit_letters[j] + res
                bit_nums[j] -= 1
    # for i in range(all_):
    #     while True:
    #         index = int(random.uniform(0, dim-1e-8))
    #         if bit_nums[index] == 0:
    #             continue
    #         else:
    #             bit_nums[index] -= 1
    #             res += bit_letters[index]
    #             break
    return res
# random_bit_distribution()


# In[16]:

def store_all_windows(windows, dim, num, pattern):
    windows_path = 'windows/' + \
        str(dim) + "/" + str(num) + "/" + str(int(pattern)) + '/windows.csv'
    f = open(windows_path, 'w')
    writer = csv.writer(f)
    rows = []
    for window in windows:
        temp = []
        temp.extend(window.dimension_low_raw)
        temp.extend(window.dimension_high_raw)
        rows.append(temp)
        writer.writerow(temp)
    f.close()


def quilts_curve_design(window, dim, bits_nums):
    bits_nums_copy = copy.deepcopy(bits_nums)
    dim_bit_num = []
    for i in range(dim):
        dim_len = window.dimension_high[i] - window.dimension_low[i]
        bit_num = math.ceil(math.log(dim_len) / math.log(2))
        dim_bit_num.append(bit_num)
    most_sig_res = ""
    for i, bit_num in enumerate(dim_bit_num):
        bits_nums_copy[i] -= bit_num
        for j in range(bit_num):
            most_sig_res = most_sig_res + bit_letters[i]

    C_least_sig_res = ""
    Z_least_sig_res = ""
    for i, bit_num in enumerate(bits_nums_copy):
        for j in range(bit_num):
            C_least_sig_res = C_least_sig_res + bit_letters[i]
    length = sum(bits_nums_copy)
    while length > 0:
        for i in range(dim):
            if bits_nums_copy[i] > 0:
                bits_nums_copy[i] -= 1
                Z_least_sig_res = Z_least_sig_res + bit_letters[i]
                length -= 1

    C_res = C_least_sig_res + most_sig_res
    Z_res = Z_least_sig_res + most_sig_res
    return C_res, Z_res


def random_bit_distribution(old_bit_nums=[8, 8]):
    bit_nums = copy.deepcopy(old_bit_nums)
    res = ""
    all_ = sum(old_bit_nums)
    dim = len(bit_nums)
    for i in range(all_):
        while True:
            index = int(random.uniform(0, dim-1e-8))
            if bit_nums[index] == 0:
                continue
            else:
                bit_nums[index] -= 1
                res += bit_letters[index]
                break
    return res


def exchange_bit(string, position):
    if position >= len(string) - 1:
        return string
    left = string[position]
    right = string[position + 1]
    return string[:position] + right + left + string[position+2:]


def bit_distribution_to_array(bit_distribution, dim):
    res = []
    for s in bit_distribution:
        res.append((bit_letters.index(s) + 1.0) / dim)
    return res

iteration = 40000
MEMORY_CAPACITY = 1000
train_gap = 5
def choose_RL(name, length):
    if name == 'dqn':
        RL = DeepQNetwork(length, length, width=128, reward_decay=0.9,
                      e_greedy=0.9, learning_rate=0.01, memory_size=MEMORY_CAPACITY, 
                      batch_size=32, e_greedy_increment=float(train_gap) / iteration)
    elif name == 'ddpg':
        REPLACEMENT = [
                dict(name='soft', tau=0.01),
                dict(name='hard', rep_iter_a=600, rep_iter_c=500)
            ][0]
        LR_A=0.01
        LR_C=0.01
        GAMMA=0.9
        EPSILON=0.1
        VAR_DECAY=.9995
        RL = DeepDeterministicPolicyGradient(length, length, 1, LR_A, LR_C, REPLACEMENT, GAMMA,
                                                EPSILON)
    return RL

def train_optimal_curve(bit_distribution, bits_nums, windows):
    iteration = 40000
    MEMORY_CAPACITY = 1000
    train_gap = 5
    length = sum(bits_nums)
    dim = len(bits_nums)
    # RL = DeepQNetwork(length, length, width=128, reward_decay=0.9,
    #                   e_greedy=0.9, learning_rate=0.01, memory_size=MEMORY_CAPACITY, 
    #                   batch_size=32, e_greedy_increment=float(train_gap) / iteration)
    RL = choose_RL("dqn", length)
    GC = GlobalCost(windows, bits_nums)
    GC.each_curve_length_to_formula()
    GC.get_factor_value_via_bit_distribution(bit_distribution)

    # GC.each_curve_length_to_formula()
    global_bit_distribution = GC.get_global_optimal_curve()

    global_cost, _ = GC.global_cost(bit_distribution)
    config = Config(bits_nums)
    LC = LocalCost(windows, bits_nums, config)
    LC.prepare_tables()
    local_cost, _ = LC.local_cost(bit_distribution)
    local_bit_distribution = LC.get_local_optimal_curve()

    pre_total_cost = global_cost * local_cost
    step = 0
    pre_local_cost = local_cost
    pre_global_cost = global_cost
    min_cost = pre_total_cost
    min_distribution = bit_distribution
    initial_cost = pre_total_cost
    all_rewards = []
    M = 5
    for i in range(M):
        while step < iteration:
            # print(bit_distribution)
            state = bit_distribution_to_array(bit_distribution, dim)
            # print(state)
            force_random = step < MEMORY_CAPACITY
            action = RL.choose_action(np.array(state), force_random)

            # print(action)
            new_bit_distribution = exchange_bit(bit_distribution, action)
            gc, _ = GC.global_cost(new_bit_distribution)
            lc, _ = LC.local_cost(new_bit_distribution)
            r_g = 0
            r_l = 0
            if gc < pre_global_cost:
                r_g = 0.5
            elif gc > pre_global_cost:
                r_g = -0.5
            
            if lc < pre_local_cost:
                r_l = 0.5
            elif lc > pre_local_cost:
                r_l = -0.5
            
            reward = r_g + r_l

            # current_total_cost = gc * lc
            # reward = (pre_total_cost - current_total_cost) / reward_gap
            if reward > 0:
                bit_distribution = new_bit_distribution
                all_rewards.append(reward)
                # all_rewards.append(current_total_cost / initial_cost)
            # all_rewards.append(current_total_cost)
            # print(reward)
            current_state = bit_distribution_to_array(new_bit_distribution, dim)
            # if pre_total_cost < min_cost:
            #     min_cost = pre_total_cost
            #     min_distribution = bit_distribution
            #     min_step = step
            pre_local_cost = lc
            pre_global_cost = gc
            # pre_total_cost = current_total_cost
            RL.store_transition(state, action, reward, current_state)
            if (step > MEMORY_CAPACITY) and (step % train_gap == 0):
                RL.learn()
            step += 1
    return min_distribution, bit_distribution, global_bit_distribution, local_bit_distribution

# def train_optimal_curve(bit_distribution, bits_nums, windows):
#     iteration = 20000
#     MEMORY_CAPACITY = 500
#     length = sum(bits_nums)
#     RL = DeepQNetwork(length, length, width=128, reward_decay=1.0, e_greedy=0.9, learning_rate=0.01, memory_size=MEMORY_CAPACITY)
#     GC = GlobalCost(windows, bits_nums)
#     # GC.each_curve_length_to_formula()
#     GC.get_factor_value_via_bit_distribution(bit_distribution)

#     # GC.each_curve_length_to_formula()
#     global_bit_distribution = GC.get_global_optimal_curve()

#     global_cost, _ = GC.global_cost(bit_distribution)
#     config = Config(bits_nums)
#     LC = LocalCost(windows, bits_nums, config)
#     LC.prepare_tables()
#     local_cost, _ = LC.local_cost(bit_distribution)
#     local_bit_distribution = LC.get_local_optimal_curve()

#     pre_total_cost = global_cost * local_cost
#     step = 0
#     reward_gap = pre_total_cost
#     # print("initial distribution", bit_distribution)
#     # print("pre_total_cost", pre_total_cost)
#     min_cost = pre_total_cost
#     min_distribution = bit_distribution
#     initial_cost = pre_total_cost
#     min_step = 0
#     all_rewards = []
#     while step < iteration:
#         # print(bit_distribution)
#         state = bit_distribution_to_array(bit_distribution)
#         # print(state)
#         force_random = step < MEMORY_CAPACITY
#         action = RL.choose_action(np.array(state), force_random)

#         # print(action)
#         new_bit_distribution = exchange_bit(bit_distribution, action)
#         gc, _ = GC.global_cost(new_bit_distribution)
#         lc, _ = LC.local_cost(new_bit_distribution)
#         current_total_cost = gc * lc
#         reward = (pre_total_cost - current_total_cost) / reward_gap
#         if reward >= 0:
#             bit_distribution = new_bit_distribution
#             all_rewards.append(current_total_cost / initial_cost)
#         # all_rewards.append(current_total_cost)

#         # print(reward)
#         current_state = bit_distribution_to_array(bit_distribution)
#         if pre_total_cost < min_cost:
#             min_cost = pre_total_cost
#             min_distribution = bit_distribution
#             min_step = step
#         pre_total_cost = current_total_cost
#         if reward != 0:
#             RL.store_transition(state, action, reward, current_state)
#         if (step > 200) and (step % 5 == 0):
#             RL.learn()
#         step += 1
#     # print("final distribution", bit_distribution, current_total_cost)
#     # print("min_distribution", min_distribution, min_cost)
#     # print("min_step", min_step)
#     return min_distribution, bit_distribution, global_bit_distribution, local_bit_distribution

def read_data(file_name, dim=2):
    dim_smallest = []
    dim_largest = []
    for i in range(dim):
        dim_smallest.append(sys.float_info.max)
        dim_largest.append(sys.float_info.min)
    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        for i, row in enumerate(reader):
            for j in range(dim):
                dj = float(row[j])
                if dj < dim_smallest[j]:
                    dim_smallest[j] = dj
                if dj > dim_largest[j]:
                    dim_largest[j] = dj
    print(dim_smallest)    
    print(dim_largest)
    return dim_smallest, dim_largest

def partition(granualitys, dim_smallest, dim_largest):
    dim = len(dim_smallest)
    dim_unit_len = []
    for i in range(dim):
        dim_unit_len.append((dim_largest[i] - dim_smallest[i]) / granualitys[i])
    return dim_unit_len

def gen(current, granualitys, index, dim_unit_len, dim_smallest, windows, sides):
    if len(current) == len(granualitys):
        dimension_low = []
        dimension_low_raw = []
        dimension_high = []
        dimension_high_raw = []
        for i in range(len(current)):
            dimension_low.append(current[i] * sides[i])
            dimension_low_raw.append(current[i])
            dimension_high.append((current[i] + dim_unit_len[i]) * sides[i])
            dimension_high_raw.append(current[i] + dim_unit_len[i])
        window = Window(dimension_low, dimension_high, dimension_low_raw, dimension_high_raw)
        windows.append(window)
    else:
        for i in range(granualitys[index]):
            temp = copy.deepcopy(current)
            temp.append(dim_smallest[index] + i * dim_unit_len[index])
            gen(temp, granualitys, index + 1, dim_unit_len, dim_smallest, windows, sides)

def count_window_ratio(file_name, windows, dim_smallest, dim_unit_len, scale = [100, 1]):
    window_ratios = [0 for i in range(len(windows))]
    dim = len(dim_unit_len)
    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        for i, row in enumerate(reader):
            res = 0
            for j in range(dim):
                dj = float(row[j]) - dim_smallest[j]
                index = math.floor(dj / dim_unit_len[j])
                res += scale[j] * index
                
            if res < 0:
                res = 0
            if res >= len(window_ratios):
                res = len(window_ratios) - 1
            window_ratios[res] += 1
    i += 1
    for j in range(len(window_ratios)):
        windows[j].ratio = window_ratios[j] / i
    candidate_windows = []
    for window in windows:
        if window.ratio > 0:
            candidate_windows.append(window)
    return candidate_windows

def gen_windows(file_name, dim=2, granuality=100, bits_num=[32, 32]):
    sides = [pow(2, bit_num) for bit_num in bits_num]
    granualitys = [granuality for i in range(dim)]

    dim_smallest, dim_largest = read_data(file_name, dim)

    dim_unit_len = partition(granualitys, dim_smallest, dim_largest)
    windows = []
    dim_smallest = [0 for i in range(dim)]
    gen([], granualitys, 0, dim_unit_len, dim_smallest, windows, sides)
    return windows, dim_smallest, dim_unit_len


def store_granuality_windows(windows, dim, granuality):
    windows_path = 'windows/' + str(dim) + '/granuality/' + str(granuality) + '/granuality_windows.csv'
    f = open(windows_path, 'w')
    writer = csv.writer(f)
    rows = []
    for window in windows:
        temp = []
        temp.extend(window.dimension_low_raw)
        temp.extend(window.dimension_high_raw)
        rows.append(temp)
        writer.writerow(temp)
    f.close()

def run_each(dim, granualitys, file_name, bits_nums):
    for granuality in granualitys:
        json_path = 'windows/' + str(dim) + '/granuality/' + str(granuality) + '/bit_distributions.json'
        # print(json_path)
        # continue
        scale = []
        for i in range(dim):
            scale.append(int(pow(granuality, dim - i - 1)))
        print(scale)
        windows, dim_smallest, dim_unit_len = gen_windows(file_name, dim=dim, granuality=granuality, bits_num=bits_nums)
        windows = count_window_ratio(file_name, windows, dim_smallest, dim_unit_len, scale)
        windows = sorted(windows, key=attrgetter('ratio'), reverse=True)
        if (len(windows) > 1000):
            windows = windows[0:1000]
        store_granuality_windows(windows, dim, granuality)

        init_bit_distribution = random_bit_distribution(bits_nums)
        min_distribution, rl_bit_distribution, global_bit_distribution, local_bit_distribution = train_optimal_curve(
            init_bit_distribution, bits_nums, windows)
        quilts_bit_distribution = quilts_curve_design(
            windows[0], dim, bits_nums)
        data = {'md': min_distribution, "rld": rl_bit_distribution, "quilts_Z": quilts_bit_distribution[1]}
        with open(json_path, 'w') as outfile:
            json.dump(data, outfile)

def run_all():
    granualitys = [20, 40, 60, 80, 100]
    file_name = "/home/research/datasets/OSM_100000000_1_2_.csv"
    run_each(2, granualitys, file_name, [16, 16])
    file_name = "/home/research/datasets/yellow_tripdata_2015_normalized_3d.csv"
    run_each(3, granualitys, file_name, [16, 16, 16])


run_all()