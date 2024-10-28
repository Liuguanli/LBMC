#!/usr/bin/env python
# coding: utf-8


# from threading import enumerate
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
import os.path
import utils
from utils import ratio_to_pattern
from utils import Point
from utils import Window
from utils import Config
import numpy as np
import os
import csv
from time import perf_counter

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
floder = "SIGMOD2023"
# random.seed(10)
bit_letters = ["A", "B", "C", "D", "E"]
factor_letters = ["a", "b", "c", "d", "e"]
logger_print = True
# value_letters = ["x", "y", "z"]



# In[3]:


# class Point:
#     def __init__(self, xs, value=0):
#         self.xs = xs
#         self.value = value
#         self.dim = len(xs)

#     def __str__(self):
#         return "pos: " + " ".join(map(str, self.xs)) + " val: " + str(self.value) + "\n"

#     def __repr__(self):
#         return "pos: " + " ".join(map(str, self.xs)) + " val: " + str(self.value) + "\n"


# class Window:
#     def __init__(self, dimension_low, dimension_high, dimension_low_raw, dimension_high_raw):
#         assert len(dimension_low) == len(
#             dimension_high), "dimension_low and dimension_high should be same dimension"
#         self.point_l = Point(dimension_low)
#         self.point_h = Point(dimension_high)
#         self.dimension_low = [int(_) for _ in dimension_low]
#         self.dimension_high = [int(_) for _ in dimension_high]
#         self.dimension_low_raw = dimension_low_raw
#         self.dimension_high_raw = dimension_high_raw
#         self.dim = len(dimension_low)
#         self.ratio = 1

#     def get_area(self):
#         area = 1
#         for high, low in zip(self.dimension_high, self.dimension_low):
#             area *= (high - low + 1)
#         return area

#     def __str__(self):
#         return "pl: " + str(self.point_l) + " ph:" + str(self.point_h)


# def generate_a_window(unit_len, dim, ratio, dim_scalar):
#     lengths = []
#     for i in range(dim):
#         lengths.append(unit_len[i] * ratio[i])
#     # lengths = [unit_len * rat for rat in ratio]
#     dimension_low = []
#     dimension_high = []
#     dimension_low_raw = []
#     dimension_high_raw = []
# #     random.seed(10)
#     for i in range(dim):
#         # set the random range [0, 1-dim_i_length]
#         start_dim_i = random.random() * (1 - lengths[i])
#         end_dim_i = start_dim_i + lengths[i]
#         dimension_low.append(math.floor(start_dim_i * dim_scalar[i]))
#         dimension_high.append(math.floor(end_dim_i * dim_scalar[i]))
#         dimension_low_raw.append(start_dim_i)
#         dimension_high_raw.append(end_dim_i)

#     window = Window(dimension_low, dimension_high,
#                     dimension_low_raw, dimension_high_raw)
#     return window


# def ratio_to_pattern(ratios):
#     res = ""
#     for i in range(len(ratios) - 1):
#         res += str(int(ratios[i])) + "_"
#     res += str(int(ratios[-1]))
#     return res

def generate_a_window(unit_len, dim, ratio, dim_scalar):
    lengths = []
    for i in range(dim):
        lengths.append(unit_len[i] * ratio[i])
    # lengths = [unit_len * rat for rat in ratio]
    dimension_low = []
    dimension_high = []
    dimension_low_raw = []
    dimension_high_raw = []
#     random.seed(10)
    for i in range(dim):
        # set the random range [0, 1-dim_i_length]
        start_dim_i = pow(random.random()) * (1 - lengths[i])
        end_dim_i = start_dim_i + lengths[i]
        dimension_low.append(math.floor(start_dim_i * dim_scalar[i]))
        dimension_high.append(math.floor(end_dim_i * dim_scalar[i]))
        dimension_low_raw.append(start_dim_i)
        dimension_high_raw.append(end_dim_i)

    window = Window(dimension_low, dimension_high,
                    dimension_low_raw, dimension_high_raw)
    return window

def get_query_windows(unit_len, dim, ratios, num, bits_nums, query_size):
    windows = []
    pattern = ratios[-1]
    windows_path = 'windows/' + \
        str(dim) + "/" + str(query_size) + "/" + str(num) + \
        "/" + ratio_to_pattern(ratios) + '/windows.csv'
    dim_scalar = [(pow(2, l) - 1) for l in bits_nums]

    if os.path.exists(windows_path):
        # read
        with open(windows_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
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
    else:
        for i in range(num):
            windows.append(generate_a_window(
                unit_len, dim, ratios, dim_scalar))
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

    return windows


def get_mix_windows(unit_len, dim, ratios, num, bits_nums, query_size, mix_index):
    each_num = int(num / len(ratios))
    windows = []
    dim_scalar = [(pow(2, l) - 1) for l in bits_nums]
    for i, ratio in enumerate(ratios):
        temp_windows = get_query_windows(
            unit_len, dim, ratio, num, bits_nums, query_size)
        windows.extend(temp_windows[i * each_num: (i + 1) * each_num])
        # for j in range(each_num):
        #     windows.append(generate_a_window(unit_len, dim, ratio[::-1], dim_scalar))

    windows_path = 'windows/' + \
        str(dim) + '/mix/' + str(mix_index) + '/windows.csv'
    # f = open(windows_path, 'w')
    # writer = csv.writer(f)
    # rows = []
    # for window in windows:
    #     temp = []
    #     temp.extend(window.dimension_low_raw)
    #     temp.extend(window.dimension_high_raw)
    #     rows.append(temp)
    #     writer.writerow(temp)
    # f.close()
    write_windows(windows_path, windows)
    return windows


def write_windows(windows_path, windows):
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


def get_random_size_windows():
    pass


def get_skewed_windows():
    pass


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
    while len(res) != all_:
        for j in range(dim - 1, -1, -1):
            if bit_nums[j] == 0:
                continue
            else:
                res = res + bit_letters[j]
                bit_nums[j] -= 1
    return res
    # for i in range(all_):
    #     while True:
    #         index = int(random.uniform(0, dim-1e-8))
    #         if bit_nums[index] == 0:
    #             continue
    #         else:
    #             bit_nums[index] -= 1
    #             res += bit_letters[index]
    #             break


def quilts_curve_design_multi(windows, frequent_window, dim, bits_nums):
    bits_nums_copy = copy.deepcopy(bits_nums)
    l_i = copy.deepcopy(bits_nums)
    u_i = [0 for i in range(dim)]
    d_i = []
    for i in range(dim):
        dim_len = frequent_window.dimension_high[i] - \
            frequent_window.dimension_low[i]
        bit_num = math.ceil(math.log(dim_len) / math.log(2))
        d_i.append(bit_num)
    for window in windows:
        for i in range(dim):
            dim_len = window.dimension_high[i] - window.dimension_low[i]
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
    while length > 0:
        for i in range(dim):
            if u_i[i] > 0:
                bits_nums_copy[i] -= 1
                u_i[i] -= 1
                # least_sig_res = bit_letters[i] + least_sig_res
                least_sig_res += bit_letters[i]
                length -= 1
    Z_least_sig_res = ""
    C_least_sig_res = ""
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


def quilts_curve_design(window, dim, bits_nums):
    bits_nums_copy = copy.deepcopy(bits_nums)
    dim_bit_num = []
    for i in range(dim):
        dim_len = window.dimension_high[i] - window.dimension_low[i]
        if dim_len == 0:
            dim_bit_num.append(0)
        else:
            bit_num = math.ceil(math.log(dim_len) / math.log(2))
            dim_bit_num.append(bit_num)
    most_sig_res = ""
    for i, bit_num in enumerate(dim_bit_num):
        bits_nums_copy[i] -= bit_num
        for j in range(bit_num):
            # most_sig_res += bit_letters[i]
            most_sig_res = bit_letters[i] + most_sig_res

    # length = sum(dim_bit_num)
    # while length > 0:
    #     for i in range(dim):
    #         if dim_bit_num[i] > 0:
    #             bits_nums_copy[i] -= 1
    #             dim_bit_num[i] -= 1
    #             # most_sig_res = bit_letters[i] + most_sig_res
    #             most_sig_res += bit_letters[i]
    #             length -= 1
    C_least_sig_res = ""
    Z_least_sig_res = ""
    for i, bit_num in enumerate(bits_nums_copy):
        for j in range(bit_num):
            C_least_sig_res += bit_letters[i]

    length = sum(bits_nums_copy)
    while length > 0:
        for i in range(dim):
            if bits_nums_copy[i] > 0:
                bits_nums_copy[i] -= 1
                Z_least_sig_res = bit_letters[i] + Z_least_sig_res
                # Z_least_sig_res += bit_letters[i]
                length -= 1

    C_res = C_least_sig_res + most_sig_res
    Z_res = Z_least_sig_res + most_sig_res
    return C_res, Z_res


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

# def bit_distribution_to_array(bit_distribution, dim):
#     res = []
#     for s in bit_distribution:
#         res.append((bit_letters.index(s) + 1.0) / dim)
#     return res


iteration = 40000
MEMORY_CAPACITY = 1000
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

def train_optimal_curve(initial_bit_distribution, bits_nums, windows):
    length = sum(bits_nums)
    dim = len(bits_nums)
    bit_distribution = initial_bit_distribution
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
    pre_local_cost = local_cost
    pre_global_cost = global_cost
    min_cost = pre_total_cost
    min_distribution = bit_distribution
    initial_cost = pre_total_cost
    all_rewards = []
    step = 0
    M = 1
    for i in range(M):
        bit_distribution = initial_bit_distribution
        while step < iteration:
            # print(bit_distribution)
            state = bit_distribution_to_array(bit_distribution, dim, None)
            # print(state)
            force_random = step < MEMORY_CAPACITY
            action = RL.choose_action(np.array(state), force_random)

            exchange, new_bit_distribution = exchange_bit(bit_distribution, action)
            # if not exchange:
            #     # step += 1
            #     continue
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
            # print(reward)
            bit_distribution = new_bit_distribution

            if reward > 0:
                result_bit_distribution = new_bit_distribution
                all_rewards.append(reward)
                # all_rewards.append(current_total_cost / initial_cost)
            # all_rewards.append(current_total_cost)
            # print(reward)
            current_state = bit_distribution_to_array(new_bit_distribution, dim, None)
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
    return min_distribution, result_bit_distribution, global_bit_distribution, local_bit_distribution


def train_optimal_curve_legacy(bit_distribution, bits_nums, windows, encoding="one_hot"):
    start_time = perf_counter()
    
    iteration = 40000
    MEMORY_CAPACITY = 1000
    dim = len(bits_nums)
    length = sum(bits_nums)
    if encoding is None:
        scale = 1
    else:
        scale = dim if encoding == "one_hot" else 1
    RL = choose_RL("dqn", length * scale)
    # RL = DeepQNetwork(length, length, width=64, reward_decay=0.9,
    #                   e_greedy=0.9, learning_rate=0.01, memory_size=MEMORY_CAPACITY,
    #                   batch_size=32, e_greedy_increment=float(train_gap) / iteration)
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
    # local_bit_distribution = LC.get_local_optimal_curve()

    pre_total_cost = global_cost * local_cost
    step = 0
    reward_gap = pre_total_cost
    # print("initial distribution", bit_distribution)
    # print("pre_total_cost", pre_total_cost)
    min_cost = pre_total_cost
    min_distribution = bit_distribution
    initial_cost = pre_total_cost
    min_step = 0
    all_rewards = []
    result_bit_distribution = bit_distribution

    while step < iteration:
        # print(bit_distribution)
        state = bit_distribution_to_array(bit_distribution, dim)
        # print(state)
        force_random = step < MEMORY_CAPACITY
        action = RL.choose_action(np.array(state), force_random)

        # print(action)
        exchange, new_bit_distribution = exchange_bit(bit_distribution, action)
        gc, _ = GC.global_cost(new_bit_distribution)
        lc, _ = LC.local_cost(new_bit_distribution)
        current_total_cost = gc * lc
        reward = (pre_total_cost - current_total_cost) / reward_gap
        bit_distribution = new_bit_distribution

        if reward >= 0:
            result_bit_distribution = new_bit_distribution
            all_rewards.append(current_total_cost / initial_cost)
        # all_rewards.append(current_total_cost)

        # print(reward)
        current_state = bit_distribution_to_array(bit_distribution, dim)
        if pre_total_cost < min_cost:
            min_cost = pre_total_cost
            min_distribution = bit_distribution
            min_step = step
        pre_total_cost = current_total_cost
        if reward != 0:
            RL.store_transition(state, action, reward, current_state)
        # print("memorize size:", RL.memory_size, "reward:", reward)
        if (step > 200 and RL.memory_counter > 0) and (step % 5 == 0):
            RL.learn()
        step += 1
    # print("final distribution", bit_distribution, current_total_cost)
    # print("min_distribution", min_distribution, min_cost)
    # print("min_step", min_step)
    # bit_distribution = min_distribution # for 2d case not for 2d case
    end_time = perf_counter()
    print("training time:", end_time-start_time)
    return min_distribution, result_bit_distribution, global_bit_distribution


def get_bit_dis_comb(res, current, bit_nums, total_length):
    if (len(current) == total_length):
        res.append(current)
    else:
        for i in range(len(bit_nums)):
            if bit_nums[i] > 0:
                bit_nums[i] -= 1
                get_bit_dis_comb(
                    res, current+bit_letters[i], bit_nums, total_length)
                bit_nums[i] += 1


def get_opt_curve(bits_nums, num, GC):
    res = []
    total_length = sum(bits_nums)
    get_bit_dis_comb(res, "", bits_nums, total_length)
    bits_sum = sum(bits_nums)
    min_cost = num * pow(2, bits_sum - 1)
    min_bd = []
    for bd in res:
        GC.get_factor_value_via_bit_distribution(bd)
        temp, _ = GC.global_cost(bd)
        if min_cost > temp:
            min_cost = temp
            min_bd = []
            min_bd.append(bd)
        elif min_cost == temp:
            min_bd.append(bd)
    print(min_bd, "minimal global cost", min_cost)


def test_pattern(dim, num, ratios, unit_len, bits_nums, index):
    for ratio in ratios:
        json_path = 'windows/' + \
            str(dim) + "/" + str(index) + "/" + str(num) + "/" + \
                ratio_to_pattern(ratio) + '/bit_distributions.json'
        print(json_path)
        # continue
        windows = get_query_windows(
            unit_len, dim, ratio, num, bits_nums, index)
        quilts_bit_distribution = quilts_curve_design(
            windows[0], dim, bits_nums)
        init_bit_distribution = random_bit_distribution(bits_nums)
        # init_bit_distribution = quilts_bit_distribution[1]
        min_distribution, rl_bit_distribution, global_bit_distribution = train_optimal_curve(
            init_bit_distribution, bits_nums, windows)
        quilts_bit_distribution = quilts_curve_design(
            windows[0], dim, bits_nums)
        data = {'md': min_distribution, "rld": rl_bit_distribution,
            "quilts_Z": quilts_bit_distribution[1]}

        with open(json_path, 'w') as outfile:
            json.dump(data, outfile)


def test_mix(unit_len, dim, ratios, num, bits_nums, unit_len_scale, mix_index):
    json_path = 'windows/' + str(dim) + '/mix/' + \
                                 str(mix_index) + '/bit_distributions.json'
    print(json_path)
    windows = get_mix_windows(unit_len, dim, ratios,
                              num, bits_nums, unit_len_scale, mix_index)
    init_bit_distribution = random_bit_distribution(bits_nums)
    quilts_bit_distribution = quilts_curve_design_multi(
        windows, windows[0], dim, bits_nums)
    init_bit_distribution = quilts_bit_distribution[1]

    min_distribution, rl_bit_distribution, global_bit_distribution = train_optimal_curve_legacy(
        init_bit_distribution, bits_nums, windows)
    

    data = {'md': min_distribution, "rld": rl_bit_distribution,
        "quilts_Z": quilts_bit_distribution[1]}
    with open(json_path, 'w') as outfile:
        json.dump(data, outfile)


def test_mix_by_windows(windows, dim, bits_nums, mix_index):
    json_path = 'windows/' + str(dim) + '/mix/' + \
                                 str(mix_index) + '/bit_distributions.json'
    print(json_path)
    init_bit_distribution = random_bit_distribution(bits_nums)
    min_distribution, rl_bit_distribution, global_bit_distribution = train_optimal_curve_legacy(
        init_bit_distribution, bits_nums, windows)
    quilts_bit_distribution = quilts_curve_design_multi(
        windows, windows[0], dim, bits_nums)
    data = {'md': min_distribution, "rld": rl_bit_distribution,
            "quilts_Z": quilts_bit_distribution[1]}

    with open(json_path, 'w') as outfile:
        json.dump(data, outfile)


def test_3d_mix():
    bits_nums = [16, 16, 16]
    unit_len = [0.01, 0.01, 0.001]
    dim = 3
    num = 1000
    ratios = [[1.0, 1.0, 1.0], [1.0, 1.0, 4.0], [
        1.0, 1.0, 16.0], [1.0, 1.0, 64.0], [1.0, 1.0, 256.0]]
    test_mix(unit_len, dim, ratios, num, bits_nums, 1, 1)
    query_sizes = [1, 2, 3, 4, 5]
    windows = []
    for query_size in query_sizes:
        windows.extend(get_query_windows(unit_len, dim, [
                       1.0, 1.0, 16.0], num, bits_nums, query_size)[-201:-1])
    test_mix_by_windows(windows, dim, bits_nums, 2)
    windows_path = 'windows/3/mix/2/windows.csv'
    write_windows(windows_path, windows)
    windows = []
    for query_size in query_sizes:
        for ratio in ratios:
            windows.extend(get_query_windows(
                unit_len, dim, ratio, num, bits_nums, query_size)[-51:-1])
    test_mix_by_windows(windows, dim, bits_nums, 3)
    windows_path = 'windows/3/mix/3/windows.csv'
    write_windows(windows_path, windows)


def test_2d_mix():
    delta = 0.004
    unit_len = [delta, delta]
    bits_nums = [16, 16]
    dim = 2
    query_num = 1000

    ratios = [[1.0, 1.0], [1.0, 4.0], [1.0, 16.0], [1.0, 64.0], [1.0, 256.0]]
    test_mix(unit_len, dim, ratios, query_num, bits_nums, 1, 1)
    ratios = [[1.0, 256.0], [2, 128], [4, 64], [8, 32], [16, 16]]
    test_mix(unit_len, dim, ratios, query_num, bits_nums, 1, 2)
    ratios = [[128.0, 2.0], [256.0, 1.0], [
        4, 64], [1.0, 256.0],  [2, 128],  [64, 4]]
    test_mix(unit_len, dim, ratios, query_num, bits_nums, 1, 3)


def test_2d_pattern_size():
    scales = [0, 1, 2, 3, 4]
    bits_nums = [16, 16]
    ratios = [[1.0, 1.0], [1.0, 4.0], [1.0, 16.0], [1.0, 64.0], [1.0, 256.0]]
    delta = 0.001
    query_num = 1000
    dim = 2
    # for index, scale in enumerate(scales):
    #     unit_len = [delta * math.pow(2, scale), delta * math.pow(2, scale)]
    #     test_pattern(dim, query_num, ratios, unit_len, bits_nums, index + 1)
    unit_len = [delta * math.pow(2, 2), delta * math.pow(2, 2)]
    # ratios = [[1, 256], [2, 128], [4, 64], [8.0, 32.0], [16, 16], [32.0, 8.0], [64.0, 4.0], [128.0, 2.0], [256, 1]]
    ratios = [[4, 64], [16, 16], [64.0, 4.0], [256, 1]]
    test_pattern(dim, query_num, ratios, unit_len, bits_nums, 3)


def test_3d_pattern_size():
    scales = [0, 1, 2, 3, 4]
    bits_nums = [16, 16, 16]
    query_num = 1000
    ratios = [[1.0, 1.0, 1.0], [1.0, 1.0, 4.0], [
        1.0, 1.0, 16.0], [1.0, 1.0, 64.0], [1.0, 1.0, 256.0]]
    # ratios = [[1.0, 1.0, 1.0]]
    delta = 0.01
    dim = 3
    for index, scale in enumerate(scales):
        unit_len = [delta * math.pow(2, scale),
                                     delta * math.pow(2, scale), delta * 0.1]
        test_pattern(dim, query_num, ratios, unit_len, bits_nums, index + 1)
        # break


def test_tpc():
    bits_nums = [12, 16, 4]
    dim_scalar = [(pow(2, l) - 1) for l in bits_nums]
    dim = 3
    # dirs = ['sql1', 'sql3', 'sql6', 'sql7', 'sql14', 'sql15', 'sql17']
    dirs = ['sql1', 'sql3', 'sql6', 'sql7', 'sql17']
    dirs = ['sql6', 'sql17']

# AAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBBCCCC 31
# AAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBCCCC 14
# "AAAAAAAAAACACACACBABABABABABABABBBBBBB" 11
# AAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBBCCCC 1276
# AAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBBCCCC 2132
    for d in dirs:
        windows = []
        windows_path = 'windows/tpch/' + d + '/windows.csv'
        if d == "sql1":
            init_bit_distribution = "BBBBBABABABABABABABABABABABACCCC"
            for i in range(4):
                window = Window([0.01 * dim_scalar[0], 1, 0.25 * i * dim_scalar[2]], [0.02 * dim_scalar[0], dim_scalar[1], 0.25 * (i + 1) * dim_scalar[2]], [0.01, 0, 0.25 * i], [0.02, 1, 0.25 * (i + 1)])
                windows.append(window)
        elif d == "sql3":
            init_bit_distribution = "BBBBBABABABABABABABABABABABACCCC"
            for i in range(4):
                window = Window([0.98 * dim_scalar[0], 1, 0.25 * i * dim_scalar[2]], [0.99 *dim_scalar[0], dim_scalar[1], 0.25 * (i + 1 ) * dim_scalar[2]], [0.98, 0, 0.25 * i], [0.99, 1, 0.25 * (i + 1)])
                windows.append(window)
        elif d == "sql6":
            init_bit_distribution = "BBBBBABABABABABABABABABABABACCCC"
            for i in range(4):
                for j in range(100):
                    window = Window([1, 0.01 * j * dim_scalar[1], 0.25 * i * dim_scalar[2]], [0.1 * dim_scalar[0], 0.01 * (j + 1) * dim_scalar[1], 0.25 * (i + 1) * dim_scalar[2]], [0, 0.01 * j, 0.25 * i], [0.1 * 1, 0.01 * (j + 1), 0.25 * (i + 1)])
                    windows.append(window) 
        elif d == "sql7":
            init_bit_distribution = "BBBBBABABABABABABABABABABABACCCC"
            for i in range(1000):
                window = Window([0.001 * i * dim_scalar[0], 1, 1], [0.001 * (i + 1) * dim_scalar[0], dim_scalar[1], dim_scalar[2]], [0.001 * i, 0, 0], [0.001 * (i + 1), 1, 1])
                windows.append(window) 
        elif d == "sql17":
            init_bit_distribution = "AAAAAAABABABACCCCBBBBBBBBBBBBBAA"
            for i in range(100):
                window = Window([0.001 * i * dim_scalar[0], 1, 1], [0.001 * (i + 1) * dim_scalar[0], 0.1 * dim_scalar[1], dim_scalar[2]], [0.001 * i, 0, 0], [0.001 * (i + 1), 0.1, 1])
                windows.append(window)
  
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
    
        json_path = 'windows/tpch/' + d + '/bit_distributions.json'
        # init_bit_distribution = random_bit_distribution(bits_nums)
        # print("z", init_bit_distribution)
        
        # print("quilts", init_bit_distribution)

        quilts_bit_distribution = quilts_curve_design(windows[0], dim, bits_nums)
        init_bit_distribution = quilts_bit_distribution[1]
        min_distribution, rl_bit_distribution, global_bit_distribution = train_optimal_curve(
            init_bit_distribution, bits_nums, windows)

        data = {'md': min_distribution, "rld": rl_bit_distribution,
            "quilts_Z": quilts_bit_distribution[1]}
        with open(json_path, 'w') as outfile:
            json.dump(data, outfile)





def test_3d():
    # test_3d_pattern_size()
    test_3d_mix()

def test_2d():
    test_2d_pattern_size()
    test_2d_mix()

test_2d()
# test_3d()

# test_tpc()