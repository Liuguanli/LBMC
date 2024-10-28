
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

import utils

import numpy as np
import os

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
random.seed(10)
bit_letters = ["A", "B", "C", "D", "E"]
factor_letters = ["a", "b", "c", "d", "e"]
logger_print = True
# value_letters = ["x", "y", "z"]


# In[2]:


class Config:
    def __init__(self, dim_length):
        self.dim_length = dim_length
        self.dim = len(dim_length)


# In[3]:


class Point:
    def __init__(self, xs, value=0):
        self.xs = xs
        self.value = value
        self.dim = len(xs)

    def __str__(self):
        return "pos: " + " ".join(map(str, self.xs)) + " val: " + str(self.value) + "\n"

    def __repr__(self):
        return "pos: " + " ".join(map(str, self.xs)) + " val: " + str(self.value) + "\n"


# In[4]:


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
#         self.dimension_low = [math.ceil(_) for _ in dimension_low]
#         self.dimension_high = [math.floor(_) for _ in dimension_high]
#         self.dimension_low = [math.ceil(_ - 0.5) for _ in dimension_low]
#         self.dimension_high = [math.floor(_ - 0.5) for _ in dimension_high]
        self.dim = len(dimension_low)
        self.ratio = 1

    def get_area(self):
        area = 1
        for high, low in zip(self.dimension_high, self.dimension_low):
            area *= (high - low + 1)
        return area

    def __str__(self):
        return "pl: " + str(self.point_l) + " ph:" + str(self.point_h)


# In[5]:


class WindowInfo:
    def __init__(self, window, config):
        self.patterns_num = {}
        self.gaps_num = {}
        for i in range(config.dim):
            dim_low = window.dimension_low[i]
            dim_high = window.dimension_high[i]
            dim_low_copy = dim_low
            dim_high_copy = dim_high
            total_pattern_num = 0
            for bit_num in range(config.dim_length[i] + 1):
                if bit_num == 0:
                    # TODO dim_high - dim_low + 1 ro dim_high - dim_low
                    self.patterns_num[bit_letters[i] +
                                      str(bit_num)] = dim_high - dim_low + 1
                    self.gaps_num[bit_letters[i] +
                                  str(bit_num)] = dim_high - dim_low + 1

                else:
                    gap = int(dim_high - dim_low) - 1
                    pattern_range = int(pow(2, bit_num))
                    pattern_num = int(gap / pattern_range)
                    temp_dim_low = dim_low + pattern_num * pattern_range
                    if int((temp_dim_low + pattern_range / 2) / pattern_range) != int((dim_high + pattern_range / 2) / pattern_range):
                        pattern_num += 1
                    self.patterns_num[bit_letters[i] +
                                      str(bit_num)] = pattern_num
#                     gap = int(dim_high - dim_low) + 1
                    gap_num = 0
                    gap = dim_high_copy - dim_low_copy + 1
                    if bit_num >= 1 and pattern_range <= gap:
                        gap_num = int((dim_high_copy+1)/pattern_range) - \
                            math.ceil(dim_low_copy/pattern_range)
                    # gap_num
                    self.gaps_num[bit_letters[i]+str(bit_num)] = gap_num

                    total_pattern_num += pattern_num
#             print(self.patterns_num, dim_high, dim_low)
            assert total_pattern_num == dim_high - dim_low, "wrong pattern_num calculation" + \
                str(total_pattern_num) + " " + str(dim_high - dim_low)
#         print(self.patterns_num)


# In[6]:


class TableCellEachWindow:
    def __init__(self, column_num, row_num, config, dim_index):
        self.column_num = column_num
        self.row_num = row_num
        self.dim_index = dim_index
        self.config = config
        self.key_num = {}
        self.combinations = []

    def gen_all_keys(self, window_info):
        key_letters = [bit_letters[i]
                       for i in range(self.config.dim) if i is not self.dim_index]

        candidate_dims = [i for i in range(
            self.config.dim) if i is not self.dim_index]
        self.combination(0, candidate_dims, [], 0)
#         print(self.combinations)
#         print(self.row_num)
        if self.row_num > 1:
            for comb in self.combinations:
                key = ""
                num = 1
                for index, dim_index in enumerate(candidate_dims):
                    temp_key = bit_letters[dim_index] + str(comb[index])
                    key += temp_key
#                     if comb[index] == 0:
#                         continue
                    num *= window_info.gaps_num[temp_key]
                    # if comb[index] = 0 , this is the length of that dimension,
                    # if comb[index] != 0, this is the number of x[i]
                self.key_num[key] = num * \
                    window_info.patterns_num[bit_letters[self.dim_index] +
                                             str(self.column_num)]
        else:
            for comb in self.combinations:
                key = ""
                num = 1
                for index, dim_index in enumerate(candidate_dims):
                    temp_key = bit_letters[dim_index] + str(comb[index])
                    key += temp_key
                    num *= window_info.patterns_num[temp_key]
                    # if comb[index] = 0 , this is the length of that dimension,
                    # if comb[index] != 0, this is the number of x[i]
                self.key_num[key] = num * \
                    window_info.patterns_num[bit_letters[self.dim_index] +
                                             str(self.column_num)]
#                 print("表中的key:", key, "边的长度", bit_letters[self.dim_index] + str(self.column_num), window_info.patterns_num[bit_letters[self.dim_index] + str(self.column_num)])
    #             self.key_num[key] = num
    #         print("TableCellEachWindow", self.key_num)

    def combination(self, current_index, candidate_dims, temp, current_length):
        if current_index == len(candidate_dims):
            if self.row_num == current_length:
                i = 0
                for temp_dim_len in temp:
                    if i == self.dim_index:
                        i += 1
                    if self.config.dim_length[i] < temp_dim_len:
                        #                         print(self.config.dim_length, temp)
                        return
                    i += 1

                self.combinations.append(temp)
#                 print(temp)

        else:
            start = min(self.row_num - current_length,
                        self.config.dim_length[current_index])
#             print("start:", start, "self.row_num", self.row_num, "current_length", current_length,
#                  "self.config.dim_length[current_index]", self.config.dim_length[current_index],
#                  "self.dim_index", self.dim_index, "current_index", current_index)
# start: 7 self.row_num 9 current_length 0 self.config.dim_length[current_index] 7 self.dim_index 0 current_index 0
# start: 9 self.row_num 9 current_length 0 self.config.dim_length[current_index] 9 self.dim_index 0 current_index 1
# start: 8 self.row_num 9 current_length 1 self.config.dim_length[current_index] 9 self.dim_index 0 current_index 1
# start: 7 self.row_num 9 current_length 2 self.config.dim_length[current_index] 9 self.dim_index 0 current_index 1
# start: 6 self.row_num 9 current_length 3 self.config.dim_length[current_index] 9 self.dim_index 0 current_index 1
# start: 5 self.row_num 9 current_length 4 self.config.dim_length[current_index] 9 self.dim_index 0 current_index 1
# start: 4 self.row_num 9 current_length 5 self.config.dim_length[current_index] 9 self.dim_index 0 current_index 1
# start: 3 self.row_num 9 current_length 6 self.config.dim_length[current_index] 9 self.dim_index 0 current_index 1
# start: 2 self.row_num 9 current_length 7 self.config.dim_length[current_index] 9 self.dim_index 0 current_index 1

            start = self.row_num - current_length
            for i in range(start + 1):
                if (self.config.dim_length[current_index] < i):
                    continue
                temp_copy = copy.deepcopy(temp)
                temp_copy.append(i)
#                 if current_length + i <= self.config.dim_length[current_index + 1]:
                self.combination(current_index + 1, candidate_dims,
                                 temp_copy, current_length + i)
# test      TableCell
# config = Config([4, 4, 4, 4])
# tc = TableCellEachWindow(2, 5, config, 1)
# tc.gen_all_keys()


# In[7]:


class TableCell:
    def __init__(self, windows, column_num, row_num, config, dim_index=0):
        # each window get the window info
        self.key_num = {}
        for window in windows:
            window_info = WindowInfo(window, config)
#             print(window_info.gaps_num)
            tc_each_window = TableCellEachWindow(
                column_num, row_num, config, dim_index)
            tc_each_window.gen_all_keys(window_info)
            for key in tc_each_window.key_num.keys():
                #                 print(key, tc_each_window.key_num[key])
                self.key_num[key] = self.key_num.get(
                    key, 0) + tc_each_window.key_num[key]
        # use each window infor to get tablecellfor each window
        # accumulate all.
#         print(self.key_num)


# In[8]:


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


# In[9]:


def get_query_windows(unit_len, dim, ratios, nums, dim_scalar):
    windows = []
    for i in range(len(nums)):
        for j in range(nums[i]):
            windows.append(generate_a_window(
                unit_len, dim, ratios[i], dim_scalar))
    return windows


# In[10]:


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


# In[11]:


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


# In[12]:

class GlobalCost:
    def __init__(self, windows, bits_nums):
        self.windows = windows
        self.bits_nums = bits_nums
        self.factor_map = {}
        self.dim = len(bits_nums)
        self.dim_counter = []

#     when given a distribution, use this one to calculate the cost

    def cla_each_curve_length(self):
        res_temp_map = {}
        for window in self.windows:
            l_range = window.dimension_low
            h_range = window.dimension_high
            dim = window.dim

            for i in range(dim):
                low_temp = l_range[i]
                high_temp = h_range[i]

                for j in range(self.bits_nums[i]):
                    key = factor_letters[i] + str(j + 1)
                    res_temp_map[key] = res_temp_map.get(key, 0)
                    if (low_temp & 1):
                        #                     low_temp_map[key] = low_temp_map.get(key, 0) + 1
                        res_temp_map[key] = res_temp_map.get(key, 0) - 1

                    if (high_temp & 1):
                        # key = factor_letters[i] + str(j + 1)
                        #                     high_temp_map[key] = high_temp_map.get(key, 0) + 1
                        res_temp_map[key] = res_temp_map.get(key, 0) + 1

                    low_temp = low_temp >> 1
                    high_temp = high_temp >> 1
    #             low.append(low_temp_map)
    #             high.append(high_temp_map)
        #         res.append(res_temp_map)
        #     print(low)
        #     print(high)
        #     print(res)
        return res_temp_map

#     @timer
    def get_factor_value_via_bit_distribution(self, bit_distribution):
        #         start_time = perf_counter()
        start_time = time.time_ns()
        bits_nums = copy.deepcopy(self.bits_nums)
        index = len(bit_distribution) - 1
        res = {}
        for char in bit_distribution:
            for i in range(self.dim):
                if char == bit_letters[i]:
                    res[factor_letters[i]+str(bits_nums[i])] = index
                    bits_nums[i] -= 1
                    break
            index -= 1
        self.factor_value = res
#         return res
        if len(self.factor_map) == 0:
            self.each_curve_length_to_formula()
#         end_time = perf_counter()
        end_time = time.time_ns()
        return end_time - start_time

    def each_curve_length_to_formula(self):
        # if len(self.factor_map) == 0:
        self.factor_map = self.cla_each_curve_length()
        # print("self.bits_nums", self.bits_nums)
        if len(self.dim_counter) == 0:
            for index, bit_num in enumerate(self.bits_nums):
                counter = []
                for i in range(bit_num):
                    counter.append(
                        self.factor_map[factor_letters[index] + str(i + 1)])
                self.dim_counter.append(counter)

        formula = ""
        dim = len(self.bits_nums)
        for i in range(dim):
            for j in range(self.bits_nums[i]):
                key = factor_letters[i] + str(j + 1)
                factor = self.factor_map.get(key, 0)
                if factor != 0:
                    if i == 0 and j == 0:
                        if factor > 0:
                            formula += str(factor) + "*" + "2^" + key
                        else:
                            formula += str(factor) + "*" + "2^" + key
                    else:
                        if factor > 0:
                            formula += " + " + str(factor) + "*" + "2^" + key
                        else:
                            formula += " - " + str(-factor) + "*" + "2^" + key
    #     print(formula)
        return formula

#     @timer
    def global_cost(self, bit_distribution):
        #         factor_value =
        #         if len(self.factor_map) == 0:
        #             self.factor_map = self.cla_each_curve_length()
        #         start_time = perf_counter()
        start_time = time.time_ns()
        res = 0
        for key in self.factor_map:
            res += self.factor_map[key] * pow(2, self.factor_value[key])
#         end_time = perf_counter()
        end_time = time.time_ns()
        return res, end_time - start_time
#         return res

#     @timer
    def naive_global_cost(self, bit_distribution):
        #         start_time = perf_counter()
        start_time = time.time_ns()
        res = 0
        for window in self.windows:
            res += get_curve_value_via_location(
                window.dimension_high, bit_distribution, self.bits_nums)
            res -= get_curve_value_via_location(
                window.dimension_low, bit_distribution, self.bits_nums)
#         end_time = perf_counter()
        end_time = time.time_ns()
        return res, end_time - start_time

#     [[-2, 2, 1, 1, 2, 0, 0],
#  [1, 0, 0, -1, -2, 3, -2, 2, 0],
#  [4, 1, -2, 2, -1, -1, 1]]

    def get_global_optimal_curve(self):
        cursors = [bit_num - 1 for bit_num in self.bits_nums]
        length = sum(self.bits_nums)
        res = ""
#         print(self.dim_counter)
        for i in range(length):
#             print("----------")
            min_index = 0
            min_val = sys.maxsize
            for index, cursor in enumerate(cursors):
                if cursor >= 0:
                    if self.dim_counter[index][cursor] < min_val:
                        min_val = self.dim_counter[index][cursor]
                        min_index = index
#                         print("if :", min_val, min_index)
                    elif self.dim_counter[index][cursor] == min_val:
                       #  tie : self.dim_counter[min_index][cursors[min_index]] == self.dim_counter[index][cursor] 
                        pre_min_cursot = cursors[min_index]
                        while pre_min_cursot >= 0 and cursor >= 0:
                            if pre_min_cursot == 0 and cursor == 0:
                                break
                            if self.dim_counter[index][cursor] < self.dim_counter[min_index][pre_min_cursot]:
                                min_val = self.dim_counter[index][cursor]
                                min_index = index
                                break
                            elif self.dim_counter[index][cursor] > self.dim_counter[min_index][pre_min_cursot]:
                                break
                            else:
                                if pre_min_cursot > 0:
                                    pre_min_cursot -= 1
                                if cursor > 0:
                                    cursor -= 1
#                         print("elif :", min_val, min_index)
#             print(bit_letters[min_index])
                     
            cursors[min_index] -= 1
            res += bit_letters[min_index]
        return res



# In[13]:


class LocalCost:
    #     @timer
    def __init__(self, windows, bits_nums, config):
        self.windows = windows
        self.bits_nums = bits_nums
        self.factor_map = {}
        self.dim = len(bits_nums)
        self.area = 0
        self.config = config

    def prepare_tables(self):
        #         start_time = perf_counter()
        start_time = time.time_ns()
        for window in self.windows:
            self.area += window.get_area()
        total_length = 0
        for length in self.bits_nums:
            total_length += length
        self.table = []
        self.total_combination_num = 0
        for i in range(self.dim):
            dim_table = {}
            other_dims_length = total_length - self.bits_nums[i]
            for j in range(other_dims_length + 1):  # row
                for k in range(self.bits_nums[i]):  # column
                    #             print(k + 1, j)
                    # column_num, row_num
                    tc = TableCell(self.windows, k + 1, j, self.config, i)
                    dim_table[str(k + 1) + "_" + str(j)] = tc
                    self.total_combination_num += len(tc.key_num)
            self.table.append(dim_table)
#         print("total_combination_num", self.total_combination_num)
#         end_time = perf_counter()
        end_time = time.time_ns()
        return end_time - start_time

#     @timer
    def local_cost(self, bit_distribution):
        #         start_time = perf_counter()
        start_time = time.time_ns()
        self.get_table_reading_map(bit_distribution)
        res = 0
        cross = 0
        for i in range(self.dim):
            for j in range(self.bits_nums[i]):
                column_num = j + 1
                row_num = self.length_list[i][bit_letters[i] + str(column_num)]
                key = str(column_num) + "_" + str(row_num)
                tc = self.table[i].get(key, None)
                if tc is None:
                    continue
                table_cell_key = self.map_list[i][bit_letters[i] +
                                                  str(column_num)]
                res += tc.key_num[table_cell_key]
                if column_num > 1 and row_num > 1:
                    cross += tc.key_num[table_cell_key]
#         end_time = perf_counter()
        end_time = time.time_ns()
        return self.area - res, end_time - start_time

    def get_local_optimal_curve(self):
        res = ""
        bits_nums_copy = copy.deepcopy(self.bits_nums)
        all_bits_num = sum(bits_nums_copy)
        while all_bits_num > 0:
            max_index = -1
            max_res = -1
            for i in range(self.dim):
                if bits_nums_copy[i] == 0:
                    continue
                table_cell_key = ""
                for j in range(self.dim):
                    if i == j:
                        continue
                    table_cell_key += bit_letters[j] + str(bits_nums_copy[j])
                key = str(bits_nums_copy[i]) + "_" + \
                    str(all_bits_num - bits_nums_copy[i])
#                 print(key)
                tc = self.table[i].get(key, None)
#                 print(tc.key_num)
                if table_cell_key not in tc.key_num.keys():
                    max_index = i
                    max_res = 0
                else:
                    #                     print("here", tc.key_num[table_cell_key])
                    if max_res < tc.key_num[table_cell_key]:
                        max_res = tc.key_num[table_cell_key]
                        max_index = i
            if max_res == 0:
                all_bits_num -= self.dim
                for j in range(self.dim):
                    bits_nums_copy[j] -= 1
                    res += bit_letters[self.dim - 1 - j]
            else:
                all_bits_num -= 1
                bits_nums_copy[max_index] -= 1
                res += bit_letters[max_index]
        return res

    def print_table(self):
        for i in range(self.dim):
            for key in self.table[i].keys():
                print(key, self.table[i][key].key_num)
#             print(self.table[i].keys())

    def get_table_reading_map(self, bit_distribution):
        counter = [0 for _ in range(self.dim)]
        map_list = [{} for _ in range(self.dim)]
        length_list = [{} for _ in range(self.dim)]

        lbd = len(bit_distribution)
        for i in range(lbd - 1, -1, -1):
            bit_index = bit_letters.index(bit_distribution[i])
            counter[bit_index] += 1
            key = bit_distribution[i] + str(counter[bit_index])
            val_str = ""
            bit_dis_len = 0
            for j in range(self.dim):
                if j == bit_index:
                    continue
    #             if counter[j] != 0:
                val_str += bit_letters[j] + str(counter[j])
                bit_dis_len += counter[j]
            map_list[bit_index][key] = val_str
            length_list[bit_index][key] = bit_dis_len
        self.map_list = map_list
        self.length_list = length_list

#     def get_map_from_bit_distribution(self):
    def get_map_bit_distribution(self, bit_distribution):
        bits_nums = copy.deepcopy(self.bits_nums)
        self.bits_nums_map = []
        self.bit_index_map = []
        self.masks_map = []
        masks = [pow(2, bits_num - 1) for bits_num in bits_nums]
        for char in bit_distribution:
            bit_index = bit_letters.index(char)
            bits_nums[bit_index] -= 1
            self.bits_nums_map.append(bits_nums[bit_index])
            self.bit_index_map.append(bit_index)
            self.masks_map.append(masks[bit_index])
            masks[bit_index] = masks[bit_index] >> 1

#             if mask & value != 0:
#                 vals[bit_index] += pow(2, bits_nums[bit_index])
#             mask = mask >> 1
#         return vals

    def get_location_via_curve_value(self, value, bit_distribution):
        vals = [0 for i in range(len(self.bits_nums))]
        mask = pow(2, len(bit_distribution) - 1)
        for i in range(len(bit_distribution)):
            bit_index = self.bit_index_map[i]
            if mask & value != 0:
                vals[bit_index] += pow(2, self.bits_nums_map[i])
            mask = mask >> 1
        return vals
#         bits_nums = copy.deepcopy(self.bits_nums)
#         bit_map = []
#         mask = pow(2, len(bit_distribution) - 1)
#         vals = [0 for i in range(len(bits_nums))]
#         for char in bit_distribution:
#             bit_index = bit_letters.index(char)
#             bits_nums[bit_index] -= 1
#             if mask & value != 0:
#                 vals[bit_index] += pow(2, bits_nums[bit_index])
#             mask = mask >> 1
#         return vals

#     def get_curve_value_via_location(location, bit_distribution, bits_nums):
#         res = 0
#         bit_current_location = len(bit_distribution) - 1
#         masks = [pow(2, bits_num - 1) for bits_num in bits_nums]
#         for char in bit_distribution:
#             bit_index = bit_letters.index(char)
#             if masks[bit_index] & location[bit_index] !=0:
#                 res += pow(2, bit_current_location)
#             masks[bit_index] = masks[bit_index] >> 1
#             bit_current_location -= 1
#         return res

    def get_curve_value_via_location(self, location, bit_distribution, bits_nums):
        res = 0
        bit_current_location = len(bit_distribution) - 1
        for i in range(len(bit_distribution)):
            bit_index = self.bit_index_map[i]
            if self.masks_map[i] & location[bit_index] != 0:
                res += pow(2, bit_current_location)
            bit_current_location -= 1
        return res

#     @timer
    def naive_local_cost(self, bit_distribution):
        #         start_time = perf_counter()
        start_time = time.time_ns()
        res = 0
        dim = len(self.bits_nums)
        self.get_map_bit_distribution(bit_distribution)
        for window in self.windows:
            low = self.get_curve_value_via_location(
                window.dimension_low, bit_distribution, self.bits_nums)
            high = self.get_curve_value_via_location(
                window.dimension_high, bit_distribution, self.bits_nums)
            is_in = True
            num = 1
#             print(low, high)
            for val in range(low, high + 1, 1):
                location = self.get_location_via_curve_value(
                    val, bit_distribution)
                flag = True
                for i in range(dim):
                    if location[i] < window.dimension_low[i] or location[i] > window.dimension_high[i]:
                        flag = False
                        break
                if flag:
                    if is_in:
                        continue
                    else:
                        is_in = True
                        num += 1
                else:
                    is_in = False
            res += num
#         end_time = perf_counter()
        end_time = time.time_ns()
        return res, end_time - start_time


# In[14]:


def get_2d(bits_nums, num=10, unit_len=0.01):
    dim = 2
#     ratios = [[1.0, 2.0], [1.0, 1.0],  [2.0, 1.0],[1.0, 4.0],[4.0, 1.0]]
    ratios = [[1.0, 1.0]]
    nums = [num]
#     nums = [1, 1, 1, 1, 1]
#     ratios = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0],[1.0, 1.0],[1.0, 1.0]]
#     nums = [num, num, num, num, num]
#     bits_nums = [9,8]
#     bits_nums = [9,7]
#     bits_nums = [8,8]
#     bit_distribution='BABABABABABABBAAB'
#     bit_distribution='AAAAAAAAABBBBBBB'
#     bit_distribution='BABABABABABABABA'

    dim_scalar = [(pow(2, l) - 1) for l in bits_nums]
    windows = get_query_windows(unit_len, dim, ratios, nums, dim_scalar)
    return windows, bits_nums, dim


def get_3d(bits_nums=[8, 8, 8], num=10, unit_len=0.01):
    dim = 3
#     ratios = [[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [2.0, 1.0, 2.0],[1.0, 4.0, 1.0],[4.0, 1.0, 1.0]]
#     ratios = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],[1.0, 1.0, 1.0],[1.0, 1.0, 1.0]]
#     nums = [num, num, num, num, num]
    ratios = [[1.0, 1.0, 1.0]]
    nums = [num]
#     bits_nums = [8,8,8]
#     bit_distribution='ABCABCABCABCABCABCABCABC'
    dim_scalar = [(pow(2, l) - 1) for l in bits_nums]
    windows = get_query_windows(unit_len, dim, ratios, nums, dim_scalar)
    return windows, bits_nums, dim


def get_4d(bits_nums=[8, 8, 8, 8], num=10, unit_len=0.01):
    dim = 4
#     ratios = [[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 1.0, 1.0], [1.0, 4.0, 1.0, 1.0],[1.0, 1.0, 1.0, 1.0],[1.0, 1.0, 1.0, 1.0]]
#     ratios = [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0],[1.0, 1.0, 1.0, 1.0],[1.0, 1.0, 1.0, 1.0]]
#     nums = [num, num, num, num, num]
    ratios = [[1.0, 1.0, 1.0, 1.0]]
    nums = [num]
    dim_scalar = [(pow(2, l) - 1) for l in bits_nums]
    windows = get_query_windows(unit_len, dim, ratios, nums, dim_scalar)
    return windows, bits_nums, dim,


# In[15]:


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
    import csv
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
            most_sig_res += bit_letters[i]

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
                Z_least_sig_res += bit_letters[i]
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


# windows, bits_nums, dim = get_2d(bits_nums = [8,8], num=100, unit_len=0.01)
# bit_distribution = "BABABABABABABABA"
# GC = GlobalCost(windows, bits_nums)
# GC.get_factor_value_via_bit_distribution(bit_distribution)
# print(GC.each_curve_length_to_formula())
# res1, _ = GC.global_cost(bit_distribution)
# res2, _ = GC.naive_global_cost(bit_distribution)
# assert res1 == res2, ("wrong global cost calculation res1:%d, res2:%d", (res1,res2))
# print("---------------------Finish Global cost---------------------:")

# print(GC.each_curve_length_to_formula())
# bit_distribution = GC.get_global_optimal_curve()
# print(bit_distribution)
# GC.get_factor_value_via_bit_distribution(bit_distribution)
# print(GC.global_cost(bit_distribution))

# def get_bit_dis_comb(res, current, bit_nums):
#     if (len(current) == 16):
#         res.append(current)
#     else:
#         for i in range(len(bit_nums)):
#             if bit_nums[i] > 0:
#                 bit_nums[i] -= 1
#                 get_bit_dis_comb(res, current+bit_letters[i], bit_nums)
#                 bit_nums[i] += 1
# res = []
# get_bit_dis_comb(res, "", bits_nums)
# min_cost = 10 * pow(2, 15)
# min_bd = []
# min_total_bd = []
# for bd in res:
#     GC.get_factor_value_via_bit_distribution(bd)
#     temp, _ = GC.global_cost(bd)
# #     print(bd, temp)
#     if min_cost > temp:
#         min_cost = temp
#         min_bd = []
#         min_bd.append(bd)
#     elif min_cost == temp:
#         min_bd.append(bd)
# print(min_bd, "minimal global cost", min_cost)


# print(bits_nums)
# store_all_windows(windows, dim)
# config = Config(bits_nums)
# print("total windows:", len(windows))
# # print("window[0]:", windows[0])
# # print("window[0]:", windows[0].dimension_low)
# # print("window[0]:", windows[0].dimension_high)
# GC = GlobalCost(windows, bits_nums)
# GC.each_curve_length_to_formula()
# res1 = GC.global_cost(bit_distribution)
# res2 = GC.naive_global_cost(bit_distribution)
# assert res1 == res2, ("wrong global cost calculation res1:%d, res2:%d", (res1,res2))
# print("---------------------Finish Global cost---------------------:")
# LC = LocalCost(windows, bits_nums)
# # LC.get_table_reading_map(bit_distribution)
# # our_res = LC.local_cost(bit_distribution)
# # naive_res = LC.naive_local_cost(bit_distribution)
# # print(LC.area)
# # LC.print_table()
# print("---------------------Finish Local cost---------------------:")
# # assert our_res == naive_res, ("wrong local cost calculation our_res:%d, naive_res:%d", (our_res,naive_res))


class DeepQNetwork():
    def __init__(self,
                 n_actions,
                 n_features,
                 width=64,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.8,
                 replace_target_iter=500,
                 memory_size=5000,
                 batch_size=32,
                 e_greedy_increment=None,
                 output_graph=False, ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.width = width
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = 1.0
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = 0.0001
        # self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.epsilon = e_greedy

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [
                tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        tf.reset_default_graph()
        # tf.compat.v1.reset_default_graph()
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(
            tf.float32, [None, self.n_features], name='s')  # input State
        # print('_build_net self.n_features :', self.n_features)
        self.s_ = tf.placeholder(
            tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(
            0., 0.1), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, self.width * 2, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            e2 = tf.layers.dense(e1, self.width, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e2')
            # e3 = tf.layers.dense(e2, self.width / 2, tf.nn.relu, kernel_initializer=w_initializer,
            #                      bias_initializer=b_initializer, name='e3')
            self.q_eval = tf.layers.dense(e2, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, self.width * 2, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            t2 = tf.layers.dense(t1, self.width, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t2')
            # t3 = tf.layers.dense(t2, self.width / 2, tf.nn.relu, kernel_initializer=w_initializer,
            #                      bias_initializer=b_initializer, name='t3')
            self.q_next = tf.layers.dense(t2, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * \
                tf.reduce_max(self.q_next, axis=1,
                              name='Qmax_s_')  # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope('q_eval'):
            a_indices = tf.stack(
                [tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(
                params=self.q_eval, indices=a_indices)  # shape=(None, )

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(
                self.q_target, self.q_eval_wrt_a, name='TD_error'))

        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(
                self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation, force_random=False):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        random_val = np.random.uniform()
        # print(self.epsilon, random_val)
        if random_val < self.epsilon and not force_random:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(
                self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(
                self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(
                self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            })

        self.cost_his.append(cost)

        # increasing epsilon
        # self.epsilon = self.epsilon + \
        #     self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


def exchange_bit(string, position):
    if position >= len(string) - 1:
        return string
    left = string[position]
    right = string[position + 1]
    return string[:position] + right + left + string[position+2:]

# for i in range(6):
#     print(exchange_bit("abcde", i))


def bit_distribution_to_array(bit_distribution):
    res = []
    for s in bit_distribution:
        res.append(bit_letters.index(s))
    return res


def train_optimal_curve(bit_distribution, bits_nums, windows):
    iteration = 20000
    MEMORY_CAPACITY = 500
    length = sum(bits_nums)
    RL = DeepQNetwork(length, length, width=128, reward_decay=1.0, e_greedy=0.9, learning_rate=0.01, memory_size=MEMORY_CAPACITY)
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
    reward_gap = pre_total_cost
    # print("initial distribution", bit_distribution)
    # print("pre_total_cost", pre_total_cost)
    min_cost = pre_total_cost
    min_distribution = bit_distribution
    initial_cost = pre_total_cost
    min_step = 0
    all_rewards = []
    while step < iteration:
        # print(bit_distribution)
        state = bit_distribution_to_array(bit_distribution)
        # print(state)
        force_random = step < MEMORY_CAPACITY
        action = RL.choose_action(np.array(state), force_random)

        # print(action)
        new_bit_distribution = exchange_bit(bit_distribution, action)
        gc, _ = GC.global_cost(new_bit_distribution)
        lc, _ = LC.local_cost(new_bit_distribution)
        current_total_cost = gc * lc
        reward = (pre_total_cost - current_total_cost) / reward_gap
        if reward >= 0:
            bit_distribution = new_bit_distribution
            all_rewards.append(current_total_cost / initial_cost)
        # all_rewards.append(current_total_cost)

        # print(reward)
        current_state = bit_distribution_to_array(bit_distribution)
        if pre_total_cost < min_cost:
            min_cost = pre_total_cost
            min_distribution = bit_distribution
            min_step = step
        pre_total_cost = current_total_cost
        if reward != 0:
            RL.store_transition(state, action, reward, current_state)
        if (step > 200) and (step % 5 == 0):
            RL.learn()
        step += 1
    # print("final distribution", bit_distribution, current_total_cost)
    # print("min_distribution", min_distribution, min_cost)
    # print("min_step", min_step)
    return min_distribution, bit_distribution, global_bit_distribution, local_bit_distribution


# # for 2d
bits_nums = [32, 32]
ratios = [[1.0, 1.0], [1.0, 4.0], [1.0, 16.0], [1.0, 64.0], [1.0, 256.0]]
unit_len = 0.001
# dim = 2
# dims = [2, 3, 4]
dims = [2]
nums = [1000]
for dim in dims:
    for num in nums:
        for ratio in ratios:
            dim_scalar = [(pow(2, l) - 1) for l in bits_nums]
            windows = get_query_windows(
                unit_len, dim, [ratio], [num], dim_scalar)
            store_all_windows(windows, dim, num, ratio[dim - 1])
            init_bit_distribution = random_bit_distribution(bits_nums)
            min_distribution, rl_bit_distribution, global_bit_distribution, local_bit_distribution = train_optimal_curve(
                init_bit_distribution, bits_nums, windows)
            quilts_bit_distribution = quilts_curve_design(
                windows[0], dim, bits_nums)
            GC = GlobalCost(windows, bits_nums)
            # GC.get_factor_value_via_bit_distribution(bit_distribution)
            GC.each_curve_length_to_formula()
            global_bit_distribution = GC.get_global_optimal_curve()
            # use json to store all the distribution, and record the window path.
            # data = {'md': min_distribution, "rld": rl_bit_distribution,
            #         "gd": global_bit_distribution, "ld": local_bit_distribution, "quilts_C": quilts_bit_distribution[0], "quilts_Z": quilts_bit_distribution[1]}
            data = {"rld": rl_bit_distribution, "gd": global_bit_distribution, "quilts_Z": quilts_bit_distribution[1]}
            json_path = 'windows/' + \
                str(dim) + "/" + str(num) + "/" + \
                str(int(ratio[dim - 1])) + '/bit_distributions.json'

            with open(json_path, 'w') as outfile:
                json.dump(data, outfile)


