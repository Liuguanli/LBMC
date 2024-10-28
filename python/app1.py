#!/usr/bin/env python
# coding: utf-8


# from threading import enumerate
from app_query_gen import gen_3d_query_nyc, gen_3d_query_tpch
from app_query_gen import gen_2d_query_osm
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
import os.path
from app_query_gen import write_windows
import utils
from utils import ratio_to_pattern
from utils import Point
from utils import Window
from utils import Config
import numpy as np
import os
import csv
from time import perf_counter

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
bit_letters = ["A", "B", "C", "D", "E", "F", "G"]
factor_letters = ["a", "b", "c", "d", "e", "f", "g"]
logger_print = True
# value_letters = ["x", "y", "z"]


def get_reading_map(BMC: str, dim: int):
    reading_map = {}
    dim_counter = [0 for i in range(dim)]
    for i in range(len(BMC) - 1, -1, -1):
        letter_index = bit_letters.index(BMC[i])
        dim_counter[letter_index] += 1
        temp = []
        for i in range(dim):
            if i == letter_index:
                continue
            # temp.append(str(i) + "_" + str(dim_counter[i]))
            temp.append((i, dim_counter[i]))
        # key = str(letter_index) + "_" + str(dim_counter[letter_index])
        key = (letter_index, dim_counter[letter_index])
        reading_map[key] = temp
    return reading_map
    # reading_map = {}
    # dim_counter = [0 for i in range(dim)]
    # for i in range(len(BMC) - 1, -1, -1):
    #     letter_index = bit_letters.index(BMC[i])
    #     dim_counter[letter_index] += 1
    #     temp = []
    #     for i in range(dim):
    #         if i == letter_index:
    #             continue
    #         temp.append(str(i) + "_" + str(dim_counter[i]))
    #     key = str(letter_index) + "_" + str(dim_counter[letter_index])
    #     reading_map[key] = temp
    # return reading_map

def get_index_map(BMC: str):
    BMC_index_map = []
    for char in BMC:
        bit_index = utils.bit_letters.index(char)
        BMC_index_map.append(bit_index)
    return BMC_index_map

pow_of_two = [int(pow(2, i)) for i in range(128)]

def calculate_curve_value_via_location(location: list, BMC_index_map: list, bits_nums: list):
    res = 0
    bit_current_location = len(BMC_index_map) - 1
    masks = [pow_of_two[bits_num - 1] for bits_num in bits_nums]
    # for char in BMC:
    for bit_index in BMC_index_map:
        # bit_index = utils.bit_letters.index(char)
        if masks[bit_index] & location[bit_index] != 0:
            res += pow_of_two[bit_current_location]
        masks[bit_index] = masks[bit_index] >> 1
        bit_current_location -= 1
    return res

def calculate_global_cost(window: Window, BMC_index_map: list, bits_nums: list):
    return calculate_curve_value_via_location(window.dimension_high, BMC_index_map, bits_nums) - \
        calculate_curve_value_via_location(window.dimension_low, BMC_index_map, bits_nums)

def calculate_local_cost(window: Window, reading_map: dict) -> int:
    # config = Config(bit_nums)
    # win_info = WindowInfo(window, config)
    cost_val = 0
    for key, value in reading_map.items():
        each_combination_value = 1
        # rise_pattern = key.split("_")
        # rise_dim = int(rise_pattern[0])
        # rise_index = int(rise_pattern[1])
        rise_dim = key[0]
        rise_index = key[1]
        each_combination_value *= window.calculate_rise_pattern(rise_dim, rise_index)

        for val in value:
            # drop_patterns = val.split("_")
            # drop_dim = int(drop_patterns[0])
            # drop_index = int(drop_patterns[1])
            drop_dim = val[0]
            drop_index = val[1]
            each_combination_value *= window.calculate_drop_pattern(drop_dim, drop_index)
            # calculate_drop_pattern([window.dimension_low[drop_dim], window.dimension_high[drop_dim]], drop_index)
        cost_val += each_combination_value
    points_in_window = window.area
    # for i in range(len(window.dimension_low)):
    #     points_in_window *= (window.dimension_high[i] - window.dimension_low[i] + 1)
    return points_in_window - cost_val

def select_index(window: Window, BMCs: list, BMC_reading_maps:dict, BMC_index_maps:dict, dim: int, bit_nums: list):
    minimal_total_cost = 0
    opt_BMC = ""
    for i, BMC in enumerate(BMCs):
        # reading_map = get_reading_map(BMC, dim)
        reading_map = BMC_reading_maps[BMC]
        BMC_index_map = BMC_index_maps[BMC]
        local_cost = calculate_local_cost(window, reading_map)
        global_cost = calculate_global_cost(window, BMC_index_map, bit_nums)
        total_cost = global_cost * local_cost
        total_cost = local_cost
        # print(BMC, local_cost, global_cost, total_cost)
        if i == 0:
            minimal_total_cost = total_cost
            opt_BMC = BMC
            continue
        if total_cost < minimal_total_cost:
            minimal_total_cost = total_cost
            opt_BMC = BMC
    return opt_BMC

def select_index_for_all(windows: list, BMCs: list, dim: int, bit_nums: list, tag: str):

    windows_for_index = {}

    BMC_reading_maps = {}
    BMC_index_maps = {}

    for BMC in BMCs:
        windows_for_index[BMC] = []
        BMC_reading_maps[BMC] = get_reading_map(BMC, dim)
        BMC_index_maps[BMC] = get_index_map(BMC)

    start_time = perf_counter()
    for window in windows:
        opt_BMC = select_index(window, BMCs, BMC_reading_maps, BMC_index_maps, dim, bit_nums)
        # if opt_BMC == "CCCCCCCCCCCCCCCCAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBB":
        # print(opt_BMC, [int((window.dimension_high_raw[i] - window.dimension_low_raw[i]) * 1000) for i in range(dim)])
        windows_for_index[opt_BMC].append(window)
    end_time = perf_counter()
    print('{0} costs {1:.8f}s'.format("average index selection cost:", (end_time - start_time) / 1000))

    for key, value in windows_for_index.items():
        print(key, str(round(len(value) * 100.0/len(windows), 2)) + "%")
        windows_path = "windows/app1/OPT/" + tag + "/" + key + ".csv"
        write_windows(windows_path, value)


ZC_A_2d = "ABABABABABABABABABABABABABABABAB"
ZC_D_2d = "BABABABABABABABABABABABABABABABA"
LO_A_2d = "AAAAAAAAAAAAAAAABBBBBBBBBBBBBBBB"
LO_D_2d = "BBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAA"


BMC_ZC_1 = "CBACBACBACBACBACBACBACBACBACBACBACBACBACBACBACBA"
BMC_ZC_2 = "BCABCABCABCABCABCABCABCABCABCABCABCABCABCABCABCA"
BMC_ZC_3 = "CABCABCABCABCABCABCABCABCABCABCABCABCABCABCABCAB"
BMC_ZC_4 = "ACBACBACBACBACBACBACBACBACBACBACBACBACBACBACBACB"
BMC_ZC_5 = "BACBACBACBACBACBACBACBACBACBACBACBACBACBACBACBAC"
BMC_ZC_6 = "ABCABCABCABCABCABCABCABCABCABCABCABCABCABCABCABC"

BMC_LO_1 = "AAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBCCCCCCCCCCCCCCCC"
BMC_LO_2 = "AAAAAAAAAAAAAAAACCCCCCCCCCCCCCCCBBBBBBBBBBBBBBBB"
BMC_LO_3 = "BBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAACCCCCCCCCCCCCCCC"
BMC_LO_4 = "BBBBBBBBBBBBBBBBCCCCCCCCCCCCCCCCAAAAAAAAAAAAAAAA"
BMC_LO_5 = "CCCCCCCCCCCCCCCCAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBB"
BMC_LO_6 = "CCCCCCCCCCCCCCCCBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAA"


datasets = {"OSM": "/home/research/datasets/OSM_100000000_1_2_.csv",
"NYC": "/home/research/datasets/yellow_tripdata_2015_normalized_3d.csv",
"TPCH": "/home/research/datasets/tpch_normalized_3d.csv"}
if __name__ == "__main__":
    test_dataset = sys.argv[1]
    read_file = False
    if len(sys.argv) > 2 and sys.argv[2] == "read":
        read_file = True

    if test_dataset == "OSM":
        window_OMS = gen_2d_query_osm(
            1000, datasets[test_dataset], test_dataset, read_file)
        select_index_for_all(window_OMS, [ZC_A_2d, ZC_D_2d, LO_A_2d, LO_D_2d], 2, [16, 16], test_dataset)
    elif test_dataset == "NYC":
        window_NYC = gen_3d_query_nyc(
            1000, datasets[test_dataset], test_dataset, read_file)
        select_index_for_all(window_NYC, [BMC_ZC_1,BMC_ZC_2,BMC_ZC_3,BMC_ZC_5], 3, [16, 16, 16], test_dataset)
        # select_index_for_all(window_NYC, [BMC_ZC_4,BMC_ZC_6,BMC_LO_4,BMC_LO_6], 3, [16, 16, 16], test_dataset)
    elif test_dataset == "TPCH":
        window_TPCH = gen_3d_query_tpch(
            1000, datasets[test_dataset], test_dataset, read_file)
        select_index_for_all(window_TPCH, [BMC_LO_5,BMC_LO_2,BMC_ZC_5,BMC_ZC_2], 3, [16, 16, 16], test_dataset)
        # select_index_for_all(window_TPCH, [BMC_ZC_1,BMC_ZC_2,BMC_ZC_3,BMC_ZC_4], 3, [16, 16, 16], test_dataset)

