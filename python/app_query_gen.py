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
import os.path
from utils import Point
from utils import Window
import numpy as np
import os
import csv
from time import perf_counter
from utils import ratio_to_pattern

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

def read_data_set(input_data_file, gap=10000):
    with open(input_data_file) as csvfile:
        lines = csvfile.readlines()
        points = []
        index = 0
        for line in lines:
            index += 1
            if index % gap == 0:
                row = line.strip().split(',')
                point = Point(list(map(float, row)))
                points.append(point)
    return points


def generate_a_window(points, unit_len, dim, ratio, dim_scalar):
    lengths = []
    for i in range(dim):
        lengths.append(unit_len[i] * ratio[i])

    while True:
        point = random.choice(points)
        dimension_low = []
        dimension_high = []
        dimension_low_raw = []
        dimension_high_raw = []
        for i in range(dim):
            start_dim_i = point.xs[i] - lengths[i] / 2
            end_dim_i = point.xs[i] + lengths[i] / 2
            if start_dim_i >= 0 and end_dim_i <= 1:
                dimension_low.append(math.floor(start_dim_i * dim_scalar[i]))
                dimension_high.append(math.floor(end_dim_i * dim_scalar[i]))
                dimension_low_raw.append(start_dim_i)
                dimension_high_raw.append(end_dim_i)
        if len(dimension_low) == dim:
            window = Window(dimension_low, dimension_high,
                    dimension_low_raw, dimension_high_raw)
            return window


def data_vis(points, tag, windows=None):

    xs_sp = [point.xs[0] for point in points]
    ys_sp = [point.xs[1] for point in points]
    plt.axis('off')
    plt.scatter(xs_sp, ys_sp, s=0.5)

    # TODO add windows!!!!

    plt.savefig(tag + ".png", format='png', bbox_inches='tight')
    plt.show()

def write_windows(windows_path, windows):
    if not os.path.exists(os.path.dirname(windows_path)):
        os.makedirs(os.path.dirname(windows_path))
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

def read_windows(windows_path, dim_scalar):
    windows = []
    dim = len(dim_scalar)
    if os.path.exists(windows_path):
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
    return windows

data_set_ratios = {
    # "OSM":[[1.0, 1.0], [1.0, 4.0],[1.0, 16.0], [1.0, 64.0], [1.0, 256.0]],
    "OSM":[[1.0, 1.0], [1.0, 4.0],[1.0, 16.0], [1.0, 64.0], [1.0, 256.0],
            [2.0, 2.0],[4.0, 4.0],[8.0, 8.0],[16.0, 16.0]],
                    "NYC":[[1.0, 1.0, 1.0], [1.0, 1.0, 4.0],[1.0, 1.0, 16.0], [1.0, 1.0, 64.0], [1.0, 1.0, 256.0],
            [2.0, 2.0, 16.0], [4.0, 4.0, 16.0], [8.0, 8.0, 16.0], [16.0, 16.0, 16.0]],
                    "TPCH":[[1.0, 10.0, 25], [10.0, 10.0, 25], [1, 10, 10], [1, 10, 100], [1, 100, 25]]}

def gen_3d_query_tpch(query_num: int, dataset: str, data_name: str, is_read_required: bool):
    # TODO read data set and sample num points.

    windows_path = "windows/app1/" + data_name + ".csv"
    bits_nums = [16, 16, 16]
    dim_scalar = [(pow(2, l) - 1) for l in bits_nums]
    windows = []
    if is_read_required:
        windows = read_windows(windows_path, dim_scalar)
    if len(windows) > 0:
        print("windows exists, size:", len(windows))
        return windows
    ratios = data_set_ratios[data_name]
    deltas = [0.004 for i in range(len(ratios))]
    return generate_windows_app(dataset, data_name, bits_nums, ratios, deltas, query_num)

def gen_3d_query_nyc(query_num: int, dataset: str, data_name: str, is_read_required: bool):
    windows_path = "windows/app1/" + data_name + ".csv"
    bits_nums = [16, 16, 16]
    dim_scalar = [(pow(2, l) - 1) for l in bits_nums]
    windows = []
    if is_read_required:
        windows = read_windows(windows_path, dim_scalar)
    if len(windows) > 0:
        print("windows exists, size:", len(windows))
        return windows
    ratios = data_set_ratios[data_name]
    deltas = [0.001 for i in range(len(ratios))]
    return generate_windows_app(dataset, data_name, bits_nums, ratios, deltas, query_num)

def gen_2d_query_osm(query_num: int, dataset: str, data_name: str, is_read_required: bool):
    windows_path = "windows/app1/" + data_name + ".csv"
    bits_nums = [16, 16]
    dim_scalar = [(pow(2, l) - 1) for l in bits_nums]
    windows = []
    if is_read_required:
        windows = read_windows(windows_path, dim_scalar)
    if len(windows) > 0:
        print("windows exists, size:", len(windows))
        return windows
    
    ratios = data_set_ratios[data_name]
    deltas = [0.0001 for i in range(len(ratios))]
    return generate_windows_app(dataset, data_name, bits_nums, ratios, deltas, query_num)


def generate_windows_app(dataset:str, data_name: str, bits_nums:list, ratios:list, deltas:list, query_num=1000):
    dim_scalar = [(pow(2, l) - 1) for l in bits_nums]
    dim = len(bits_nums)
    points = read_data_set(dataset)
    all_windows = []
    for ratio, delta in zip(ratios, deltas):
        unit_len = [delta for i in range(dim)]
        windows = []
        for i in range(query_num):
            window = generate_a_window(points, unit_len, dim, ratio, dim_scalar)
            windows.append(window)
        windows_path = "windows/app2/" + data_name + "/" + ratio_to_pattern(ratio) + ".csv"
        write_windows(windows_path, windows)
        if ratio[0] == ratio[1] and ratio[1] > ratio[2]:
            continue
        all_windows.extend(windows)

    mix_windows = random.choices(all_windows, k = query_num)
    mix_path = "windows/app1/" + data_name + ".csv"
    write_windows(mix_path, mix_windows)
    mix_path = "windows/app2/" + data_name + ".csv"
    write_windows(mix_path, mix_windows)
    return mix_windows
