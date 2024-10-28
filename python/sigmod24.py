
# load data and queries
import argparse
import copy
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from time import perf_counter

import numpy as np
import psycopg2
import psycopg2.extras
from hilbertcurve.hilbertcurve import HilbertCurve
from psycopg2 import sql

from utils import Point, Window

np.random.seed(1)
random.seed(1)

bit_letters = ["A", "B", "C", "D", "E"]

def basic_func_pg(data_size=10000, dataset='OSM', method="QUILTS", width=1024, height=16384, lbmc_cost="", bit_num=20):

    command = [
        'python', 'python/pg_test.py',
        '--bit_num', str(bit_num),
        '--dataset', f'{dataset}_{data_size}',
        '--width', str(width),
        '--height', str(height),
        '--training_query', f'{dataset}_1000_{width}_{height}_dim2_norm',
        '--test_query', f'{dataset}_2000_{width}_{height}_dim2_norm',
        '--name', method,
        '--db_password', '123456',
        '--lbmc_cost', lbmc_cost
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    error = process.stderr.read()
    if error:
        print("ERROR:", error.strip())

    process.wait()

def basic_func_pg_mix(data_size=10000, dataset='OSM', method="QUILTS"):

    result = subprocess.run(['python', 'python/pg_test.py', '--dataset', '{}_{}'.format(dataset, data_size),
                             '--training_query', '{}_1000_mix_norm'.format(dataset), '--test_query', '{}_2000_mix_norm'.format(dataset), '--name', method, '--db_password', '123456'], capture_output=True, text=True)
    print(result)


def fig_6_7():
    # varying dataset
    datasets = ['OSM', 'NYC', 'uniform', 'SKEW']
    methods = ['LBMC', 'QUILTS', 'LC', 'ZC', 'HC']
    methods = ['LBMC']
    data_sizes = [10000000]
    # for method in methods:
    #     for dataset in datasets:
    #         for data_size in data_sizes:
    #             # basic_func_pg(data_size=data_size,
    #             #               dataset=dataset, method=method)
    #             basic_func_pg_mix(data_size=data_size,
    #                           dataset=dataset, method=method)
                
    # methods = ['LBMC']
    for method in methods:
        for dataset in datasets:
            for data_size in data_sizes:
                basic_func_pg(data_size=data_size,
                              dataset=dataset, method=method)
                


def fig_8_9():
    # varying dataset cardinality
    methods = ['QUILTS', 'LC', 'ZC', 'HC']
    methods = ['LBMC', 'QUILTS', 'LC', 'ZC', 'HC']
    # methods = ['LBMC']
    datasets = ['OSM', 'SKEW']
    datasets = ['OSM']
    data_sizes = [10000, 100000, 1000000, 100000000]
    for method in methods:
        for dataset in datasets:
            for data_size in data_sizes:
                # basic_func_pg(data_size=data_size,
                #               dataset=dataset, method=method)
                basic_func_pg_mix(data_size=data_size,
                              dataset=dataset, method=method)
            #     break
            # break
    methods = ['LC']
    for method in methods:
        for dataset in datasets:
            for data_size in data_sizes:
                basic_func_pg(data_size=data_size,
                              dataset=dataset, method=method)
                # basic_func_pg_mix(data_size=data_size,
                #               dataset=dataset, method=method)


def fig_10():
    # varying query workload skewness
    methods = ['LC']
    datasets = ['OSM', 'SKEW']
    data_size = 10000000
    widths = [16384, 4096, 1024, 1024, 1024, 1024, 1024]
    heights = [1024, 1024, 1024, 4096, 16384, 65536, 262144]
    for method in methods:
        for dataset in datasets:
            for width, height in zip(widths, heights):
                basic_func_pg(data_size=data_size, dataset=dataset, method=method,
                              width=width, height=height)
                
    # methods = ['QUILTS', 'LC', 'ZC', 'HC']
    # datasets = ['OSM', 'SKEW']
    # data_size = 10000000
    # widths = [16384, 4096]
    # heights = [1024, 1024]
    # for method in methods:
    #     for dataset in datasets:
    #         for width, height in zip(widths, heights):
    #             basic_func_pg(data_size=data_size, dataset=dataset, method=method,
    #                           width=width, height=height)


def fig_11():
    # varying selectivity
    methods = ['LBMC', 'QUILTS', 'LC', 'ZC', 'HC']
    datasets = ['OSM', 'SKEW']
    methods = ['LC']
    data_size = 10000000
    widths = [256, 512, 1024, 2048, 4096]
    heights = [4096, 8192, 16384, 32768, 65536]
    for method in methods:
        for dataset in datasets:
            for width, height in zip(widths, heights):
                basic_func_pg(data_size=data_size, dataset=dataset, method=method,
                              width=width, height=height)


def ablation():
    # varying dataset
    datasets = ['OSM', 'NYC']
    methods = ['LBMC']
    lbmc_costs = ['product', 'sum', 'global', 'local']
    data_sizes = [10000000]
    for method in methods:
        for dataset in datasets:
            for lbmc_cost in lbmc_costs:
                for data_size in data_sizes:
                    basic_func_pg(data_size=data_size,
                                  dataset=dataset, method=method, lbmc_cost=lbmc_cost)
                
def fig_vary_bit_num_revision():
    # varying dataset
    datasets = ['OSM', 'SKEW']
    methods = ['LBMC', 'LC', 'ZC', 'HC', 'QUILTS']
    bit_nums = [16, 20, 24, 28, 31]
    data_sizes = [10000000]

    for method in methods:
        for dataset in datasets:
            for data_size in data_sizes:
                for bit_num in bit_nums:
                    basic_func_pg(data_size=data_size,
                                dataset=dataset, method=method, bit_num=bit_num)


def incremental_func_pg(data_size=10000, dataset='OSM', method="QUILTS", width=1024, height=16384, lbmc_cost="", bit_num=20, ratio=0):

    command = [
        'python', 'python/pg_test.py',
        '--bit_num', str(bit_num),
        '--dataset', f'{dataset}_{data_size}',
        '--width', str(width),
        '--height', str(height),
        '--training_query', f'{dataset}_1000_{width}_{height}_dim2_norm',
        '--test_query', f'{dataset}_2000_norm_incremental' + str(ratio) if ratio > 0 else "",
        '--name', method,
        '--db_password', '123456',
        '--lbmc_cost', lbmc_cost
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    error = process.stderr.read()
    if error:
        print("ERROR:", error.strip())

    process.wait()


def incremental_pg_traditional():
    # varying dataset
    datasets = ['OSM']
    methods = ['LC', 'ZC', 'HC']
    data_sizes = [10000000]
    ratios = [0, 50, 100, 25, 75]

    for ratio in ratios:
        for method in methods:
            for dataset in datasets:
                for data_size in data_sizes:
                    incremental_func_pg(data_size=data_size,
                                dataset=dataset, method=method, ratio=ratio)


if __name__ == "__main__":
    # fig_6_7()
    # fig_8_9()
    # fig_10()
    # fig_11()

    ablation()
