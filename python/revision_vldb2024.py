from E1 import *
import json
import sys
import os
import random

from sigmod24 import fig_vary_bit_num_revision
from sigmod24 import incremental_pg_traditional

import subprocess



def basic_func_redis(data_size=10000, dataset='OSM', method="QUILTS", width=1024, height=16384, lbmc_cost="", bit_num=20, ratio=0):

    # result = subprocess.run(['python', 'python/pg_test.py', '--bit_num', str(bit_num), '--dataset', '{}_{}'.format(dataset, data_size), '--width', str(width), '--height', str(height),
    #                          '--training_query', '{}_1000_{}_{}_dim2_norm'.format(dataset, width, height), '--test_query', '{}_2000_{}_{}_dim2_norm'.format(dataset, width, height), '--name', method, '--db_password', '123456', '--lbmc_cost', lbmc_cost], capture_output=True, text=True)
    # print("STDOUT:", result.stdout)
    # print("STDERR:", result.stderr)
    suffix = str(ratio) if ratio > 0 else ""
    command = [
        'python', 'python/redis_test.py',
        '--bit_num', str(bit_num),
        '--dataset', f'{dataset}_{data_size}',
        '--width', str(width),
        '--height', str(height),
        '--training_query', f'{dataset}_1000_norm_incremental',
        '--test_query', f'{dataset}_2000_norm_incremental' + suffix,
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

def read_windows_file(filename='../Learned-BMTree/query/OSM_1000_1024_16384_dim2.json'):
    with open(filename, 'r') as f:
        windows = json.load(f)
        return windows

def adapt_BMTree_windows(windows, dim=2, bit_nums=[20,20]):

    res = []

    for window in windows:
        dimension_low = []
        dimension_high = []
        dimension_low_raw = []
        dimension_high_raw = []
        for i in range(dim):
            dimension_low.append(window[i])
            dimension_high.append(window[i + dim])
            dimension_low_raw.append(window[i] * 1.0 / pow(2, bit_nums[i]))
            dimension_high_raw.append(window[i + dim] * 1.0 / pow(2, bit_nums[i]))

        new_window = Window(dimension_low, dimension_high,
                        dimension_low_raw, dimension_high_raw)
        res.append(new_window)
    return res


def exp1(bits_nums = [20, 20], dim=2):

    # BMTree-SP are reported by running calculate_cost_for_revision() in exp_paper.py 
    # figures are plotted in revision_vldb2024.ipynb
    datasets = ["OSM", "SKEW"]
    BMC = "ABABABABABABABABABABABABABABABABABABABAB"
    # run GC and LC using 1,000 queries with 20 bits on OSM and SKEW
    
    reading_map = get_reading_map(BMC, dim)
    index_map = get_index_map(BMC)
    my_gc = 0
    my_lc = 0

    for dataset in datasets:
        windows = read_windows_file('../Learned-BMTree/query/{}_1000_1024_16384_dim2.json'.format(dataset))
        windows = adapt_BMTree_windows(windows)

    
        GC = GlobalCost(windows, bits_nums)
        LC = LocalCost(windows, bits_nums)

        start_time = time.time_ns()
        for i, window in enumerate(windows):
            my_gc += calculate_global_cost(window, index_map, bits_nums)
        end_time = time.time_ns()
        print("my global cost:", (end_time - start_time))

        start_time = time.time_ns()
        for i, window in enumerate(windows):
            my_lc += calculate_local_cost(window, reading_map)
        end_time = time.time_ns()
        print("my local cost:", (end_time - start_time))

        naive_gc, time_cost = GC.naive_global_cost(BMC)
        print("naive globe cost:", time_cost)

        # naive_lc, time_cost = LC.naive_local_cost(BMC)
        # print("naive local cost:", time_cost)

def exp2_1():
    
    # BMTree are reported by running vary_bit_number() in exp_paper.py 
    # run bits in {16, 20, 24, 28, 32} on OSM and SKEW,
    fig_vary_bit_num_revision()

def exp2_2():
    # read queries and turn them into cost functions
    # run LBMC with bits in {16, 20, 24, 28, 32}
    # run 
    pass


def swap(BMC, swap_type="swap_adjacent"):
    if swap_type == "swap_adjacent":
        length = len(BMC) - 2
        i = random.randint(0, length)
        j = i + 1
        lst = list(BMC)
        lst[i], lst[j] = lst[j], lst[i]
        return ''.join(lst)
    else:
        length = len(BMC) - 1
        i = random.randint(0, length)
        j = random.randint(0, length)
        while i == j:
            j = random.randint(0, length)
        lst = list(BMC)
        lst[i], lst[j] = lst[j], lst[i]
        return ''.join(lst)

def exp_swap_bits(windows, BMC, epoch=200, bits_nums = [8, 8], dim=2, swap_type="swap_adjacent"):
    
    GC = GlobalCost(windows, bits_nums)
    LC = LocalCost(windows, bits_nums)
    my_gc_all = GC.global_cost(BMC)[0]
    my_lc_all = LC.local_cost(BMC)[0]
    total_cost = my_lc_all * my_gc_all
    costs = []
    costs.append(total_cost)
    for i in range(epoch - 1):
        if random.randint(1, 10) < 10:
            temp_BMC = swap(BMC, type)
            temp_cost = GC.global_cost(temp_BMC)[0] * LC.local_cost(temp_BMC)[0]
            if temp_cost < total_cost:
                BMC = temp_BMC
                total_cost = temp_cost
        else:
            BMC = swap(BMC, swap_type)
            total_cost = GC.global_cost(BMC)[0] * LC.local_cost(BMC)[0]
        costs.append(total_cost)
    # print(costs)
    with open(f'{swap_type}.json', 'w') as file:
        json.dump(costs, file)


def exp3_1():
    BMC = "ABABABABABABABAB"
    windows, bits_nums, dim = get_2d(bits_nums=[8, 8], BMC=BMC, num=100)
    exp_swap_bits(windows, BMC)
    exp_swap_bits(windows, BMC, swap_type="swap_any_two")

# def exp4_1():
    

def exp_redis():
    datasets = ['OSM']
    methods = ['LC', 'ZC', 'RANDOM']
    data_sizes = [10000000]
    ratios = [0, 50, 100, 25, 75]

    for ratio in ratios:
        for method in methods:
            for dataset in datasets:
                for data_size in data_sizes:
                    basic_func_redis(data_size=data_size, dataset=dataset, method=method, ratio=ratio)


if __name__ == "__main__":
    
    # exp1()

    # exp2_1()

    # exp3_1()
    exp_redis()

    # python python/redis_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental --name ZC --db_password 123456 --lbmc_cost 100
    # python python/redis_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental25 --name ZC --db_password 123456 --lbmc_cost 100
    # python python/redis_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental50 --name ZC --db_password 123456 --lbmc_cost 100
    # python python/redis_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental75 --name ZC --db_password 123456 --lbmc_cost 100
    # python python/redis_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental100 --name ZC --db_password 123456 --lbmc_cost 100

    # python python/redis_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental --name LC --db_password 123456 --lbmc_cost 100
    # python python/redis_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental25 --name LC --db_password 123456 --lbmc_cost 100
    # python python/redis_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental50 --name LC --db_password 123456 --lbmc_cost 100
    # python python/redis_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental75 --name LC --db_password 123456 --lbmc_cost 100
    # python python/redis_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental100 --name LC --db_password 123456 --lbmc_cost 100

    # python python/redis_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental --name HC --db_password 123456 --lbmc_cost 100
    # python python/redis_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental25 --name HC --db_password 123456 --lbmc_cost 100
    # python python/redis_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental50 --name HC --db_password 123456 --lbmc_cost 100
    # python python/redis_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental75 --name HC --db_password 123456 --lbmc_cost 100
    # python python/redis_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental100 --name HC --db_password 123456 --lbmc_cost 100

    incremental_pg_traditional()

    # python python/pg_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental --name ZC --db_password 123456 --lbmc_cost 100
    # python python/pg_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental25 --name ZC --db_password 123456 --lbmc_cost 100
    # python python/pg_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental50 --name ZC --db_password 123456 --lbmc_cost 100
    # python python/pg_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental75 --name ZC --db_password 123456 --lbmc_cost 100
    # python python/pg_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental100 --name ZC --db_password 123456 --lbmc_cost 100

    # python python/pg_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental --name LC --db_password 123456 --lbmc_cost 100
    # python python/pg_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental25 --name LC --db_password 123456 --lbmc_cost 100
    # python python/pg_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental50 --name LC --db_password 123456 --lbmc_cost 100
    # python python/pg_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental75 --name LC --db_password 123456 --lbmc_cost 100
    # python python/pg_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental100 --name LC --db_password 123456 --lbmc_cost 100

    # python python/pg_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental --name HC --db_password 123456 --lbmc_cost 100
    # python python/pg_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental25 --name HC --db_password 123456 --lbmc_cost 100
    # python python/pg_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental50 --name HC --db_password 123456 --lbmc_cost 100
    # python python/pg_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental75 --name HC --db_password 123456 --lbmc_cost 100
    # python python/pg_test.py --bit_num 20 --dataset OSM_10000000 --width 800 --height 600 --training_query OSM_1000_norm_incremental --test_query OSM_2000_norm_incremental100 --name HC --db_password 123456 --lbmc_cost 100

