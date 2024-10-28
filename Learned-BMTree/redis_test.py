# Copyright (c) 2022. This code belongs to Jiangneng Li, Nanyang Technological University.

import psycopg2
import json
import torch
import subprocess
import os
import sys
import random
import math
from bmtree.bmtree_env import BMTEnv
from int2binary import Int2BinaryTransformer
import redis

import configs
args = configs.set_config()
import csv
from utils.query import Query
import numpy as np
import time
from utils.curves import ZorderModule, DesignQuiltsModule, HighQuiltsModule,  Z_Curve_Module


'''Set the state_dim and action_dim, for now'''
bit_length = args.bit_length
data_dim = len(bit_length)
bit_length = [int(i) for i in bit_length]
data_space = [2**(bit_length[i]) - 1 for i in range(len(bit_length))]
smallest_split_card = args.smallest_split_card
cost_types = {1:'global', 2:'local', 3:'both', 0:'sampling'}

# python pg_test.py --pg_test_method bmtree  --data uniform_1000000 --query skew_2000_dim2 --bmtree mcts_bmtree_uni_skew_1000 --db_password ''


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
            if dim_val < query_range[0][i] or dim_val > query_range[1][i]:
                break
        res.append(candidate)
    return res


def get_current_branch():
    return "-------------branch: " + subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode('utf8')

if __name__ == '__main__':

    '''fetch the information of exp data, query, method, etc.'''
    test_method = args.pg_test_method # bmtree
    data_path = args.data # dataset
    query_path = args.query # query dataset
    warm_cache_repeat = args.warm_cache_repeat

    binary_transfer = Int2BinaryTransformer(data_space)

    '''redirect the std out to file'''
    result_save_path = 'redis_result/'
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)

    if test_method == 'bmtree':
        f = open(result_save_path + '{}_{}_{}_{}.txt'.format(data_path, query_path, test_method, args.bmtree), 'a')
    else:
        f = open(result_save_path + '{}_{}_{}.txt'.format(data_path, query_path, test_method), 'a')
    sys.stdout = f
    print(get_current_branch())
    print('--load data and query--')

    bit_gap = bit_length[0] - 20

    # load data
    with open('data/{}.json'.format(data_path), 'r') as f:
        dataset = json.load(f)

    if bit_gap != 0:
        scale = 2 ** bit_gap
        for i in range(len(dataset)):
            dataset[i] = [int(item * scale) for item in dataset[i]]

    # load query
    with open('query/{}.json'.format(query_path), 'r') as f:
        queryset = json.load(f)

    if bit_gap != 0:
        scale = 2 ** bit_gap
        for i in range(len(queryset)):
            queryset[i] = [int(item * scale) for item in queryset[i]]

    queries_box = []
    for query in queryset:
        min_point = query[0:data_dim]
        max_point = query[data_dim :]
        queries_box.append([min_point, max_point])

    curve_values = []
    if test_method == 'bmtree' or test_method == 'z_curve' or test_method == 'quilts' or test_method == 'c_curve':

        if test_method == 'bmtree':
            bmtree = args.bmtree
            '''Set up the bmtree curve'''
            env = BMTEnv(list(dataset), None, bit_length, 'learned_sfcs/{}.txt'.format(bmtree), binary_transfer, smallest_split_card, args.max_depth)
            curve = env.tree
        if test_method == 'z_curve':
            # curve = z_order = ZorderModule(data_space)
            # curve = Z_Curve_Module(binary_transfer)

            env = BMTEnv(list(dataset), None, bit_length, None, binary_transfer,
                         smallest_split_card, args.max_depth)
            initial_action_z = 1
            env.generate_tree_zorder(initial_action_z)

            curve = env.tree

        if test_method == 'quilts':
            quilt_ind = args.quilts_ind
            if len(bit_length) == 2:
                curve = DesignQuiltsModule(binary_transfer)
                curve.order = curve.possible_orders[args.quilts_ind]
            else:
                curve = HighQuiltsModule(binary_transfer, data_dim)

        queries = []
        for box in queries_box:
            # print([curve.output(box[0]), curve.output(box[1])])
            # print(box)
            queries.append([curve.output(box[0]), curve.output(box[1])])

        print("-------------cost: " + cost_types[args.is_opt_cost] + " -------------\n")
        print('--start build bmtree index--')
        '''compute sfc value and merge'''
        # if not os.path.exists('data/sfcvalue/{}_sfc_{}_{}.csv'.format(test_method, data_path, query_path)):

        # with open('data/{}.csv'.format(data_path), 'w') as f:
        #     writer = csv.writer(f)
        #     id = 0
        for data in dataset:
            value = curve.output(data)
            curve_values.append(value)
            # writer.writerow([id] + data + [value])
            # id += 1

    r = redis.Redis(host='localhost', port=6379, db=0)

    dim_names = [f'dim{i + 1}' for i in range(data_dim)]
    
    redis_data = []
    for i, data in enumerate(dataset):
        value_dict = {dim_names[0]: data[0], dim_names[1]: data[1]}
        value = json.dumps(value_dict)
        redis_data.append((curve_values[i], value))

    insert_data(r, redis_data)

    query_time_usage = 0
    filter_time_usage = 0
    number_of_records = 0


    for i, query in enumerate(queries):
       
        curve_value_low = query[0]
        curve_value_high = query[1]
        
        start_time = time.time()
        candidates = range_query(r, curve_value_low, curve_value_high)
        end_time = time.time()
        query_time_usage += end_time - start_time

        start_time = time.time()
        result = filter_query(candidates, queries_box[i], dim_names)
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

    sys.exit(0)