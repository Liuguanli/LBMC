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
    result_save_path = 'pgsql_result/'
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

        with open('data/{}.csv'.format(data_path), 'w') as f:
            writer = csv.writer(f)
            id = 0
            for data in dataset:
                value = curve.output(data)
                writer.writerow([id] + data + [value])
                id += 1

    print('--start build the database environment--')
    '''connect the pgdb'''
    conn = psycopg2.connect("dbname=postgres user=postgres password={} host=localhost port=5432".format(args.db_password))
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS location;")
    cur.execute("DROP TABLE IF EXISTS values;")
    cur.execute("DROP TABLE IF EXISTS location_value;")
    cur.execute('SET enable_bitmapscan TO off;')

    cur.execute('SELECT pg_stat_statements_reset();')

    dim_names = [f'dim{i + 1}' for i in range(data_dim)]
    sqlstr_dim_name = ' int, '.join(dim_names) + ' int'
    cur.execute("CREATE TABLE location_value (id integer, {}, sfcvalue bigint);".format(sqlstr_dim_name))
    print('--create table--')

    dim_id_names = tuple(['id'] + dim_names + ['sfcvalue'])
    with open('data/{}.csv'.format(data_path), 'r') as f:
        cur.copy_from(f, 'location_value', sep=',', columns=dim_id_names)

    print('--finished--')

    cur.execute('create index on location_value USING btree (sfcvalue) WITH (fillfactor=100);')
    cur.execute('SELECT indexname FROM pg_indexes WHERE schemaname = \'public\';')
    index_name = cur.fetchall()[0][0]
    cur.execute("cluster location_value using {};".format(index_name))

    print('--build finished--')

    print('--start window query--')
    print('training query: {}'.format(args.train_query))
    print('testing query: {}'.format(args.query))
    print('action_depth: {}'.format(args.action_depth))



    number_of_records = 0
    start_time = time.time()
    for i, query in enumerate(queries):
        filter = ' '.join(['and (dim{} between {} and {})'.format(dim + 1, queries_box[i][0][dim], queries_box[i][1][dim])  for dim in range(data_dim)])
        cur.execute(
            "select * from location_value where \
            (sfcvalue between {} and {}) {};".format(
                query[0], query[1], filter))
        result = cur.fetchall()

        # Get the number of records
        number_of_records += len(result)

    end_time = time.time()
    time_usage = end_time - start_time
    print('finish running, time usage: {} ms'.format(time_usage * 1000 / len(queries_box)))

    cur.execute(
        'select mean_exec_time, shared_blks_hit, shared_blks_read, local_blks_hit, local_blks_read, temp_blks_read from pg_stat_statements where query like \'select * from location_value where%\';')
    # cur.execute(
    #     'select * from pg_stat_statements where query like \'select * from location_value where%\';')
    result = cur.fetchall()

    # print(result)

    avg_block_hit = sum([row[1] for row in result]) / len(queries_box)
    avg_block_read= sum([row[2] for row in result]) / len(queries_box)

    print('avg block hit: {}, avg block read:{}, avg block access:{}'.format(avg_block_hit, avg_block_read, avg_block_hit + avg_block_read))
    print('all_result_num: {}'.format(number_of_records))
    # mean_exec_time, shared_blks_hit, shared_blks_read, local_blks_hit, local_blks_read, temp_blks_read

    conn.commit()
    cur.close()
    conn.close()
    # os.remove('data/{}.csv'.format(data_path))
    # pass



    sys.exit(0)