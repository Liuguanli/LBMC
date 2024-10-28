
# load data and queries
import argparse
import copy
import csv
import json
import math
import os
import random
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

bit_letters_map = {"A":0, "B":1, "C":2, "D":3, "E":4}

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

def load_queries(file_name="OSM_1000_64_262144_dim2_norm", dim_scalar=[pow(2, 20) - 1, pow(2, 20) - 1]):
    query_path = "./Learned-BMTree/query"
    dim = 2
    windows = []
    if not os.path.exists(os.path.join(query_path, file_name + ".json")):
        return windows
    with open(os.path.join(query_path, file_name + ".json"), 'r') as f:
        queries = json.load(f)

        for i, row in enumerate(queries):
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


def get_bit_distribution(windows, name, dim=2, bits_nums=[20, 20]):

    res = ""
    if name == "LC":
        for i in range(dim):
            res = res + bit_letters[i] * bits_nums[i]
    elif name == "ZC":
        total_bits = sum(bits_nums)
        while total_bits > 0:
            for i in range(dim):
                if (bits_nums[i] > 0):
                    res = res + bit_letters[i]
                    bits_nums[i] -= 1
                    total_bits -= 1
    elif name == "QUILTS":
        _, res = quilts_curve_design_multi(windows, windows[0], dim, bits_nums)
    elif name == "LBMC":
        # TODO: LBMC
        # TODO: calculate value
        # TODO: order data by value
        pass

    return res


def load_data(file_name="OSM_10000"):
    data_path = "./Learned-BMTree/data"
    data = []
    if not os.path.exists(os.path.join(data_path, file_name+ ".json")):
        return data
    with open(os.path.join(data_path, file_name + ".json"), 'r') as f:
        dataset = json.load(f)
        for row in dataset:
            point = Point([int(row[0]), int(row[1])])
            data.append(point)
    return data


def get_BMC_value(point, bit_distribution, dim = 2):

    xs = point

    merged_value = 0
    idxs = [0 for i in range(dim)]
    shift_count = 0
    
    for char in reversed(bit_distribution): 
        index = bit_letters_map[char]
        bit = (xs[index] >> idxs[index]) & 1
        idxs[index] += 1
        merged_value |= (bit << shift_count)
        
        shift_count += 1
    
    return merged_value

def order_data_points(data, name, bit_distribution):

    # TODO: gen model based on bit_distribution
    if name == "HC":
        hilbert_curve = HilbertCurve(20, 2)
        for point in data:
            point.value = hilbert_curve.distances_from_points([[max(x - 1, 0) for x in point.xs]])[0]
        data.sort(key=lambda x: x.value)
    else:
        for point in data:
            point.value = get_BMC_value(point.xs, bit_distribution)
        data.sort(key=lambda x: x.value)
    return data



def test_pg(data, queries, bit_distribution, db_password=123456, dim=2):
    conn = psycopg2.connect("dbname=postgres user=postgres password={} host=localhost port=5432".format(db_password))
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS location;")
    cur.execute("DROP TABLE IF EXISTS values;")
    cur.execute("DROP TABLE IF EXISTS location_value;")
    cur.execute('SET enable_bitmapscan TO off;')

    cur.execute('SELECT pg_stat_statements_reset();')

    dim_names = [f'dim{i + 1}' for i in range(dim)]
    sqlstr_dim_name = ' int, '.join(dim_names) + ' int'
    cur.execute("CREATE TABLE location_value (id integer, {}, sfcvalue bigint);".format(sqlstr_dim_name))

    print('--create table--')

    # dim_id_names = tuple(['id'] + dim_names + ['sfcvalue'])
    # with open('data/{}.csv'.format(data_path), 'r') as f:
    #     cur.copy_from(f, 'location_value', sep=',', columns=dim_id_names)


    # Prepare SQL query
    insert_query = sql.SQL('INSERT INTO location_value (id, dim1, dim2, sfcvalue) VALUES %s')

    # Execute the query with the data
    data_list = []
    for i in range(len(data)):
        data_list.append((i, data[i].xs[0], data[i].xs[1], data[i].value))
    psycopg2.extras.execute_values(cur, insert_query, data_list)
    print('--finished--')

    cur.execute('create index on location_value USING btree (sfcvalue) WITH (fillfactor=100);')
    cur.execute('SELECT indexname FROM pg_indexes WHERE schemaname = \'public\';')
    index_name = cur.fetchall()[0][0]
    cur.execute("cluster location_value using {};".format(index_name))

    print('--build finished--')

    print('--start window query--')

    hilbert_curve = HilbertCurve(20, 2)
    
    start_time = time.time()
    for i, query in enumerate(queries):
        if bit_distribution == "":
            curve_value_low = hilbert_curve.distances_from_points([[max(x - 1, 0) for x in query.dimension_low]])[0]
            curve_value_high = hilbert_curve.distances_from_points([[max(x - 1, 0) for x in query.dimension_high]])[0]
            # curve_value_low = 0
            # curve_value_high = int(pow(2, 40))
        else:
            curve_value_low = get_BMC_value(query.dimension_low, bit_distribution)
            curve_value_high = get_BMC_value(query.dimension_high, bit_distribution)
        filter = ' '.join(['and (dim{} between {} and {})'.format(i + 1, query.dimension_low[i], query.dimension_high[i]) for i in range(dim)])
        cur.execute(
            "select * from location_value where \
            (sfcvalue between {} and {}) {};".format(curve_value_low, curve_value_high, filter))
    end_time = time.time()
    time_usage = end_time - start_time
    print('finish running, time usage: {} ms'.format(time_usage * 1000 / len(queries)))
    cur.execute(
        'select mean_exec_time, shared_blks_hit, shared_blks_read, local_blks_hit, local_blks_read, temp_blks_read from pg_stat_statements where query like \'select * from location_value where%\';')
    # cur.execute(
    #     'select * from pg_stat_statements where query like \'select * from location_value where%\';')
    result = cur.fetchall()

    # print(result)

    avg_block_hit = sum([row[1] for row in result]) / len(queries)
    avg_block_read= sum([row[2] for row in result]) / len(queries)

    print('avg block hit: {}, avg block read:{}, avg block access:{}'.format(avg_block_hit, avg_block_read, avg_block_hit + avg_block_read))
    conn.commit()
    cur.close()
    conn.close()

def main():
    windows = load_queries(args.training_query)
    data = load_data(args.dataset)
    bit_distribution = get_bit_distribution(windows, args.name)
    data = order_data_points(data, args.name, bit_distribution)
    windows = load_queries(args.test_query)
    test_pg(data, windows, bit_distribution, args.db_password)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset name.")
    parser.add_argument("--test_query", help="")
    parser.add_argument("--training_query", help="")
    parser.add_argument("--name", help="")
    parser.add_argument("--db_password", help="")

    # parser.add_argument("path", help="")
    args = parser.parse_args()

    '''redirect the std out to file'''
    result_save_path = f'result/postgresql/{args.name}/'
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)

    f = open(result_save_path + '{}_{}_{}.txt'.format(args.dataset, args.test_query, args.name), 'a')
    sys.stdout = f
    main()


# python python/pg_test.py --dataset OSM_10000 --training_query OSM_1000_64_262144_dim2_norm --test_query OSM_2000_64_262144_dim2_norm --name HC --db_password 123456