import json
import random
import time
import psycopg2

from io import StringIO

from global_cost import GlobalCost
from local_cost import LocalCost
from generators import random_sampling
from generators import BMC_generation
from generators import get_query_windows
from generators import generage_dataset


from utils import Window
# import generator

random.seed(2023)


def get_block_acc_num(BMC, bit, dim, distribution, windows):
    # print('--start build the database environment--')
    conn = psycopg2.connect("dbname=postgres user=postgres password=123456 host=localhost port=5432")
    cur = conn.cursor()
    cur.execute('SET enable_bitmapscan TO off;')
    cur.execute('SELECT pg_stat_statements_reset();')
    
    dim_names = [f'dim{i + 1}' for i in range(dim)]
    sqlstr_dim_name = ' float, '.join(dim_names) + ' float'
    cur.execute("CREATE TABLE location (id integer, {}, sfcvalue bigint);".format(sqlstr_dim_name))

    dim_id_names = tuple(['id'] + dim_names + ['sfcvalue'])
    
    
    dim_len = 2 ** bit
    lengths = [dim_len for i in range(dim)]
    bits_nums = [bit for i in range(dim)]
    
    GC = GlobalCost(windows, bits_nums)
    
    # print("number of windows: ", len(windows))

    curve_value_ranges = []
    for window in windows:
        low = GC.get_curve_value_via_location(window.dimension_low, BMC, bits_nums)
        high = GC.get_curve_value_via_location(window.dimension_high, BMC, bits_nums)
        curve_value_ranges.append((low, high))
    
    with open('../data/{}_{}_{}_points.csv'.format(distribution, dim, BMC), 'r') as f:
        cur.copy_from(f, 'location', sep=',', columns=dim_id_names)
        # print('--add data point to table location--')

    # with open('../data/{}_{}d_points.csv'.format(distribution, dim), 'r') as f:
    #     cur.execute('create table values (id integer, sfcvalue bigint)')
    #     index = 0
    #     buffer = StringIO()
    #     for line in f:
    #         items = line.strip().split(',')
    #         location = []
            
    #         for i in range(dim):
    #             location.append(int(float(items[i + 1]) * lengths[i]))
    #         curve_value = GC.get_curve_value_via_location(location, BMC, bits_nums)
    #         buffer.write(str(index) + ',' + str(curve_value) + '\n')
    #         index += 1
    #     buffer.seek(0)
    #     cur.copy_from(buffer, 'values', sep=',', columns=('id', 'sfcvalue'))    
        
    # pg_string = ['location.{}'.format(dim) for dim in dim_names]
    # pg_string = ', '.join(pg_string)
    # cur.execute(
    #     'create table location_value as select {}, values.sfcvalue \
    #     from location inner join values on location.id=values.id;'.format(pg_string))

    cur.execute('create index on location USING btree (sfcvalue) WITH (fillfactor=100);')
    # cur.execute('create index on location USING btree (dim1, dim2) WITH (fillfactor=100);')
    # cur.execute('CREATE EXTENSION btree_gist;')
    # cur.execute('create index on location USING gist (sfcvalue) WITH (fillfactor=100);')

    cur.execute('SELECT indexname FROM pg_indexes WHERE schemaname = \'public\';')
    index_name = cur.fetchall()[0][0]
    cur.execute("cluster location using {};".format(index_name))

    # print('--build finished--')  

    # print('--start window query--')
    # start_time = time.time()
    for i, window in enumerate(windows):
        # filter = ' '.join(['and (dim{} between {} and {})'.format(dim + 1, window.dimension_low_raw[dim], window.dimension_high_raw[dim])  for dim in range(dim)])
        filter = ' and '.join([' (dim{} between {} and {})'.format(j + 1, window.dimension_low_raw[j], window.dimension_high_raw[j])  for j in range(dim)])
        low, high = curve_value_ranges[i]
        cur.execute(
                "select * from location_value where \
                (sfcvalue between {} and {}) {};".format(
                    low, high, filter))
        # cur.execute(
        #         "select * from location where \
        #          {};".format(filter))
        # print(filter)
        result = cur.fetchall()
    end_time = time.time()
    # time_usage = end_time - start_time
    cur.execute(
            'select * from pg_stat_statements where query like \'select * from location where%\';')
    
    colnames = [desc[0] for desc in cur.description]

    rows = cur.fetchall()

    print(colnames)
    for row in rows:
        print(row)
    result = cur.fetchall()
    print(result)
    cur.execute(
            'select mean_time, shared_blks_hit, shared_blks_read, local_blks_hit, local_blks_read, temp_blks_read from pg_stat_statements where query like \'select * from location where%\';')
    
    result = cur.fetchall()
    print(result)
    block_hit = sum([row[1] for row in result])
    # avg_block_read= sum([row[2] for row in result]) / len(windows)
    return block_hit


def true_rank(BMCs, bit, dim, distribution, windows):

    res_list = []
    for BMC in BMCs:
        num = get_block_acc_num(BMC, bit, dim, distribution, windows)
        res_list.append((num, BMC))
    res_list.sort(key=lambda x: x[0])

    return res_list


def global_cost_only(BMCs, bit, dim, windows=None):

    bits_nums = [bit for i in range(dim)]
    res_list = []
    GC = GlobalCost(windows, bits_nums)
    for BMC in BMCs:
        cost_value = GC.global_cost(BMC)[0]
        res_list.append((cost_value, BMC))
    res_list.sort(key=lambda x: x[0])
    
    return res_list



def local_cost_only(BMCs, bit, dim, windows=None):
    bits_nums = [bit for i in range(dim)]
    res_list = []
    LC = LocalCost(windows, bits_nums)
    for BMC in BMCs:
        cost_value = LC.local_cost(BMC)[0]
        res_list.append((cost_value, BMC))
    res_list.sort(key=lambda x: x[0])
    
    return res_list


def both_global_local_cost(BMCs, bit, dim, windows=None):
    
    bits_nums = [bit for i in range(dim)]
    res_list = []
    LC = LocalCost(windows, bits_nums)
    GC = GlobalCost(windows, bits_nums)
    for BMC in BMCs:
        cost_value = LC.local_cost(BMC)[0] * GC.global_cost(BMC)[0]
        res_list.append((cost_value, BMC))
    res_list.sort(key=lambda x: x[0])
    
    return res_list


def compare_costs(BMCs, bit, distribution, dim, ratio, windows=None):
    candidate_ranks = {}
    
    candidate_ranks["ground_truth"] = true_rank(BMCs, bit, dim, distribution, windows)
    candidate_ranks["global_only"] = global_cost_only(BMCs, bit, dim, windows)
    candidate_ranks["local_only"] = local_cost_only(BMCs, bit, dim, windows)
    candidate_ranks["both_global_local"] = both_global_local_cost(BMCs, bit, dim, windows)
    sampled_windows = random_sampling(ratio=ratio, windows=windows)
    candidate_ranks["sampled_global"] = global_cost_only(BMCs, bit, dim, sampled_windows)
    candidate_ranks["sampled_local"] = local_cost_only(BMCs, bit, dim, sampled_windows)

    json.dump(candidate_ranks, open("../result/effectiveness/candidate_ranks_{}.json".format(distribution), "w"))


if __name__ == '__main__':
    default_bit = 14
    default_num = 256
    default_dim = 2
    default_cardinality = 10000
    default_ratio = [0.01, 0.01]
    default_sampling_ratio = 0.1
    default_distribution = "uniform"
    BMCs = BMC_generation(default_bit, default_dim, num=6)
    windows = get_query_windows(default_bit, default_dim, default_num, default_ratio)
    generage_dataset(BMCs, default_bit, default_dim, default_cardinality, default_distribution, windows)
    compare_costs(BMCs, default_bit, default_distribution, default_dim, default_sampling_ratio, windows)
