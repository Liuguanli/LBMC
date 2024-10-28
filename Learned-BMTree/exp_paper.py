import subprocess
from enum import Enum
import shutil

from utils.query_generator import query_gen_mix
from utils.query_generator import query_gen_mix_incremental
from utils.query_generator import query_gen_mix_distribution
from utils.query_generator import query_gen_mix_distribution_incremental


ORIGINAL = '0'
GLOBAL = '1'
LOCAL = '2'
ALL = '3'


def basic_func(data_size=10000000, sampling_rate=0.005, action_depth=10, cost=GLOBAL, dataset='OSM', query_num=1000):
    result = subprocess.run(['python', 'exp_opt_fast.py', '--data', '{}_{}'.format(dataset, data_size),
                             '--query', '{}_{}_dim2'.format(dataset, query_num), '--is_opt_cost', cost, '--action_depth', str(
                                 action_depth), '--data_sample_points', str(int(data_size * sampling_rate)),
                             '--exp_query_file', '{}_2000_dim2'.format(dataset)], capture_output=True, text=True)

def test_zcurve(data_size=10000000, sampling_rate=0.005, action_depth=10, cost=GLOBAL, dataset='OSM'):
    result = subprocess.run(['python', 'exp_opt_fast.py', '--method', 'zcurve', '--data', '{}_{}'.format(dataset, data_size),
                             '--query', '{}_1000_dim2'.format(dataset), '--is_opt_cost', cost, '--action_depth', str(
                                 action_depth), '--data_sample_points', str(int(data_size * sampling_rate)),
                             '--exp_query_file', '{}_2000_dim2'.format(dataset)], capture_output=True, text=True)

# python exp_opt_fast.py --method zcurve --data OSM_10000000 --query OSM_1000_dim2 --exp_query_file OSM_2000_dim2

def test_overall_cardinality(cost):
    data_sizes = [10000, 100000, 1000000, 10000000, 100000000]
    for data_size in data_sizes:
        basic_func(data_size=data_size, cost=cost)


def test_varying_tree_height(cost):
    # action_depths = [6, 7,8,9,10,11,12]
    # datasets = ['OSM', 'NYC', 'TPCH', 'uniform', 'SKEW']
    datasets = ['NYC']
    action_depths = [5, 6, 7, 8, 9, 10]
    # action_depths = [11, 12]
    for dataset in datasets:
        for action_depth in action_depths:
            basic_func(dataset=dataset, action_depth=action_depth, cost=cost)


def test_varying_dataset(cost, depth, sampling_rate):
    datasets = ['OSM', 'NYC', 'TPCH', 'uniform', 'SKEW']
    for dataset in datasets:
        basic_func(dataset=dataset, cost=cost, action_depth=depth, sampling_rate=sampling_rate)


def test_varing_sampling_rate(cost):
    # sampling_rates = [0.001, 0.0025, 0.005, 0.0075, 0.01]
    # sampling_rates = [0.001, 0.0025, 0.0075, 0.01]
    # action_depths = [6,7,8,9]
    action_depths = [4,6,8,10]
    sampling_rates = [0.0001, 0.00025, 0.0005,
                      0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01]
    datasets = ['TPCH', 'SKEW']
    datasets = ['NYC']
    for dataset in datasets:
        for sampling_rate in sampling_rates:
            for action_depth in action_depths:
                basic_func(dataset=dataset, action_depth=action_depth,
                        sampling_rate=sampling_rate, cost=cost)


def test_varing_query_number(cost):
    query_nums = [100, 500, 1000, 1500, 2000]
    for query_num in query_nums:
        basic_func(sampling_rate=0.005, cost=cost, query_num=query_num)


def fig_1_and_2():
    test_overall_cardinality(GLOBAL)
    test_overall_cardinality(LOCAL)
    test_overall_cardinality(ORIGINAL)


def fig_3():
    # test_varying_tree_height(GLOBAL)
    # test_varying_tree_height(LOCAL)
    test_varing_sampling_rate(ORIGINAL)
    # test_zcurve()


def fig_4():
    test_varying_dataset(GLOBAL, 7, 0.001)
    test_varying_dataset(LOCAL, 7, 0.001)
    test_varying_dataset(ORIGINAL, 6, 0.001)
    
def fig_5():
    test_varing_query_number(GLOBAL)
    test_varing_query_number(LOCAL)
    test_varing_query_number(ORIGINAL)


def test_varing_bit_number(cost):
    datasets = ['OSM', 'SKEW']
    datasets = ['SKEW']
    bit_lengthes = [[16, 16], [20, 20], [24, 24], [28, 28], [31, 31]]
    i = 1
    for dataset in datasets:
        for bit_length in bit_lengthes:
            basic_func_pg(cost=cost, dataset=dataset, bit_length=bit_length, file_time=i)
            i += 1


def vary_bit_number():
    # test_varing_bit_number(GLOBAL)
    # test_varing_bit_number(LOCAL)
    # test_varing_bit_number(ORIGINAL)
    costs = [GLOBAL, LOCAL, ORIGINAL]
    datasets = ['OSM', 'SKEW']
    datasets = ['OSM']
    bit_lengthes = [[16, 16], [20, 20], [24, 24], [28, 28], [31, 31]]
    # bit_lengthes = [[20, 20], [24, 24], [28, 28], [31, 31]]

    for cost in costs:
        i = 1
        for dataset in datasets:
            for bit_length in bit_lengthes:
                # basic_func_pg(cost=cost, dataset=dataset, bit_length=bit_length, file_time=i)
                i += 1
                basic_func_pg_mix_incremental(dataset=dataset, cost=cost, train_query_num=2000, bit_length=bit_length, test_query_num=2000, file_time=i)

def vary_cardinality():
    data_sizes = [100000, 1000000, 100000000]
    costs = [GLOBAL, LOCAL, ORIGINAL]
    for cost in costs:
        i = 1
        for data_size in data_sizes:
            i += 1
            basic_func_pg_mix_incremental(data_size=data_size, cost=cost, train_query_num=2000, test_query_num=2000, file_time=i)


def basic_func_pg(data_size=10000000, sampling_rate=0.001, action_depth=10, cost=GLOBAL, dataset='OSM', width=1024, height=16384, bit_length=[20, 20], file_time=1, query_num=2000):
    result = subprocess.run(['python', 'exp_opt_fast.py', '--data', '{}_{}'.format(dataset, data_size),
                             '--query', '{}_{}_{}_{}_dim2'.format(dataset, query_num, width, height), '--is_opt_cost', cost, '--action_depth', str(
                                 action_depth), '--data_sample_points', str(int(data_size * sampling_rate)), '--bit_length', *map(str, bit_length), '--file_time', str(file_time),
                             '--exp_query_file', '{}_2000_{}_{}_dim2'.format(dataset, width, height)], capture_output=True, text=True)

    # destination_directory = f"./learned_sfcs/mcts_bmtree_{args.data}_{args.exp_query_file}.txt"
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    result = subprocess.run(['python', 'pg_test.py', '--pg_test_method', 'bmtree', '--data', '{}_{}'.format(dataset, data_size), '--train_query', '{}_{}_{}_{}_dim2'.format(dataset, query_num, width, height),
                    '--query', '{}_2000_{}_{}_dim2'.format(dataset, width, height), '--is_opt_cost', cost, '--bit_length', *map(str, bit_length), '--bmtree', 
                    f'mcts_bmtree_{dataset}_{data_size}_{dataset}_2000_{width}_{height}_dim2', '--db_password', '123456'], capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

def basic_func_pg_mix(data_size=10000000, sampling_rate=0.001, action_depth=10, cost=GLOBAL, dataset='OSM', bit_length=[20, 20], file_time=1, train_query_num=1000, test_query_num=2000):
    result = subprocess.run(['python', 'exp_opt_fast.py', '--data', '{}_{}'.format(dataset, data_size),
                             '--query', '{}_{}_mix'.format(dataset, train_query_num), '--is_opt_cost', cost, '--action_depth', str(
                                 action_depth), '--data_sample_points', str(int(data_size * sampling_rate)), '--bit_length', *map(str, bit_length), '--file_time', str(file_time),
                             '--exp_query_file', '{}_{}_mix'.format(dataset, test_query_num)], capture_output=True, text=True)

    # destination_directory = f"./learned_sfcs/mcts_bmtree_{args.data}_{args.exp_query_file}.txt"
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    result = subprocess.run(['python', 'pg_test.py', '--pg_test_method', 'bmtree', '--data', '{}_{}'.format(dataset, data_size), '--train_query', 
                             '{}_{}_mix'.format(dataset, train_query_num), '--query', '{}_{}_mix'.format(dataset, test_query_num), 
                             '--is_opt_cost', cost, '--bit_length', *map(str, bit_length), '--bmtree', 
                    f'mcts_bmtree_{dataset}_{data_size}_{dataset}_{test_query_num}_mix', '--db_password', '123456'], capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

def basic_func_pg_mix_incremental(data_size=10000000, sampling_rate=0.001, action_depth=10, cost=GLOBAL, dataset='OSM', bit_length=[20, 20], file_time=1, train_query_num=1000, test_query_num=2000):
    result = subprocess.run(['python', 'exp_opt_fast.py', '--data', '{}_{}'.format(dataset, data_size),
                             '--query', '{}_{}_incremental'.format(dataset, train_query_num), '--is_opt_cost', cost, '--action_depth', str(
                                 action_depth), '--data_sample_points', str(int(data_size * sampling_rate)), '--bit_length', *map(str, bit_length), '--file_time', str(file_time),
                             '--exp_query_file', '{}_{}_incremental'.format(dataset, test_query_num)], capture_output=True, text=True)

    # destination_directory = f"./learned_sfcs/mcts_bmtree_{args.data}_{args.exp_query_file}.txt"
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    # result = subprocess.run(['python', 'pg_test.py', '--pg_test_method', 'bmtree', '--data', '{}_{}'.format(dataset, data_size), '--train_query', 
    #                          '{}_{}_incremental'.format(dataset, train_query_num), '--query', '{}_{}_incremental'.format(dataset, test_query_num), 
    #                          '--is_opt_cost', cost, '--bit_length', *map(str, bit_length), '--bmtree', 
    #                 f'mcts_bmtree_{dataset}_{data_size}_{dataset}_{test_query_num}_incremental', '--db_password', '123456'], capture_output=True, text=True)
    # print("STDOUT:", result.stdout)
    # print("STDERR:", result.stderr)

    # test by 2000 queries.
    result = subprocess.run(['python', 'pg_test.py', '--pg_test_method', 'bmtree', '--data', '{}_{}'.format(dataset, data_size), '--train_query', 
                             '{}_{}_incremental'.format(dataset, train_query_num), '--query', '{}_2000_incremental'.format(dataset), 
                             '--is_opt_cost', cost, '--bit_length', *map(str, bit_length), '--bmtree', 
                    f'mcts_bmtree_{dataset}_{data_size}_{dataset}_{test_query_num}_incremental', '--db_password', '123456'], capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

def basic_func_pg_mix_distribution(data_size=10000000, sampling_rate=0.001, action_depth=10, cost=GLOBAL, dataset='OSM', bit_length=[20, 20], file_time=1, train_query_num=1000, test_query_num=2000, distribution_ratios=[]):
    result = subprocess.run(['python', 'exp_opt_fast.py', '--data', '{}_{}'.format(dataset, data_size),
                             '--query', '{}_{}_mix'.format(dataset, train_query_num), '--is_opt_cost', cost, '--action_depth', str(
                                 action_depth), '--data_sample_points', str(int(data_size * sampling_rate)), '--bit_length', *map(str, bit_length), '--file_time', str(file_time),
                             '--exp_query_file', '{}_{}_mix'.format(dataset, test_query_num)], capture_output=True, text=True)

    # destination_directory = f"./learned_sfcs/mcts_bmtree_{args.data}_{args.exp_query_file}.txt"
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    for ratio in distribution_ratios:
        result = subprocess.run(['python', 'pg_test.py', '--pg_test_method', 'bmtree', '--data', '{}_{}'.format(dataset, data_size), '--train_query', 
                                '{}_{}_mix'.format(dataset, train_query_num), '--query', '{}_{}_mix{}'.format(dataset, test_query_num, ratio), 
                                '--is_opt_cost', cost, '--bit_length', *map(str, bit_length), '--action_depth', str(
                                 action_depth), '--bmtree',  f'mcts_bmtree_{dataset}_{data_size}_{dataset}_{test_query_num}_mix', '--db_password', '123456'], capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)


def basic_func_pg_mix_redis(data_size=10000000, sampling_rate=0.001, action_depth=10, cost=GLOBAL, dataset='OSM', bit_length=[20, 20], file_time=1, train_query_num=1000, test_query_num=2000):
    result = subprocess.run(['python', 'exp_opt_fast.py', '--data', '{}_{}'.format(dataset, data_size),
                             '--query', '{}_{}_mix'.format(dataset, train_query_num), '--is_opt_cost', cost, '--action_depth', str(
                                 action_depth), '--data_sample_points', str(int(data_size * sampling_rate)), '--bit_length', *map(str, bit_length), '--file_time', str(file_time),
                             '--exp_query_file', '{}_{}_mix'.format(dataset, test_query_num)], capture_output=True, text=True)

    # destination_directory = f"./learned_sfcs/mcts_bmtree_{args.data}_{args.exp_query_file}.txt"
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    result = subprocess.run(['python', 'redis_test.py', '--pg_test_method', 'bmtree', '--data', '{}_{}'.format(dataset, data_size), '--train_query', 
                             '{}_{}_mix'.format(dataset, train_query_num), '--query', '{}_{}_mix'.format(dataset, test_query_num), 
                             '--is_opt_cost', cost, '--bit_length', *map(str, bit_length), '--bmtree', 
                    f'mcts_bmtree_{dataset}_{data_size}_{dataset}_{test_query_num}_mix', '--db_password', '123456'], capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)


def basic_func_pg_mix_distribution_incremental_redis(data_size=10000000, sampling_rate=0.001, action_depth=8, cost=GLOBAL, dataset='OSM', bit_length=[20, 20], file_time=1, train_query_num=1000, test_query_num=2000, distribution_ratios=[]):
    result = subprocess.run(['python', 'exp_opt_fast.py', '--data', '{}_{}'.format(dataset, data_size),
                             '--query', '{}_{}_incremental'.format(dataset, train_query_num), '--is_opt_cost', cost, '--action_depth', str(
                                 action_depth), '--data_sample_points', str(int(data_size * sampling_rate)), '--bit_length', *map(str, bit_length), '--file_time', str(file_time),
                             '--exp_query_file', '{}_{}_incremental'.format(dataset, test_query_num)], capture_output=True, text=True)

    # destination_directory = f"./learned_sfcs/mcts_bmtree_{args.data}_{args.exp_query_file}.txt"
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    result = subprocess.run(['python', 'redis_test.py', '--pg_test_method', 'bmtree', '--data', '{}_{}'.format(dataset, data_size), '--train_query', 
                             '{}_{}_incremental'.format(dataset, train_query_num), '--query', '{}_{}_incremental'.format(dataset, test_query_num), 
                             '--is_opt_cost', cost, '--bit_length', *map(str, bit_length), '--bmtree', 
                    f'mcts_bmtree_{dataset}_{data_size}_{dataset}_{test_query_num}_incremental', '--db_password', '123456'], capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    for ratio in distribution_ratios:
        # test by 2000 queries.
        result = subprocess.run(['python', 'pg_test.py', '--pg_test_method', 'bmtree', '--data', '{}_{}'.format(dataset, data_size), '--train_query', 
                                '{}_{}_incremental'.format(dataset, train_query_num), '--query', '{}_{}_incremental{}'.format(dataset, test_query_num, ratio), 
                                '--is_opt_cost', cost, '--bit_length', *map(str, bit_length), '--bmtree', 
                        f'mcts_bmtree_{dataset}_{data_size}_{dataset}_{test_query_num}_incremental', '--db_password', '123456'], capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr) 

    # ratio = 0.

    # result = subprocess.run(['python', 'pg_test.py', '--pg_test_method', 'bmtree', '--data', '{}_{}'.format(dataset, data_size), '--train_query', 
    #                         '{}_{}_incremental'.format(dataset, train_query_num), '--query', '{}_{}_incremental'.format(dataset, test_query_num), 
    #                         '--is_opt_cost', cost, '--bit_length', *map(str, bit_length), '--bmtree', 
    #                 f'mcts_bmtree_{dataset}_{data_size}_{dataset}_{test_query_num}_incremental', '--db_password', '123456'], capture_output=True, text=True)
    # print("STDOUT:", result.stdout)
    # print("STDERR:", result.stderr)

    # for ratio in distribution_ratios:
    #     result = subprocess.run(['python', 'redis_test.py', '--pg_test_method', 'bmtree', '--data', '{}_{}'.format(dataset, data_size), '--train_query', 
    #                             '{}_{}_incremental'.format(dataset, train_query_num), '--query', '{}_{}_incremental{}'.format(dataset, test_query_num, ratio), 
    #                             '--is_opt_cost', cost, '--bit_length', *map(str, bit_length), '--bmtree', 
    #                     f'mcts_bmtree_{dataset}_{data_size}_{dataset}_{test_query_num}_incremental', '--db_password', '123456'], capture_output=True, text=True)
    #     print("STDOUT:", result.stdout)
    #     print("STDERR:", result.stderr) 



def fig_6_7():
    # varying dataset
    datasets = ['OSM', 'NYC', 'uniform', 'SKEW']
    # datasets = ['OSM']
    data_sizes = [10000000]
    for dataset in datasets:
        for data_size in data_sizes:
            basic_func_pg(data_size=data_size, dataset=dataset, cost=ORIGINAL)
            # basic_func_pg_mix(data_size=data_size, dataset=dataset, cost=ORIGINAL)
            
    

def fig_8_9():
    # varying dataset cardinality
    datasets = ['OSM', 'SKEW']
    datasets = ['OSM']

    data_sizes = [10000, 100000, 1000000, 100000000]
    for dataset in datasets:
        for data_size in data_sizes:
            basic_func_pg(data_size=data_size, dataset=dataset, cost=ORIGINAL)
            # basic_func_pg_mix(data_size=data_size, dataset=dataset, cost=ORIGINAL)
        #     break
        # break

def fig_10():
    # varying query workload skewness
    datasets = ['OSM', 'SKEW']
    datasets = ['OSM']
    data_size = 10000000
    # widths = [1024, 1024, 1024, 1024, 1024]
    # heights = [1024, 4096, 16384, 65536, 262144]
    widths = [16384, 4096, 1024, 1024]
    heights = [1024, 1024, 1024, 4096]
    for dataset in datasets:
        for width, height in zip(widths, heights):
            basic_func_pg(data_size=data_size, dataset=dataset, cost=ORIGINAL, width=width, height=height)


def fig_11():
    # varying selectivity
    datasets = ['OSM', 'SKEW']
    datasets = ['OSM']
    data_size = 10000000
    widths = [256, 512, 2048, 4096]
    heights = [4096 , 8192, 32768, 65536]
    for dataset in datasets:
        for width, height in zip(widths, heights):
            basic_func_pg(data_size=data_size, dataset=dataset, cost=ORIGINAL, width=width, height=height)

def gen_mix():
    datasets = ['OSM', 'NYC', 'uniform', 'SKEW']
    widths = [16384, 4096, 1024, 1024, 1024]
    heights = [1024, 1024, 1024, 4096, 16384]
    ns = [1000, 2000]
    for dataset in datasets:
        for n in ns:
            query_gen_mix('{}_10000000'.format(dataset), n,  widths, heights)

def calculate_cost_for_revision():
    datasets = ['OSM', 'SKEW']
    sampling_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    for dataset in datasets:
        for sampling_rate in sampling_rates:
            basic_func_pg(action_depth=1, sampling_rate=sampling_rate, cost=ORIGINAL, dataset=dataset)


def test_varing_small_number():
    costs = [GLOBAL, LOCAL, ORIGINAL]
    query_nums = [1, 5, 10, 15, 20, 25, 50, 60, 70, 80, 90]
    query_nums = [1, 5, 10, 15, 20, 50]
    for cost in costs:
        for query_num in query_nums:
            basic_func_pg(sampling_rate=0.001, cost=cost, query_num=query_num)


def test_mix_workloads():
    dataset = 'OSM'
    costs = [GLOBAL, LOCAL, ORIGINAL]
    widths = [16384, 4096, 1024, 256, 64]
    heights = [64, 256, 1024, 4096, 16384]
    ns = [1, 5, 10, 15, 20, 50, 100, 500, 1000, 1500, 2000]
    for n in ns:
        query_gen_mix('{}_10000000'.format(dataset), n,  widths, heights)

    for cost in costs:
        for n in ns:
            basic_func_pg_mix(sampling_rate=0.001, cost=cost, train_query_num=n, test_query_num=2000)


def test_mix_incremental_workloads():
    dataset = 'OSM'
    costs = [GLOBAL, LOCAL, ORIGINAL]
    widths = [16384, 4096, 1024, 256, 64]
    heights = [64, 256, 1024, 4096, 16384]
    ns = [1, 5, 10, 15, 20, 50, 100, 500, 1000, 1500, 2000]
    query_gen_mix_incremental('{}_10000000'.format(dataset), ns,  widths, heights)

    for cost in costs:
        for n in ns:
            basic_func_pg_mix_incremental(sampling_rate=0.001, cost=cost, train_query_num=n, test_query_num=n)

def test_mix_distribution_workloads():
    dataset = 'OSM'
    costs = [GLOBAL, LOCAL, ORIGINAL]
    widths = [16384, 4096, 1024, 256, 64]
    heights = [64, 256, 1024, 4096, 16384]
    ns = [1, 5, 10, 15, 20, 50, 100, 500, 1000, 1500, 2000]
    distribution_ratios = [25, 50, 75, 100]
    for ratio in distribution_ratios:
        query_gen_mix_distribution('{}_10000000'.format(dataset), ratio,  widths, heights)

    for cost in costs:
        for n in ns:
            basic_func_pg_mix_distribution(sampling_rate=0.001, cost=cost, train_query_num=n, test_query_num=2000, distribution_ratios=distribution_ratios)

def test_mix_workloads_redis():
    dataset = 'OSM'
    costs = [GLOBAL, LOCAL, ORIGINAL]
    ns = [1, 5, 10, 15, 20, 50, 100, 500, 1000, 1500, 2000]

    widths = [16384, 4096, 1024, 256, 64]
    heights = [64, 256, 1024, 4096, 16384]
    for n in ns:
        query_gen_mix('{}_10000000'.format(dataset), n,  widths, heights)

    for cost in costs:
        for n in ns:
            basic_func_pg_mix_redis(sampling_rate=0.001, cost=cost, train_query_num=n, test_query_num=2000)
        #     break
        # break


def test_mix_distribution_incremental_redis_workloads():
    dataset = 'OSM'
    costs = [GLOBAL, LOCAL, ORIGINAL]
    costs = [ORIGINAL]
    widths = [16384, 4096, 1024, 256, 64]
    heights = [64, 256, 1024, 4096, 16384]
    ns = [1, 5, 10, 15, 20, 50, 100, 500, 1000, 1500, 2000]
    ns = [2000]
    distribution_ratios = [25, 50, 75, 100]
    distribution_ratios = []

    # # for ratio in distribution_ratios:
    # query_gen_mix_distribution_incremental('{}_10000000'.format(dataset), widths, heights)

    dir_index_map = {1:[9,10,11], 5:[5,6,7], 10:[4,5,6], 15:[4,5,6], 20:[3,4,5], 50:[3,4,5], 100:[3,4,5], 500:[3,4,5], 1000:[3,4,5], 1500:[3,4,5], 2000:[3,4,5]}
    
    for n in ns:
        for dir_index, cost in zip(dir_index_map[n], costs):
            source_file = f'/home/research/Dropbox/research/VLDB23/code/Learned-BMTree/fast_result/OSM_10000000_OSM_{n}_incremental/mcts/10000_0_8_10_10/{dir_index}/best_tree.txt' 
            shutil.copy(source_file, f"./learned_sfcs/mcts_bmtree_OSM_10000000_OSM_2000_incremental.txt")
            basic_func_pg_mix_distribution_incremental_redis(sampling_rate=0.001, cost=cost, train_query_num=n, test_query_num=2000, distribution_ratios=distribution_ratios)

    # for cost in costs:
    #     for n in ns:
    #         basic_func_pg_mix_distribution_incremental_redis(sampling_rate=0.001, cost=cost, train_query_num=n, test_query_num=2000, distribution_ratios=distribution_ratios)



if __name__ == "__main__":
    # gen_mix()
    # fig_1_and_2()
    # fig_4()
    # fig_5()
    # fig_3()

    # fig_6_7()
    # fig_8_9()
    # fig_10()
    # fig_11()

    # calculate_cost_for_revision()

    # vary_bit_number()

    vary_cardinality()

    # test_varing_small_number()

    # test_mix_workloads()

    # test_mix_incremental_workloads()

    # test_mix_distribution_workloads()

    # test_mix_workloads_redis()

    test_mix_distribution_incremental_redis_workloads()
