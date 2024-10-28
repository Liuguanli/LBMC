import subprocess
from enum import Enum

ORIGINAL = '0'
GLOBAL = '1'
LOCAL = '2'
ALL = '3'


def test_overall_cardinality(cost):
    # data_sizes = [10000, 100000, 1000000, 10000000]
    data_sizes = [100000000]
    action_depths = [10]
    for action_depth in action_depths:
        for data_size in data_sizes:
            result = subprocess.run(['python', 'exp_opt_fast.py', '--data', 'OSM_{}'.format(data_size), '--query', 'OSM_1000_dim2', '--is_opt_cost', cost, '--action_depth'
                , str(action_depth), '--data_sample_points', str(int(data_size * 0.0001)), '--exp_query_file', 'OSM_2000_dim2'], capture_output=True, text=True)
            # print(result)
            # with open('exp_extension.txt', 'a') as f:
            #     f.write(result.stdout)

            # result = subprocess.run(['python', 'pg_test.py', '--pg_test_method', 'bmtree', '--data', 'uniform_{}'.format(data_size), '--query',
            #                             'skew_2000_dim2', '--bmtree', 'mcts_bmtree_uni_skew_1000', '--db_password', '123456'], capture_output=True, text=True)

# python exp_opt_fast.py --data OSM_100000000 --query OSM_1000_dim2 --is_opt_cost 1 --action_depth 2 --data_sample_points 10000 --exp_query_file OSM_2000_dim2


def test_varying_tree_height(cost):
    data_sizes = [1000000]
    action_depths = [8,9,10,11,12]
    for action_depth in action_depths:
        for data_size in data_sizes:
            result = subprocess.run(['python', 'exp_opt_fast.py', '--data', 'OSM_{}'.format(data_size), '--query', 'OSM_1000_dim2', '--is_opt_cost', cost, '--action_depth'
                , str(action_depth), '--data_sample_points', str(int(data_size * 0.05)), '--exp_query_file', 'OSM_2000_dim2'], capture_output=True, text=True)

            # with open('exp_extension.txt', 'a') as f:
            #     f.write(result.stdout)

            # result = subprocess.run(['python', 'pg_test.py', '--pg_test_method', 'bmtree', '--data', 'uniform_{}'.format(data_size), '--query',
            #                             'skew_2000_dim2', '--bmtree', 'mcts_bmtree_uni_skew_1000', '--db_password', '123456'], capture_output=True, text=True)


def test_varing_sampling_rate(cost):
    data_sizes = [100000000]
    sampling_rates = [0.00001, 0.000025, 0.00005, 0.000075, 0.0001]
    for data_size in data_sizes:
        for sampling_rate in sampling_rates:
            result = subprocess.run(['python', 'exp_opt_fast.py', '--data', 'OSM_{}'.format(data_size), '--query', 'OSM_1000_dim2', '--is_opt_cost', cost,
                                     '--data_sample_points', str(int(data_size * sampling_rate)), '--exp_query_file', 'OSM_2000_dim2'], capture_output=True, text=True)

            with open('exp_extension.txt', 'a') as f:
                f.write(result.stdout)

if __name__ == "__main__":
    test_overall_cardinality(GLOBAL)
    # test_varying_tree_height(GLOBAL)

    test_overall_cardinality(LOCAL)
    # test_varying_tree_height(LOCAL)
    # test_overall_cardinality(ALL)
    # test_varying_tree_height(ALL)


    test_overall_cardinality(ORIGINAL)
    test_varing_sampling_rate(ORIGINAL)