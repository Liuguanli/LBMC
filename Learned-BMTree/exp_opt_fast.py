# Copyright (c) 2022. This code belongs to Jiangneng Li, Nanyang Technological University.

import json
from datetime import datetime
import subprocess
from utils.data_generator import data_gen
from utils.query_generator import query_gen
dateTimeObj = datetime.now()
import time
import os
import random

import torch
from bmtree.bmtree_env import BMTEnv

from mcts import MCTS

from utils.curves import DesignQuiltsModule, Z_Curve_Module
from utils.metric_compute import ExperimentEnv

from int2binary import Int2BinaryTransformer

from utils.query import Query

from utils.storage import Storage
from utils.storage import Point
from utils.storage import Block
from utils.storage import BlockQuery


import shutil
import configs
args = configs.set_config()

from utils.metric_compute import execution_time

device = torch.device('cpu')


'''Set the state_dim and action_dim, for now'''
bit_length = args.bit_length
bit_length = [int(i) for i in bit_length]
data_space = [2**(bit_length[i]) - 1 for i in range(len(bit_length))]

page_size = args.page_size

# lr_actor = args.lr_actor  # learning rate for actor network
# lr_critic = args.lr_critic  # learning rate for critic network
smallest_split_card = args.smallest_split_card
max_depth = int(args.max_depth)
# discount_rate_hierarchical = args.discount_rate_hierarchical

'''compute dimension of state vector'''
data_dim = len(bit_length)

state_dim = len(bit_length) * 2
action_dim = len(bit_length)


'''solution method, scratch / heuristic opt'''
method = args.method

cost_types = {1:'global', 2:'local', 3:'both', 0:'sampling'}

def get_current_branch():
    return "-------------branch: " + subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode('utf8')

def experiment_ppo(result_save_path, file_writer, data_path = 'data/', query_path = 'query/', bit_gap=0):
    """
    This function start and end the experiment
    :param result_save_path:
    :param file_writer:
    :return: return nothing
    """

    global new_performance
    print(args.data)
    if not os.path.exists(data_path + args.data + '.json'):
        data_gen(args.data)

    '''load data and have dataset:full data, sampled_data: data for training'''
    with open(data_path + args.data + '.json', 'r') as f:
        dataset = json.load(f)

    ''' scale query workload if bit number is not 20. As BMTree hard codes the dataset and query.'''
    if bit_gap != 0:
        scale = 2 ** bit_gap
        for i in range(len(dataset)):
            dataset[i] = [int(item * scale) for item in dataset[i]]


    random.shuffle(dataset)
    data_card = len(dataset)
    # sampled_data = dataset[:int(args.data_sample_rate * data_card)]
    sampled_data = dataset[:args.data_sample_points]

    '''load query workload'''
    if not os.path.exists(query_path + args.query + '.json'):
        print(query_path + args.query + '.json')
        query_gen(args.data, args.query)
    with open(query_path + args.query + '.json', 'r') as f:
        queryset = json.load(f)

    file_writer.write(f"bit_gap: {bit_gap}\n")
    file_writer.write(f"data: {args.data}\n")
    file_writer.write(f"training query: {args.query}\n")
    file_writer.flush()

    ''' scale query workload if bit number is not 20. As BMTree hard codes the dataset and query.'''
    if bit_gap != 0:
        scale = 2 ** bit_gap
        for i in range(len(queryset)):
            queryset[i] = [int(item * scale) for item in queryset[i]]

    '''set up query workload: training set and testing set'''
    queryset = [Query(query) for query in queryset]
    print(queryset[0].length, queryset[0].width)

    query_card = len(queryset) # len: 999
    # random.shuffle(queryset)
    # training_queryset = queryset[: int(args.query_split_rate * query_card)]
    testing_queryset = queryset[int(args.query_split_rate * query_card):] # len:200
    training_queryset = queryset

    '''Some initial setting'''
    binary_transfer = Int2BinaryTransformer(data_space)

    '''Set up baselines and finish result generating'''

    agent_save_path = result_save_path + 'model.pt'


    
    '''Set up the environment for rl agent'''
    # ==========================================================================================
    env = BMTEnv(sampled_data, None, bit_length, None, binary_transfer, smallest_split_card, args.max_depth)
    # ==========================================================================================

    initial_action_z = 1

    '''Generate based on z order'''
    env.generate_tree_zorder(initial_action_z)
    '''set up training reward env & final performance reward env'''
    training_reward_env = ExperimentEnv(sampled_data, env.tree, module_name='bmtree', pagesize=page_size,
                                        core_num=args.core_num, is_opt_cost=args.is_opt_cost)
    performance_reward_env = ExperimentEnv(dataset, env.tree, module_name='bmtree', pagesize=page_size,
                                           core_num=args.core_num, is_opt_cost=args.is_opt_cost)

    # performance_reward_env.change_module(env.tree, module_name='bmtree')
    performance_reward_env.order_generate()
    scan_range = performance_reward_env.fast_compute_scan_range(training_queryset) # 1158.9059
    file_writer.write("Exact Result: bmtree z curve:training scan range: {:.04f}\n".format(scan_range))
    file_writer.flush()
    scan_range = performance_reward_env.fast_compute_scan_range(testing_queryset) # 1139.105
    file_writer.write("Exact Result: bmtree z curve:testing scan range: {:.04f}\n".format(scan_range))
    file_writer.flush()


    training_reward_env.order_generate()
    scan_range = training_reward_env.fast_compute_scan_range(training_queryset) # 1154.84
    file_writer.write("bmtree z curve:training scan range: {:.04f}\n".format(scan_range))
    file_writer.flush()

    '''Now we split the sampled data and whole data result, jiangneng'''
    total_z_performance = -scan_range
    best_performance = total_z_performance

    ''' use training reward environment here, jiangneng'''
    training_reward_env.order_generate()
    scan_range = training_reward_env.fast_compute_scan_range(training_queryset) # 1154.84 same nothing changed

    training_z_performance = -scan_range
    old_performance = training_z_performance

    
    best_result = ['s', 's']

    scan_range = training_reward_env.fast_compute_scan_range(testing_queryset)
    file_writer.write("bmtree z curve:testing scan range: {:.04f}\n".format(scan_range))
    file_writer.flush()

    env.reset()

    if method == 'zcurve':
        zcurve = Z_Curve_Module(binary_transfer)

        # performance_reward_env.change_module(zcurve, module_name='zcurve')
        # performance_reward_env.order_generate()
        if not os.path.exists('query/{}.json'.format(args.exp_query_file)):
            query_gen(args.data, args.exp_query_file)

        with open('query/{}.json'.format(args.exp_query_file), 'r') as f:
            queryset = json.load(f)
            
        queries = []
        for query in queryset:
            min_point = query[0:data_dim]
            max_point = query[data_dim :]
            queries.append(BlockQuery([min_point, max_point], [zcurve.output(min_point), zcurve.output(max_point)]))
            
        points = []
        for data in dataset:
            points.append(Point(data, zcurve.output(data)))

        storage = Storage(points, page_size)
        # print(storage)

        avg_block_acc_num = storage.window_query_all(queries)
        # print("avg_block_acc_num", avg_block_acc_num)
        file_writer.write("average block access num: {} \n".format(avg_block_acc_num))
        file_writer.write("-------------cost: " + cost_types[args.is_opt_cost] + " -------------\n")
        file_writer.write(get_current_branch() + "\n")
        file_writer.flush()

    if method == 'quilts':
        quilts_order = DesignQuiltsModule(binary_transfer)
        best_access = -1
        best_order_id = -1
        for order_ind in range(len(quilts_order.possible_orders)):
            quilts_order.order = quilts_order.possible_orders[order_ind]

            performance_reward_env.change_module(quilts_order, module_name='quilts')
            performance_reward_env.order_generate()

            scan_range = performance_reward_env.fast_compute_scan_range(training_queryset)
            writer.write('Quilts Order at id:{}: training query scanrange: \n'.format(order_ind, scan_range))
            scan_range = performance_reward_env.fast_compute_scan_range(testing_queryset)
            global_costs, local_costs, total_costs, page_access, access_rsmi = performance_reward_env.run_query(
                testing_queryset)
            writer.write('Quilts Order at id:{}: testing query scanrange: \n'.format(order_ind, scan_range))

            if best_order_id == -1 or best_access > page_access:
                best_access = page_access
                best_order_id = order_ind
        print('best order id: ', best_order_id)
        print('best_order access: ', best_access)

        return

    if method == 'mcts':
        file_writer.write("========================================== \n")
        file_writer.write("Start the mcts bmtree construct\n")
        file_writer.write("========================================== \n")
        file_writer.flush()

        training_reward_env.change_module(env.tree, module_name='bmtree')
        training_reward_env.order_generate()

        file_writer.write("-total_z_performance: {} \n".format(-total_z_performance))
        file_writer.write("args.max_depth: {} \n".format(args.max_depth))
        file_writer.write("args.split_depth: {} \n".format(args.split_depth))
        file_writer.flush()

        # initial mcts model
        mcts = MCTS(-total_z_performance, env.tree, max_depth=args.max_depth, split_depth=args.split_depth)

        # TODO: rollout the greedy model (if use the greedy as input, to see if performance will be better)

        best_perf = float("inf")
        best_perf_id = None

        no_change = 4

        # start construct BMTree based on mcts
        from utils.metric_compute import execution_time
        reward_calculation_time = execution_time
        # print("reward_calculation_time 1: ", reward_calculation_time)
        print("-------------cost: " + cost_types[args.is_opt_cost] + " -------------\n")
        print("-------------Start-------------")

        for episode in range(1):
            time_usage = 0

            file_writer.write('max depth of bmtree: {}\n'.format(args.max_depth))
            file_writer.write('***** episode: {} *****\n'.format(episode))
            # for i in range(int(sum(bit_length))):
            for i in range(args.action_depth):
                start_time = time.time()
                len_rollout = args.rollouts
                for roll in range(len_rollout):
                    path, scan_range = mcts.do_rollout(env.tree, training_reward_env, training_queryset)
                    file_writer.write('rollout {} scanrange {}\n'.format(roll, scan_range))
                    file_writer.flush()


                choose_actions, reward = mcts.choose(env.tree)

                env.tree.multi_action(choose_actions)
                end_time = time.time()
                time_usage += end_time - start_time
                file_writer.write('depth: {}, node_reward: {}, time use: {} choose actions: {}\n'.format(i, reward, end_time - start_time, choose_actions))
                file_writer.flush()

                training_reward_env.change_module(env.tree, module_name='bmtree')
                training_reward_env.order_generate()
                scan_range = training_reward_env.fast_compute_scan_range(training_queryset)

                if scan_range <= best_perf:
                    env.tree.save(result_save_path + 'best_tree.txt', args.max_depth)

                    # if abs(scan_range - best_perf) <= 1e-9:
                    #     if no_change > 0:
                    #         no_change -= 1
                    #         # best_perf = scan_range
                    #     else:
                    #         break
                    best_perf = scan_range

                # else:
                #     break
            from utils.metric_compute import execution_time
            reward_calculation_time = execution_time - reward_calculation_time
            # print("calculation cost: ", execution_time)
            file_writer.write("calculation cost: {} s\n".format(execution_time))

            # performance_reward_env.change_module(env.tree, module_name='bmtree')
            # performance_reward_env.order_generate()
            # scan_range = performance_reward_env.fast_compute_scan_range(training_queryset)
            # file_writer.write("bmtree mcts in episode {}: perf training scan range: {:.04f}\n".format(episode, scan_range))
            # scan_range = training_reward_env.fast_compute_scan_range(testing_queryset)
            # file_writer.write("bmtree mcts in episode {}: perf testing scan range: {:.04f}\n".format(episode, scan_range))
            file_writer.write("total use time: {} s\n".format(time_usage))
            
            file_writer.write("reward calculation time: {} s\n".format(reward_calculation_time))

            # TODO order points and do queries
            curve = env.tree

            if not os.path.exists('query/{}.json'.format(args.exp_query_file)):
                query_gen(args.data, args.exp_query_file)

            with open('query/{}.json'.format(args.exp_query_file), 'r') as f:
                queryset = json.load(f)

            # queries = []
            # for query in queryset:
            #     min_point = query[0:data_dim]
            #     max_point = query[data_dim :]
            #     queries.append(BlockQuery([min_point, max_point], [curve.output(min_point), curve.output(max_point)]))
                
            # points = []
            # for data in dataset:
            #     points.append(Point(data, curve.output(data)))

            # storage = Storage(points, page_size)
            # # print(storage)

            # avg_block_acc_num = storage.window_query_all(queries)
            # print("avg_block_acc_num", avg_block_acc_num)
            file_writer.write("query num: {} \n".format(len(training_queryset)))
            # file_writer.write("average block access num: {} \n".format(avg_block_acc_num))
            file_writer.write("-------------cost: " + cost_types[args.is_opt_cost] + " -------------\n")
            file_writer.write(get_current_branch() + "\n")
            file_writer.flush()
            env.tree.save(result_save_path + 'best_tree.txt', args.max_depth)

            # Specify the source file and destination directory
            source_file = result_save_path + 'best_tree.txt'
            destination_directory = f"./learned_sfcs/mcts_bmtree_{args.data}_{args.exp_query_file}.txt"
            # Use shutil to copy the file
            # print(source_file, destination_directory)
            shutil.copy(source_file, destination_directory)

            file_writer.flush()

            env.reset()

        return


if __name__ == "__main__":
    data_path, query_path, result_path = 'data/', 'query/', 'fast_result/'

    if not os.path.exists(result_path):
        os.mkdir(result_path)

    result_save_path = result_path + "{}_{}/".format(args.data, args.query)

    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)

    result_save_path = result_save_path + "{}/".format(args.method)

    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)


    result_save_path = result_save_path + "{}_{}_{}_{}_{}".format( args.data_sample_points,
                                                                        args.smallest_split_card, args.max_depth, args.action_depth, args.rollouts)

    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
        result_save_path = result_save_path + '/0/'
        os.mkdir(result_save_path)
    else:
        file_list = os.listdir(result_save_path)
        file_time = [int(x) for x in file_list]
        new_time = max(file_time) + 1
        result_save_path = result_save_path + '/{}/'.format(new_time)
        os.mkdir(result_save_path)

    '''the writer store information inside the result file with actor/critic learning rate'''
    writer = open(result_save_path + 'result_{}.txt'.format(args.result_appendix), 'w')

    experiment_ppo(result_save_path, writer, bit_gap=bit_length[0] - 20)
