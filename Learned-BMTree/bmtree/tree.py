# Copyright (c) 2022. This code belongs to Jiangneng Li, Nanyang Technological University.

import math
import random
import numpy as np
import re
import sys
import torch
from torch.distributions import Categorical
import struct
import copy
import pprint
from utils.query import Query

pow_of_two = [int(pow(2, i)) for i in range(32)]


def generate_value_from_binary(data_point):
    """
    This function returns the decimal number of data_point
    :param data_point:
    :return: decimal number
    """
    value = 0
    for i in range(len(data_point)):
        value = value * 2 + data_point[i]
    return value


def compute_child_range(dimension_choose, space_range, chose_action):
    '''
    update the dimension_choose for left and right children
    '''
    dimension_choose_new = [i for i in dimension_choose]  # i is number, is ok

    space_range_left = [[j for j in i] for i in space_range]
    space_range_left[chose_action * 2 + 1][dimension_choose_new[chose_action]] = 0

    space_range_right = [[j for j in i] for i in space_range]
    space_range_right[chose_action * 2][dimension_choose_new[chose_action]] = 1

    dimension_choose_new[chose_action] += 1

    return dimension_choose_new, space_range_left, space_range_right


def z_order_heuristic(parent_action, mask):
    """
    This function provide a z-order heuristic(policy) that guide the tree heuristicly constructed
    :param parent_action:
    :param mask:
    :return:
    """

    '''Compute the action space'''
    action_space = len(mask)

    '''Rounding the possible next action'''
    next_action = (parent_action + 1) % action_space

    '''To forbid endless loop, one should make sure there's at least one action too take'''
    while True:
        if mask[next_action] == 1:
            return next_action
        else:
            next_action = (next_action + 1) % action_space
    pass


class ZNode:
    def __init__(self, bit_length):
        """
        ZNode is a node for heuristic based value computing
        """

        '''add node type, heuristic or normal'''
        self.type = 'heuristic'

        self.bit_length = bit_length

    def get_value_queries(self, key_values_list, parent_values, parent_action, dimension_choose):
        values = copy.deepcopy(parent_values)

        '''rest_time store how many bits left to be merged'''
        rest_time = sum(self.bit_length) - sum(dimension_choose)
        assert rest_time >= 0, 'dimension choose more than bit length'
        mask = [1 for i in range(len(dimension_choose))]
        while True:
            if rest_time == 0:
                break

            '''compute the next action with mask all 1'''
            next_action = z_order_heuristic(parent_action, mask)

            '''then if it's merge-able then merge this bit, change related information'''
            if dimension_choose[next_action] != self.bit_length[next_action]:
                for i, key_values in enumerate(key_values_list):
                    bit_value = key_values[next_action][
                        dimension_choose[next_action]]

                    values[i] = values[i] * 2 + bit_value

                dimension_choose[next_action] += 1
                rest_time -= 1

            parent_action = next_action

        return values, self

    def get_value(self, key_values, parent_value, parent_action, dimension_choose):
        """
        This function compute the value based on Heuristic(Z order rule)
        :param key_values:
        :param parent_value:
        :param parent_action:
        :param dimension_choose:
        :return: value
        """

        '''store the value'''
        value = parent_value

        '''rest_time store how many bits left to be merged'''
        rest_time = sum(self.bit_length) - sum(dimension_choose)
        assert rest_time >= 0, 'dimension choose more than bit length'

        mask = [1 for i in range(len(dimension_choose))]
        while True:
            if rest_time == 0:
                break

            '''compute the next action with mask all 1'''
            next_action = z_order_heuristic(parent_action, mask)

            '''then if it's merge-able then merge this bit, change related information'''
            if dimension_choose[next_action] != self.bit_length[next_action]:
                bit_value = key_values[next_action][
                    dimension_choose[next_action]]

                value = value * 2 + bit_value

                dimension_choose[next_action] += 1
                rest_time -= 1

            parent_action = next_action

        return value, self



class Node:
    def __init__(self, node_id: int, parent_node_id: int, depth: int,
                 space_range, dimension_choose, curve_range, h_node, bit_length):  # , queries: list):
        """
        Initiation a Node object as composition of Tree object
        :param node_id: unique ID of each node, also used a traversal Num in the NodeSet
        :param depth: The depth wrt. the Tree object
        :param space_range: Space Range that the node represent of, every dimension is composed of St,End
        :param dimension_choose: show the bit been chosen of each dimension
        """

        '''add node's type, normal or heuristic'''
        self.type = 'normal'

        self.node_id = node_id
        self.parent_node_id = parent_node_id
        self.depth = depth

        self.bit_length = bit_length

        '''Store the action that choose which dimension'''
        self.chosen_bit = None  # bit chosen

        self.left_children = None
        self.right_children = None  # left and right children
        self.child = None  # child if not split
        # self.queries = queries  # queries fall in the node

        self.space_range = space_range

        '''Here we add curve range here'''
        self.curve_range = curve_range
        # self.curve_range_left, self.curve_range_right = None, None
        self.dimension_choose = dimension_choose

        '''Reward comes from children, leaf node reward come from outside'''
        self.reward = 0

        self.heuristic_node = h_node

        self.if_split = 1

        self.table = []

        self.query_volumn = 0

        pass

    def compare_bit_list(self, list1, list2):
        for i in range(len(list1)):
            if list1[i] > list2[i]:
                return True
            elif list1[i] < list2[i]:
                return False
        return True

    def is_point_in_space(self, point, space):
        x_left_bound = space[0]
        x_right_bound = space[1]
        y_left_bound = space[2]
        y_right_bound = space[3]
        x = point[0]
        y = point[1]
        if (self.compare_bit_list(x, x_left_bound) and self.compare_bit_list(x_right_bound, x)) \
                and (self.compare_bit_list(y, y_left_bound) and self.compare_bit_list(y_right_bound, y)):
            return True
        else:
            return False

    def cal_value_by_bit_list(self, bit_list):
        res = 0
        bit_len = len(bit_list)
        for i in range(bit_len):
            res += bit_list[bit_len - 1 - i] * pow_of_two[i]
        return res

    def is_split(self, min_point, max_point, space_range_left, space_range_right):
        '''
        This function has been tested. The query can be correctly partitioned.
        '''
        is_in_left = self.is_point_in_space(min_point, space_range_left)
        is_in_right = self.is_point_in_space(max_point, space_range_right)

        partition_length = 0
        # print("is_in_left", is_in_left)
        # print("is_in_right", is_in_right)
        if is_in_right and is_in_left:
            # print("is_in_right and is_in_left", " self.chosen_bit", self.chosen_bit)
            if self.chosen_bit == 0:
                split_line_left = space_range_left[1]
                split_line_right = space_range_right[0]
                middle_point_min = [split_line_left, max_point[1]]
                middle_point_max = [split_line_right, min_point[1]]
                # middle_point_min = [split_line_left, max_point[1]]
                # middle_point_max = [split_line_right, min_point[1]]
                partition_length = self.cal_value_by_bit_list(max_point[1]) - self.cal_value_by_bit_list(min_point[1])

            else:
                split_line_left = space_range_left[3]
                split_line_right = space_range_right[2]
                middle_point_min = [max_point[0], split_line_left]
                middle_point_max = [min_point[0], split_line_right]
                # middle_point_min = [max_point[0], split_line_left]
                # middle_point_max = [min_point[0], split_line_left]
                partition_length = self.cal_value_by_bit_list(max_point[0]) - self.cal_value_by_bit_list(min_point[0])

            # split_line = space_range_left[2] if self.chosen_bit == 0 else space_range_left[3]
            # middle_point_min = [split_line, max_point[1]] if self.chosen_bit == 0 else [max_point[0], split_line]
            # middle_point_max = [split_line, min_point[1]] if self.chosen_bit == 0 else [min_point[0], split_line]
            # print("min_point", min_point)
            # print("middle_point_min", middle_point_min)
            # print("middle_point_max", middle_point_max)
            # print("max_point", max_point)

            return partition_length, None, min_point, middle_point_min, middle_point_max, max_point
        else:
            return partition_length, is_in_left, min_point, None, None, max_point

    def calculate_drop_pattern(self, query, drop_dim: int, drop_index: int):
        """
            a * 2 ^k - 1 -------> (a - 1) * 2^k
        """
        dimension_high = query.max_point
        dimension_low = query.min_point
        if dimension_high[drop_dim] - dimension_low[drop_dim] < pow_of_two[drop_index]:
            return 0
        end = math.floor((dimension_high[drop_dim] + 1) / pow_of_two[drop_index])
        start = math.ceil(dimension_low[drop_dim] / pow_of_two[drop_index]) + 1
        return end - start + 1
    
    def calculate_rise_pattern(self, query, rise_dim: int, rise_index: int):
        """
            a * 2 ^k + (2^{k - 1} - 1) -------> a * 2 ^k + 2^{k - 1}
        """
        dimension_high = query.max_point
        dimension_low = query.min_point
        start = math.ceil((dimension_low[rise_dim] - pow_of_two[rise_index - 1] + 1) / pow_of_two[rise_index])
        end = math.floor((dimension_high[rise_dim] - pow_of_two[rise_index - 1]) / pow_of_two[rise_index])
        return end - start + 1

    def bit_to_int(self, bit_list):
        res = 0
        bit_len = len(bit_list)
        for i in range(bit_len):
            res += bit_list[bit_len - 1 - i] * pow_of_two[i]
        return res

    def fill_out_tables(self, min_points, max_points): # fill out the table    
        
        if len(self.table) > 0:
            return

        queries = []
        for i in range(len(min_points)):
            
            query_list = [self.bit_to_int(min_points[i][0]), self.bit_to_int(min_points[i][1]),
                          self.bit_to_int(max_points[i][0]), self.bit_to_int(max_points[i][1])]
            
            # if (i == 0):
            # print("min_points[i][0]", min_points[i][0])
            # print("min_points[i][0]", min_points[i][1])
            queries.append(Query(query_list))
            self.query_volumn += queries[i].area

        # bit_nums = [20, 20]#[len(min_points[0][0]), min_points[0][1]]
        # dim = len(bit_nums)
        # column_lens = [sum(bit_nums) - bit_nums[i] + 1 for i in range(dim)]
        # row_lens = bit_nums
        row_len = self.bit_length[0]
        dim = len(self.bit_length)
        column_len = self.bit_length[0] + 1

        self.table.append([[0 for j in range(column_len)] for i in range(row_len)])
        self.table.append([[0 for j in range(column_len)] for i in range(row_len)])
        
        for query in queries:
            drop_patterns = []
            for i in range(dim):
                dim_patterns = []
                for changed_bits in range(column_len):
                    dim_patterns.append(self.calculate_drop_pattern(query, i, changed_bits))
                drop_patterns.append(dim_patterns)
        
            rise_patterns = []
            for i in range(dim):
                dim_patterns = []
                for changed_bits in range(row_len):
                    dim_patterns.append(self.calculate_rise_pattern(query, i, changed_bits + 1))
                rise_patterns.append(dim_patterns)

            for i in range(dim):
                rise_dim = i
                drop_dim = dim - 1 - i
                for rise_index in range(row_len):
                    # cal rise pattern
                    for drop_index in range(column_len):
                        rise_res = rise_patterns[rise_dim][rise_index]
                        drop_res = drop_patterns[drop_dim][drop_index]
                        # if rise_index <= 1 and drop_index <= 1:
                            # print("rise_index", rise_index, "drop_index", drop_index, "rise_res", rise_res, "drop_res", drop_res)
                        self.table[rise_dim][rise_index][drop_index] += rise_res * drop_res
            break


    def read_lookup_tables(self, min_points, max_points, actions, dimension_choose):
        self.fill_out_tables(min_points, max_points)
        x_reading_map = {}
        y_reading_map = {}
        actions = copy.deepcopy(actions) 
        BMC_len = len(min_points[0][0])
        x_choosed = len(actions) - sum(actions)
        y_choosed = sum(actions)
        counter = [BMC_len - x_choosed, BMC_len - y_choosed]
        start = actions[-1]
        while counter[0] > 0 and counter[1] > 0:
            if start == 0:
                actions.append(1)
                counter[1] -= 1
                start = 1
            else:
                actions.append(0)
                counter[0] -= 1
                start = 0

        while counter[0] > 0:
            actions.append(0)
            counter[0] -= 1
        
        while counter[1] > 0:
            actions.append(1)
            counter[1] -= 1

        counter = [BMC_len, BMC_len]
        for i in range(len(actions)):
            if actions[i] == 0:
                x_reading_map[counter[0] - 1] = counter[1]
                counter[0] -= 1
            else:
                y_reading_map[counter[1] - 1] = counter[0]
                counter[1] -= 1
        connection_num = 0
        for i in range(BMC_len):
            connection_num += self.table[0][i][x_reading_map[i]]
            connection_num += self.table[1][i][y_reading_map[i]]
        
        volume = 0
        for i in range(len(min_points)):
            volume += (self.bit_to_int(max_points[i][0]) - self.bit_to_int(min_points[i][0])) * (
                self.bit_to_int(max_points[i][1]) - self.bit_to_int(min_points[i][1]))
        return volume, connection_num, self


    def cal_local_cost(self, min_points, max_points, actions):
        # if self.chosen_bit is None:
        #     return parent_value, self
        # else:
        # dim_index = self.chosen_bit
        # dim_value = self.dimension_choose[dim_index]
        actions = copy.deepcopy(actions)
        dimension_choose_new, space_range_left, space_range_right = \
                            compute_child_range(self.dimension_choose, self.space_range, self.chosen_bit)
        # pprint.pprint(dimension_choose_new)
        # pprint.pprint(space_range_left)
        # pprint.pprint(space_range_right)
        left_min_points = []
        left_max_points = []
        right_min_points = []
        right_max_points = []
        left_res = 0
        right_res = 0
        left_volume = 0
        right_volume = 0

        cost_make_up = 0

        if self.chosen_bit is None:
            return 0, self
        else:
            actions.append(self.chosen_bit)
            for max_point, min_point in zip(max_points, min_points):
                partition_length, is_in_left, min_point, middle_point_min, middle_point_max, max_point = self.is_split(min_point, max_point, space_range_left, space_range_right)
                # partition can increase the cost by adding two boundaries
                cost_make_up += partition_length
            # check if we need to split the query
                if partition_length > 0:
                    left_min_points.append(min_point)
                    left_max_points.append(middle_point_min)
                    right_min_points.append(middle_point_max)
                    right_max_points.append(max_point)
                else:
                    if is_in_left:
                        left_min_points.append(min_point)
                        left_max_points.append(max_point)
                    else:
                        right_min_points.append(min_point)
                        right_max_points.append(max_point)

        
        if len(left_min_points) > 0:
            if self.left_children is None or self.left_children.chosen_bit is None:
                if self.child is not None:
                    if self.child.chosen_bit is None:
                        left_volume, left_res, _ = self.read_lookup_tables(left_min_points, left_max_points, actions, dimension_choose_new)
                    else:
                        left_volume, left_res, _ = self.child.cal_local_cost(left_min_points, left_max_points, actions)
                else:
                    left_volume, left_res, _ = self.read_lookup_tables(left_min_points, left_max_points, actions, dimension_choose_new)
            else:
                left_volume, left_res, _ = self.left_children.cal_local_cost(left_min_points, left_max_points, actions)

        if len(right_min_points) > 0:
            if self.right_children is None or self.right_children.chosen_bit is None:
                if self.child is not None:
                    if self.child.chosen_bit is None:
                        right_volume, right_res, _  = self.read_lookup_tables(right_min_points, right_max_points, actions, dimension_choose_new)
                    else:
                        right_volume, right_res, _  = self.child.cal_local_cost(right_min_points, right_max_points, actions)
                else:
                    right_volume, right_res, _  = self.read_lookup_tables(right_min_points, right_max_points, actions, dimension_choose_new)
            else:
                right_volume, right_res, _  = self.right_children.cal_local_cost(right_min_points, right_max_points, actions)
        # print("volume:", right_volume + left_volume, "segments:", left_res + right_res, "cost_make_up:", cost_make_up)
        return right_volume + left_volume + cost_make_up, left_res + right_res + cost_make_up * 2 , self

    def get_value_queries(self, key_values_list, parent_values):
        bit_values_0 = []
        bit_values_1 = []
        parent_values_0 = []
        parent_values_1 = []
        key_values_list_0 = []
        key_values_list_1 = []
        dimension_choose_new, space_range_left, space_range_right = \
                            compute_child_range(self.dimension_choose, self.space_range, self.chosen_bit)

        if self.chosen_bit is None:
            return parent_values, self
        else:
            
            for i, key_values in enumerate(key_values_list):
                bit_value = key_values[self.chosen_bit][
                        self.dimension_choose[self.chosen_bit]]
                if bit_value == 0:
                    bit_values_0.append(bit_value)
                    parent_values_0.append(2 * parent_values[i] + bit_value)
                    key_values_list_0.append(key_values)
                else:
                    bit_values_1.append(bit_value)
                    parent_values_1.append(2 * parent_values[i] + bit_value)
                    key_values_list_1.append(key_values)

        value_summary_0 = []  
        value_summary_1 = []

        if len(parent_values_0) > 0:
            if (self.left_children is None) or (self.left_children.chosen_bit is None):
                if self.child is not None:
                    if self.child.chosen_bit is None:
                        dimension_choose_new, space_range_left, space_range_right = \
                            compute_child_range(self.dimension_choose, self.space_range, self.chosen_bit)
                        parent_action = self.chosen_bit
                        value_summary_0, _ = self.heuristic_node.get_value_queries(key_values_list_0, parent_values_0, parent_action, dimension_choose_new)
                    else:
                        value_summary_0, _ =  self.child.get_value_queries(key_values_list_0, parent_values_0)
                else:
                    dimension_choose_new, space_range_left, space_range_right = \
                        compute_child_range(self.dimension_choose, self.space_range, self.chosen_bit)
                    parent_action = self.chosen_bit
                    value_summary_0, _ = self.heuristic_node.get_value_queries(key_values_list_0, parent_values_0, parent_action, dimension_choose_new)
            else:
                value_summary_0, _ = self.left_children.get_value_queries(key_values_list_0, parent_values_0)

        if len(parent_values_1) > 0:
            if (self.right_children is None) or self.right_children.chosen_bit is None:

                if self.child is not None:
                    if self.child.chosen_bit is None:
                        dimension_choose_new, space_range_left, space_range_right = \
                            compute_child_range(self.dimension_choose, self.space_range, self.chosen_bit)
                        parent_action = self.chosen_bit
                        value_summary_1, _ = self.heuristic_node.get_value_queries(key_values_list_1, parent_values_1, parent_action, dimension_choose_new)
                    else:
                        value_summary_1, _ = self.child.get_value_queries(key_values_list_1, parent_values_1)
                else:
                    dimension_choose_new, space_range_left, space_range_right = \
                        compute_child_range(self.dimension_choose, self.space_range, self.chosen_bit)
                    parent_action = self.chosen_bit
                    value_summary_1, _ = self.heuristic_node.get_value_queries(key_values_list_1, parent_values_1, parent_action, dimension_choose_new)
            else:
                value_summary_1, _ = self.right_children.get_value_queries(key_values_list_1, parent_values_1)

        value_summary_0.extend(value_summary_1)
        return value_summary_0, self



    def get_value(self, key_values, parent_value):
        """
        This function serves when recursively visit one Tree path.
        :param key_values: The key values are consist of dimension based bits-lists
        :param parent_value: Parent value is value computed by the parent nodes
        :return: the curve value of a data_point
        """

        '''Compute new value follow binary computing rule'''
        if self.chosen_bit is None:
            return parent_value, self
        else:
            # try:
            # print(self.depth, parent_value, self.chosen_bit, self.dimension_choose[self.chosen_bit])
            bit_value = key_values[self.chosen_bit][
                    self.dimension_choose[self.chosen_bit]]
            # except:


        '''compute new value'''
        new_value = 2 * parent_value + bit_value

        '''
        If value at chosen_bit is None, represent a leaf node
        else see if bit_value 0 -> scan left child, else scan right child
        ** Here we add leaf node return itself, as a method to compute reward in update session
        '''

        if bit_value == 0:

            '''Debug when the tree structure is not complete and want to get performance'''
            if (self.left_children is None) or (self.left_children.chosen_bit is None):

                if self.child is not None:
                    if self.child.chosen_bit is None:
                        dimension_choose_new, space_range_left, space_range_right = \
                            compute_child_range(self.dimension_choose, self.space_range, self.chosen_bit)
                        parent_action = self.chosen_bit
                        return self.heuristic_node.get_value(key_values, new_value, parent_action, dimension_choose_new)
                    return self.child.get_value(key_values, new_value)

                dimension_choose_new, space_range_left, space_range_right = \
                    compute_child_range(self.dimension_choose, self.space_range, self.chosen_bit)
                parent_action = self.chosen_bit
                return self.heuristic_node.get_value(key_values, new_value, parent_action, dimension_choose_new)

            else:
                return self.left_children.get_value(key_values, new_value)
        else:
            if (self.right_children is None) or self.right_children.chosen_bit is None:

                if self.child is not None:
                    if self.child.chosen_bit is None:
                        dimension_choose_new, space_range_left, space_range_right = \
                            compute_child_range(self.dimension_choose, self.space_range, self.chosen_bit)
                        parent_action = self.chosen_bit
                        return self.heuristic_node.get_value(key_values, new_value, parent_action, dimension_choose_new)
                    return self.child.get_value(key_values, new_value)

                dimension_choose_new, space_range_left, space_range_right = \
                    compute_child_range(self.dimension_choose, self.space_range, self.chosen_bit)
                parent_action = self.chosen_bit

                return self.heuristic_node.get_value(key_values, new_value, parent_action, dimension_choose_new)
            else:
                return self.right_children.get_value(key_values, new_value)

    def add_children(self, children, type):
        """
        :param children: Node object
        :param type: String object, 'left' or 'right'
        :return: None
        """
        if type == 'left':
            self.left_children = children
        if type == 'right':
            self.right_children = children

        return

    def get_mask(self, bit_length):
        '''Get current node's dimension_choose info'''
        dimension_choose = self.dimension_choose

        '''Store the number of available actions'''
        count = 0

        '''
        Computing mask which enable the action taken smoothly.
        recompute the action with a mask and sampled it by probability
        '''
        mask = []

        for i in range(len(bit_length)):
            if dimension_choose[i] == bit_length[i]:
                mask.append(0)
            else:
                mask.append(1)
                count += 1
        mask = torch.tensor(mask)

        return mask, count

    def get_state(self):
        """
        This function returns the state input of Reinforcement Learning Agent
        :return: a torch tensor
        """

        # '''Generate state that follows binary tensor'''
        # states = [torch.tensor(a) for a in self.space_range]
        # state = torch.cat(states, dim=0).type(torch.FloatTensor)

        '''Generate state that follows decimalism'''
        space_value = []
        for i in range(len(self.space_range)):
            value = generate_value_from_binary(self.space_range[i])
            space_value.append(value)

        state = torch.tensor(space_value).type(torch.FloatTensor)

        return state

    def compute_reward(self):
        """
        This function compute, save and return recursively calculated
        :return: reward represent this node
        """

        '''check if is leaf node'''
        if self.chosen_bit is None:
            return self.reward

        '''When is root node or inner node'''
        reward = self.left_children.compute_reward() + \
            self.right_children.compute_reward()

        self.reward = reward

        return reward

    def update_heuristic(self, parent_action, bit_length, heuristic_rule, compute_child_range):

        '''This is when in the leaf node'''
        if self.chosen_bit is None:
            return

        mask, avaliable_count = self.get_mask(bit_length)

        new_action = heuristic_rule(parent_action, mask)

        self.chosen_bit = new_action

        chose_action = self.chosen_bit
        dimension_choose = self.dimension_choose
        space_range = self.space_range

        dimension_choose_new, space_range_left, space_range_right = \
            compute_child_range(dimension_choose, space_range, chose_action)

        self.left_children.space_range = space_range_left
        self.left_children.dimension_choose = dimension_choose_new
        self.right_children.space_range = space_range_right
        self.right_children.dimension_choose = dimension_choose_new

        self.left_children.update_heuristic(self.chosen_bit, bit_length, heuristic_rule,
                                            compute_child_range)
        self.right_children.update_heuristic(self.chosen_bit, bit_length, heuristic_rule,
                                             compute_child_range)

        return

    pass


class Tree:
    def __init__(self, *args, **kwargs):
        """
        :param args:
        :param kwargs:
        Description: Initiation of a BmTree object. the parameters *args **kwargs provides two lists with
        input parameters, which serve the functionality of overloading.
        """
        list_args = kwargs
        if 'Tree' in list_args.keys():
            # TODO construct a tree wrt the old tree
            pass
        elif 'bit_length' in list_args.keys():

            '''bit_length [bit_1, bit2, .., bitn] regards to the length of bit towards each dimension'''
            bit_length = list_args['bit_length']
            self.bit_length = bit_length

            '''dataset is set inside the tree structure and will function in action'''
            dataset = list_args['dataset']
            self.dataset = dataset # directly assign self.dataset = dataset

            '''This is the constant of a Tree'''
            smallest_split_card = list_args['smallest_split_card']
            self.smallest_split_card = smallest_split_card

            self.max_depth = list_args['max_depth']

            '''Compute the curve space range, which is the whole dataset'''
            curve_range = [0, len(dataset) - 1]

            self.heuristic_node = ZNode(self.bit_length)

            '''
            Generate initiate space range, the whole space wrt. the bit_length of 
            each dimension, which is set at the beginning of Tree initiation.
            [[0,0,0,0,..], [1,1,1,1,...]] as initiated.
            '''
            space_range = []
            for i in range(len(self.bit_length)):
                space_range.append([0 for rec in range(self.bit_length[i])])
                space_range.append([1 for rec in range(self.bit_length[i])])

            '''Generate a normalizing scale for normalizing state input'''
            self.normalize_scale = []
            for i in range(len(self.bit_length)):
                value = generate_value_from_binary(space_range[2 * i + 1])
                self.normalize_scale.append(value)
                self.normalize_scale.append(value) # do it twice
            self.normalize_scale = torch.tensor(self.normalize_scale)

            self.id_count = 0

            '''
            Generate root node representing Whole Space, save node into the node list wrt to node_id.
            The node_id add itself 1 respectively.
            '''
            self.root_node = Node(self.node_id_gen(), -1,  0, space_range,
                                  [0 for i in range(len(self.bit_length))], curve_range=curve_range, h_node=self.heuristic_node, bit_length=self.bit_length)  # root node

            '''Initiate the foremost states'''
            self.states = self.root_node.get_state() # / self.normalize_scale

            '''Set up the the node stack for action '''
            self.node_to_split = [self.root_node]  # stack store node to split
            self.current_node = self.root_node
            '''Initialize actioned nodes, in order to unaction'''
            self.actioned_node = []
            
            '''
            set the node set with a list object. the sort of nodes should be identical to the
            node_id assigned to nodes
            '''
            self.nodes = [self.root_node]

        '''If there's a binary_transfer, set self.bianry_transfer'''
        if 'binary_transfer' in list_args.keys():
            self.binary_transfer = list_args['binary_transfer']

        ''' Look-up tables for local cost'''
        # self.table = []

    def get_all_states(self):

        if len(self.actioned_node) == 0:
            actions = ''
        else:
            actions = []
            for node in self.actioned_node:
                actions.append(node.chosen_bit)
                actions.append(node.if_split)
            actions = ''.join([str(i) for i in actions])

        masks = []
        counts = []
        for node in self.node_to_split:
            mask, available_count = node.get_mask(self.bit_length)
            masks.append(mask)
            # masks.append(mask.tolist())
            counts.append(available_count)
        return actions, masks, counts

    def generate_children_action(self):

        actions, masks, counts = self.get_all_states()
        # masks = list(masks)
        # print(masks)

        def get_possible_actions(all_possible_actions, action, masks):
            # print(action, masks)
            for i, ava in enumerate(masks[0]):
                if ava != 0:
                    act = [] + action
                    act.append(i)
                    if len(masks) == 1:
                        all_possible_actions.append(''.join([str(ac) for ac in act]))
                        # print(len(all_possible_actions))
                    else:
                        get_possible_actions(all_possible_actions, act, masks[1:])
            if len(masks) == 1:
                return

        action = []
        all_possible_actions = []
        get_possible_actions(all_possible_actions, action, masks)
        return all_possible_actions
        # return set([actions + i for i in all_possible_actions])

    def sample_children_action(self, sample_num):
        actions, masks, counts = self.get_all_states()

        sampled_actions = []

        rest_sample_num = sample_num
        while rest_sample_num > 0:
            action_list = []
            for mask in masks:
                prob = mask / mask.sum(dim=0)
                dist = Categorical(prob)
                action = dist.sample().item()
                action_list.append(action)
            result_action = ''.join([str(ac) for ac in action_list])
            if result_action in sampled_actions:
                continue
            else:
                sampled_actions.append(result_action)
                rest_sample_num -= 1
        return sampled_actions
        # return set([actions + i for i in sampled_actions])


    def multi_action(self, actions):

        if len(actions) == 0:
            return

        for i in range(int(len(actions)/2)):
            _, _, _, _ = self.take_action(int(actions[2 * i]), int(actions[2 * i + 1]))
        return

    def get_mask(self):
        """
        This function get the action mask which come from the current node
        :return: mask tensor for action selection
        """

        if self.current_node is None:
            return None, 0

        mask, available_count = self.current_node.get_mask(self.bit_length)

        return mask, available_count

    def take_action(self, action, if_split=1):
        """
        action is generated by agent, input action and construct the tree one step further
        :param action: decision made by agent, normalized probabilities to different arms
        :return: signal, if done, maybe return new states
        """

        '''
        Take out the current node for splitting
        '''
        node = self.current_node
        space_range = node.space_range
        dimension_choose = node.dimension_choose

        '''add curve range here'''
        curve_range = node.curve_range

        '''Since the mask has been used inside the PPO, the chose_action is already the action itself'''
        chose_action = action

        '''Store action as chosen_bit for data-point value computing afterwards'''
        node.chosen_bit = chose_action

        node.if_split = if_split

        # TODO: add the dimension_choose for nonsplit node,
        # if node.depth >= self.max_depth:
        if if_split == 0:# or node.depth >= self.max_depth:
            dimension_choose_new = [i for i in dimension_choose]
            dimension_choose_new[chose_action] += 1

            '''Here see the split flag, put new node to split stack if flag = 1'''
            split_flag = 0
            for i in range(len(dimension_choose_new)):
                if dimension_choose_new[i] != self.bit_length[i]:
                    split_flag = 1
                    break

            child = Node(self.node_id_gen(), self.current_node.node_id, node.depth + 1, space_range,
                              dimension_choose_new, curve_range=curve_range, h_node=self.heuristic_node, bit_length=self.bit_length)
            self.nodes.append(child)
            node.child = child

            if split_flag:
                self.node_to_split.append(child)
            '''Pop the current node out, if the to-split list si empty, return done=1'''
            self.node_to_split.pop(0)

            # Add the actioned node to actioned node list
            self.actioned_node.append(node)

            '''Get rewards if the tree generation is finished'''
            if len(self.node_to_split) == 0:
                done = True
                states = None  # None when it ends

                rewards = None

                self.current_node = None
            else:
                self.current_node = self.node_to_split[0]
                states = self.current_node.get_state()  # / self.normalize_scale
                done = False
                rewards = None

            '''Updating the states wrt. the current node'''
            self.states = states

            return states, rewards, done, curve_range


        '''
        update the dimension_choose for left and right children
        '''
        dimension_choose_new, space_range_left, space_range_right = \
            compute_child_range(dimension_choose, space_range, chose_action)

        '''Get compare value and sort inside the dataset list by the chosen dimension'''
        compare_value = generate_value_from_binary(space_range_left[chose_action * 2 + 1])
        # self.dataset[curve_range[0]:curve_range[1] + 1].sort(key=lambda x: x[chose_action])
        subset = self.dataset[curve_range[0]:curve_range[1] + 1]
        subset.sort(key=lambda x: x[chose_action])
        self.dataset[curve_range[0]:curve_range[1] + 1] = subset

        '''Need to know that the generated curve_range could be make nonsense where
        curve_range[0] > curve_range[1], which means the child node does not have any node anymore.
        In next put, we say if there's not more than 4 data points, will directly direct to the ZNode and not put to
        the stack list anymore. and no more agent action will be down afterwards.
        '''
        curve_range_left, curve_range_right = [], []
        compare_flag = 0
        for i in range(curve_range[0], curve_range[1] + 1):
            if self.dataset[i][chose_action] > compare_value:
                curve_range_left = [curve_range[0], i - 1]
                curve_range_right = [i, curve_range[1]]
                compare_flag = 1
                break
        if compare_flag == 0:
            curve_range_left = list(curve_range)
            curve_range_right = [curve_range[1], curve_range[1] - 1]

        # node.curve_range_left, node.curve_range_right = curve_range_left, curve_range_right

        '''Here see the split flag, put new node to split stack if flag = 1, >1 means more then one dimension to choose'''
        split_flag = 0
        for i in range(len(dimension_choose_new)):
            if dimension_choose_new[i] != self.bit_length[i]:
                split_flag += 1
                # break

        '''Generate left/right children split by the action,
            link the split node to parent, store nodes in node-set
        '''
        # if curve_range_left[1] - curve_range_left[0] >= self.smallest_split_card:
        left_child = Node(self.node_id_gen(), self.current_node.node_id, node.depth + 1, space_range_left,
                          dimension_choose_new, curve_range=curve_range_left, h_node=self.heuristic_node, bit_length=self.bit_length)
        self.nodes.append(left_child)
        node.add_children(left_child, 'left')
        if split_flag > 1:
            self.node_to_split.append(left_child)

        '''Update, if child is None, then it's linked to '''
        # if curve_range_right[1] - curve_range_right[0] >= self.smallest_split_card:
        right_child = Node(self.node_id_gen(), self.current_node.node_id, node.depth + 1, space_range_right,
                           dimension_choose_new, curve_range=curve_range_right, h_node=self.heuristic_node, bit_length=self.bit_length)
        self.nodes.append(right_child)
        node.add_children(right_child, 'right')
        if split_flag > 1:
            self.node_to_split.append(right_child)

        '''Pop the current node out, if the to-split list si empty, return done=1'''
        self.node_to_split.pop(0)
        
        # Add the actioned node to actioned node list
        self.actioned_node.append(node)
        
        '''Get rewards if the tree generation is finished'''
        if len(self.node_to_split) == 0:
            done = True
            states = None  # None when it ends

            rewards = None

            self.current_node = None
        else:
            self.current_node = self.node_to_split[0]
            states = self.current_node.get_state() # / self.normalize_scale
            done = False
            rewards = None

        '''Updating the states wrt. the current node'''
        self.states = states

        return states, rewards, done, curve_range

    def unaction(self):
        """
        undo the former action
        :return: None
        """
        if len(self.actioned_node) == 0:
            return

        node = self.actioned_node[-1]

        node.left_children = None
        node.right_children = None
        node.child = None

        for i in reversed(range(len(self.node_to_split))):
            if self.node_to_split[i].parent_node_id == node.node_id:
                del self.node_to_split[i]
            else:
                break
        for i in reversed(range(len(self.nodes))):
            if self.nodes[i].parent_node_id == node.node_id:
                del_node = self.nodes[i]
                del self.nodes[i]
                del del_node
                self.id_count -= 1 # delete the used node id
            else:
                break

        self.node_to_split.insert(0, node)
        self.current_node = node

        del self.actioned_node[-1]

        return

    def unstep(self, number):
        for i in range(number):
            self.unaction()

    def update_bit(self, action):
        """
        This function update bits when a new action is given onwards a node
        :param action:
        :return:
        """
        node = self.current_node

        '''Update child nodes if action different'''
        if action != node.chosen_bit:
            space_range = node.space_range
            dimension_choose = node.dimension_choose

            '''Since the mask has been used inside the PPO, the chose_action is already the action itself'''
            chose_action = action

            '''Store action as chosen_bit for data-point value computing afterwards'''
            node.chosen_bit = chose_action

            '''
            update the dimension_choose for left and right children
            '''
            dimension_choose_new, space_range_left, space_range_right = \
                compute_child_range(dimension_choose, space_range, chose_action)

            '''Children update new action '''
            node.left_children.space_range = space_range_left
            node.left_children.dimension_choose = dimension_choose_new
            node.right_children.space_range = space_range_right
            node.right_children.dimension_choose = dimension_choose_new

            node.left_children.update_heuristic(node.chosen_bit, self.bit_length, z_order_heuristic,
                                                compute_child_range)
            node.right_children.update_heuristic(node.chosen_bit, self.bit_length, z_order_heuristic,
                                                 compute_child_range)

        '''
        Check if the node is split-able
        '''
        if node.left_children.chosen_bit is not None:
            self.node_to_split.append(node.left_children)
            self.node_to_split.append(node.right_children)

        '''Now, pop out the current node'''
        self.node_to_split.pop(0)

        if len(self.node_to_split) == 0:
            done = True
            states = None  # None when it ends

            '''
            ******* ATTENTION, here we no more compute reward 
            before the query-data environment
            is done with query-excution               *******
            '''
            self.current_node = None
        else:
            self.current_node = self.node_to_split[0]
            states = self.current_node.get_state()  # / self.normalize_scale
            done = False

        '''Updating the states wrt. the current node'''
        self.states = states

        return states, done

    def node_id_gen(self):
        """
        This function returns a ID for the new generated node
        :return: the new node's ID
        """
        self.id_count += 1
        return self.id_count - 1

    def get_state(self):
        """
        This function returns the sataes that node_to_split Node providing
        :return: self.states
        """
        return self.states

    def compute_value(self, data_point):
        """
        This function compute the data_point's corresponding value
        :param data_point:
        :return: generated value wrt. the data_point
        """

        '''Initialize the parent_value as zero since root_node no parent'''
        return self.root_node.get_value(data_point, 0)

    def output(self, data_point):
        """
        This function assume that data_point input is not transferred to binary form
        :param data_point:
        :return:
        """

        '''Call the binary transfer to get the binary transferred data_point'''
        data_point_transfer = self.binary_transfer.transfer(data_point)
        # print(data_point_transfer)
        '''Second output is the leaf node'''
        value, _ = self.compute_value(data_point_transfer)

        return value
    
    def output_all_queries_global(self, queries):
        dim = len(self.bit_length)
        # for i in range(dim):
        #     temp = []
        #     for j in range(self.bit_length[i]):
        #         temp.append(0)
        #     summarized_queries.append(temp)
        all_res = 0
        min_points = []
        max_points = []
        min_parents = []
        max_parents = []
        for query in queries:
            data_min = query.min_point
            data_max = query.max_point
            data_point_transfer_min = self.binary_transfer.transfer(data_min)
            data_point_transfer_max = self.binary_transfer.transfer(data_max)
            max_points.append(data_point_transfer_max)
            min_points.append(data_point_transfer_min)
            min_parents.append(0)
            max_parents.append(0)
            
        res_max, _ = self.root_node.get_value_queries(max_points, max_parents)
        res_min, _ = self.root_node.get_value_queries(min_points, min_parents)
        return sum(res_max) - sum(res_min)
    
    # need a tree traverse to read two tables
    # 1 generate two tables and fill out the table based on queries.
    # 
    def output_all_queries_local(self, queries):
        min_points = []
        max_points = []
        for query in queries:
            data_min = query.min_point
            data_max = query.max_point
            data_point_transfer_min = self.binary_transfer.transfer(data_min)
            data_point_transfer_max = self.binary_transfer.transfer(data_max)
            max_points.append(data_point_transfer_max)
            min_points.append(data_point_transfer_min)
            # pprint.pprint(data_point_transfer_min)
            # pprint.pprint(data_point_transfer_max)
        # recursively read the table

        volume, res, _ = self.root_node.cal_local_cost(min_points, max_points, [])
        # print("----volume:", volume, "res:", res, "final cost:", volume - res)
        return volume - res

    def add_reward(self, data_points, reward):
        """
        This function add reward to leaf nodes w.r.t. data_points
        :param data_points: [d1, d2, .., dn]
        :param reward: integer, reward computed by a specific query
        :return: None
        """

        '''Add the reward to every leaf node related to this dataset'''
        for data_point in data_points:
            data_point_transfer = self.binary_transfer.transfer(data_point)
            _, leaf_node = self.compute_value(data_point_transfer)
            leaf_node.reward += reward

        return

    def clear_reward(self):
        """
        This function Clear all reward top down stream
        :return:
        """

        '''clear reward based on node list'''
        for node in self.nodes:
            node.reward = 0

        return

    def recurse_reward(self):
        """
        This function generate all rewards for root node and inner node
        :return:
        """

        '''Node level compute the reward sum up from child-node'''
        self.root_node.compute_reward()

        '''Generate a list with rewards'''
        rewards = []
        for node in self.nodes:
            '''Here we compute the available to check if this node is split by agent'''
            mask, available = node.get_mask(self.bit_length)

            if (node.chosen_bit is not None) and (available != 1):
                rewards.append(node.reward)

        return rewards


    def save(self, path, max_depth):
        """
        This function save the BMTree
        :param path:
        :param max_depth:
        :return:
        """
        with open(path, 'w') as f:

            lines = []
            # lines.append('{} {} {}\n'.format(len(self.bit_length), self.bit_length[0],
            #                                self.bit_length[1]))
            first_line = ""
            first_line = first_line + "{} ".format(len(self.bit_length))
            for i in range(len(self.bit_length)):
                first_line = first_line + "{} ".format(self.bit_length[i])
            first_line = first_line + "\n"
            lines.append(first_line)

            lines.append('{}\n'.format(max_depth))

            for node in self.nodes:
                if node.chosen_bit is None:
                    chosen_bit = -1
                else:
                    chosen_bit = node.chosen_bit
                if node.left_children is None:
                    left_children = -1
                else:
                    left_children = node.left_children.node_id
                if node.right_children is None:
                    right_children = -1
                else:
                    right_children = node.right_children.node_id
                if node.child is None:
                    child = -1
                else:
                    child = node.child.node_id

                lines.append('{} {} {} {} {} {} {}\n'.format(node.node_id, node.parent_node_id,
                                               node.depth, chosen_bit, left_children, right_children, child))

            f.writelines(lines)
            f.flush()
        return
