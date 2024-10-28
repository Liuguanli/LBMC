import utils
import time
import copy
import sys

import math

class WindowInfo:
    def __init__(self, window, config):
        self.patterns_num = {}
        self.gaps_num = {}
        for i in range(config.dim):
            dim_low = window.dimension_low[i]
            dim_high = window.dimension_high[i]
            dim_low_copy = dim_low
            dim_high_copy = dim_high
            total_pattern_num = 0
            for bit_num in range(config.dim_length[i] + 1):
                if bit_num == 0:
                    # TODO dim_high - dim_low + 1 ro dim_high - dim_low
                    self.patterns_num[utils.bit_letters[i] +
                                      str(bit_num)] = dim_high - dim_low + 1
                    self.gaps_num[utils.bit_letters[i] +
                                  str(bit_num)] = dim_high - dim_low + 1

                else:
                    gap = int(dim_high - dim_low) - 1
                    pattern_range = int(pow(2, bit_num))
                    pattern_num = int(gap / pattern_range)
                    temp_dim_low = dim_low + pattern_num * pattern_range
                    if int((temp_dim_low + pattern_range / 2) / pattern_range) != int((dim_high + pattern_range / 2) / pattern_range):
                        pattern_num += 1
                    self.patterns_num[utils.bit_letters[i] +
                                      str(bit_num)] = pattern_num
#                     gap = int(dim_high - dim_low) + 1
                    gap_num = 0
                    gap = dim_high_copy - dim_low_copy + 1
                    if bit_num >= 1 and pattern_range <= gap:
                        gap_num = int((dim_high_copy+1)/pattern_range) - \
                            math.ceil(dim_low_copy/pattern_range)
                    # gap_num
                    self.gaps_num[utils.bit_letters[i]+str(bit_num)] = gap_num

                    total_pattern_num += pattern_num
#             print(self.patterns_num, dim_high, dim_low)
            # assert total_pattern_num == dim_high - dim_low, "wrong pattern_num calculation" + \
            #     str(total_pattern_num) + " " + str(dim_high - dim_low)
#         print(self.patterns_num)


# In[6]:


class TableCellEachWindow:
    def __init__(self, column_num, row_num, config, dim_index):
        self.column_num = column_num
        self.row_num = row_num
        self.dim_index = dim_index
        self.config = config
        self.key_num = {}
        self.combinations = []

    def gen_all_keys(self, window_info):
        key_letters = [utils.bit_letters[i]
                       for i in range(self.config.dim) if i is not self.dim_index]

        candidate_dims = [i for i in range(
            self.config.dim) if i is not self.dim_index]
        self.combination(0, candidate_dims, [], 0)
#         print(self.combinations)
#         print(self.row_num)
        if self.row_num > 1:
            for comb in self.combinations:
                key = ""
                num = 1
                for index, dim_index in enumerate(candidate_dims):
                    temp_key = utils.bit_letters[dim_index] + str(comb[index])
                    key += temp_key
#                     if comb[index] == 0:
#                         continue
                    num *= window_info.gaps_num[temp_key]
                    # if comb[index] = 0 , this is the length of that dimension,
                    # if comb[index] != 0, this is the number of x[i]
                self.key_num[key] = num * \
                    window_info.patterns_num[utils.bit_letters[self.dim_index] +
                                             str(self.column_num)]
        else:
            for comb in self.combinations:
                key = ""
                num = 1
                for index, dim_index in enumerate(candidate_dims):
                    temp_key = utils.bit_letters[dim_index] + str(comb[index])
                    key += temp_key
                    num *= window_info.patterns_num[temp_key]
                    # if comb[index] = 0 , this is the length of that dimension,
                    # if comb[index] != 0, this is the number of x[i]
                self.key_num[key] = num * \
                    window_info.patterns_num[utils.bit_letters[self.dim_index] +
                                             str(self.column_num)]
#                 print("表中的key:", key, "边的长度", bit_letters[self.dim_index] + str(self.column_num), window_info.patterns_num[bit_letters[self.dim_index] + str(self.column_num)])
    #             self.key_num[key] = num
    #         print("TableCellEachWindow", self.key_num)

    def combination(self, current_index, candidate_dims, temp, current_length):
        if current_index == len(candidate_dims):
            if self.column_num[current_index] == current_length:
                i = 0
                for temp_dim_len in temp:
                    if i == self.dim_index:
                        i += 1
                    if self.config.dim_length[i] < temp_dim_len:
                        #                         print(self.config.dim_length, temp)
                        return
                    i += 1

                self.combinations.append(temp)
#                 print(temp)

        else:
            start = min(self.row_num - current_length,
                        self.config.dim_length[current_index])
#             print("start:", start, "self.row_num", self.row_num, "current_length", current_length,
#                  "self.config.dim_length[current_index]", self.config.dim_length[current_index],
#                  "self.dim_index", self.dim_index, "current_index", current_index)
# start: 7 self.row_num 9 current_length 0 self.config.dim_length[current_index] 7 self.dim_index 0 current_index 0
# start: 9 self.row_num 9 current_length 0 self.config.dim_length[current_index] 9 self.dim_index 0 current_index 1
# start: 8 self.row_num 9 current_length 1 self.config.dim_length[current_index] 9 self.dim_index 0 current_index 1
# start: 7 self.row_num 9 current_length 2 self.config.dim_length[current_index] 9 self.dim_index 0 current_index 1
# start: 6 self.row_num 9 current_length 3 self.config.dim_length[current_index] 9 self.dim_index 0 current_index 1
# start: 5 self.row_num 9 current_length 4 self.config.dim_length[current_index] 9 self.dim_index 0 current_index 1
# start: 4 self.row_num 9 current_length 5 self.config.dim_length[current_index] 9 self.dim_index 0 current_index 1
# start: 3 self.row_num 9 current_length 6 self.config.dim_length[current_index] 9 self.dim_index 0 current_index 1
# start: 2 self.row_num 9 current_length 7 self.config.dim_length[current_index] 9 self.dim_index 0 current_index 1

            start = self.row_num - current_length
            for i in range(start + 1):
                if (self.config.dim_length[current_index] < i):
                    continue
                temp_copy = copy.deepcopy(temp)
                temp_copy.append(i)
#                 if current_length + i <= self.config.dim_length[current_index + 1]:
                self.combination(current_index + 1, candidate_dims,
                                 temp_copy, current_length + i)
# test      TableCell
# config = Config([4, 4, 4, 4])
# tc = TableCellEachWindow(2, 5, config, 1)
# tc.gen_all_keys()


# In[7]:


class TableCell:
    def __init__(self, windows, column_num, row_num, config, dim_index=0):
        # each window get the window info
        self.key_num = {}
        for window in windows:
            window_info = WindowInfo(window, config)
#             print(window_info.gaps_num)
            tc_each_window = TableCellEachWindow(
                column_num, row_num, config, dim_index)
            tc_each_window.gen_all_keys(window_info)
            for key in tc_each_window.key_num.keys():
                #                 print(key, tc_each_window.key_num[key])
                self.key_num[key] = self.key_num.get(
                    key, 0) + tc_each_window.key_num[key]
        # use each window infor to get tablecellfor each window
        # accumulate all.
#         print(self.key_num)




class LocalCost:
    #     @timer
    def __init__(self, windows, bits_nums, config):
        self.windows = windows
        self.bits_nums = bits_nums
        self.factor_map = {}
        self.dim = len(bits_nums)
        self.area = 0
        self.config = config

    def prepare_tables(self):
        #         start_time = perf_counter()
        start_time = time.time_ns()
        for window in self.windows:
            self.area += window.get_area()
        total_length = 0
        for length in self.bits_nums:
            total_length += length
        self.table = []
        self.total_combination_num = 0
        for i in range(self.dim):
            dim_table = {}
            other_dims_length = total_length - self.bits_nums[i]
            for j in range(other_dims_length + 1):  # row
                for k in range(self.bits_nums[i]):  # column
                    #             print(k + 1, j)
                    # column_num, row_num
                    tc = TableCell(self.windows, k + 1, j, self.config, i)
                    dim_table[str(k + 1) + "_" + str(j)] = tc
                    self.total_combination_num += len(tc.key_num)
            self.table.append(dim_table)
#         print("total_combination_num", self.total_combination_num)
#         end_time = perf_counter()
        end_time = time.time_ns()
        return end_time - start_time

#     @timer
    def local_cost(self, bit_distribution):
        #         start_time = perf_counter()
        start_time = time.time_ns()
        self.get_table_reading_map(bit_distribution)
        res = 0
        cross = 0
        for i in range(self.dim):
            for j in range(self.bits_nums[i]):
                column_num = j + 1
                if utils.bit_letters[i] + str(column_num) not in self.length_list[i].keys():
                    continue
                row_num = self.length_list[i][utils.bit_letters[i] + str(column_num)]
                key = str(column_num) + "_" + str(row_num)
                tc = self.table[i].get(key, None)
                if tc is None:
                    continue
                table_cell_key = self.map_list[i][utils.bit_letters[i] +
                                                  str(column_num)]
                if table_cell_key not in tc.key_num.keys():
                    continue
                res += tc.key_num[table_cell_key]
                if column_num > 1 and row_num > 1:
                    cross += tc.key_num[table_cell_key]
#         end_time = perf_counter()
        end_time = time.time_ns()
        return self.area - res, end_time - start_time

    def get_local_optimal_curve(self):
        res = ""
        bits_nums_copy = copy.deepcopy(self.bits_nums)
        all_bits_num = sum(bits_nums_copy)
        while all_bits_num > 0:
            max_index = -1
            max_res = -1
            for i in range(self.dim):
                if bits_nums_copy[i] == 0:
                    continue
                table_cell_key = ""
                for j in range(self.dim):
                    if i == j:
                        continue
                    table_cell_key += utils.bit_letters[j] + str(bits_nums_copy[j])
                key = str(bits_nums_copy[i]) + "_" + \
                    str(all_bits_num - bits_nums_copy[i])
#                 print(key)
                tc = self.table[i].get(key, None)
#                 print(tc.key_num)
                if  table_cell_key not in tc.key_num.keys():
                    max_index = i
                    max_res = 0
                else:
                    #                     print("here", tc.key_num[table_cell_key])
                    if max_res < tc.key_num[table_cell_key]:
                        max_res = tc.key_num[table_cell_key]
                        max_index = i
            if max_res == 0:
                all_bits_num -= self.dim
                for j in range(self.dim):
                    bits_nums_copy[j] -= 1
                    res += utils.bit_letters[self.dim - 1 - j]
            else:
                all_bits_num -= 1
                bits_nums_copy[max_index] -= 1
                res += utils.bit_letters[max_index]
        return res

    def print_table(self):
        for i in range(self.dim):
            for key in self.table[i].keys():
                print(key, self.table[i][key].key_num)
#             print(self.table[i].keys())

    def get_table_reading_map(self, bit_distribution):
        counter = [0 for _ in range(self.dim)]
        map_list = [{} for _ in range(self.dim)]
        length_list = [{} for _ in range(self.dim)]

        lbd = len(bit_distribution)
        for i in range(lbd - 1, -1, -1):
            bit_index = utils.bit_letters.index(bit_distribution[i])
            counter[bit_index] += 1
            key = bit_distribution[i] + str(counter[bit_index])
            val_str = ""
            bit_dis_len = 0
            for j in range(self.dim):
                if j == bit_index:
                    continue
    #             if counter[j] != 0:
                val_str += utils.bit_letters[j] + str(counter[j])
                bit_dis_len += counter[j]
            map_list[bit_index][key] = val_str
            length_list[bit_index][key] = bit_dis_len
        self.map_list = map_list
        self.length_list = length_list

#     def get_map_from_bit_distribution(self):
    def get_map_bit_distribution(self, bit_distribution):
        bits_nums = copy.deepcopy(self.bits_nums)
        self.bits_nums_map = []
        self.bit_index_map = []
        self.masks_map = []
        masks = [pow(2, bits_num - 1) for bits_num in bits_nums]
        for char in bit_distribution:
            bit_index = utils.bit_letters.index(char)
            bits_nums[bit_index] -= 1
            self.bits_nums_map.append(bits_nums[bit_index])
            self.bit_index_map.append(bit_index)
            self.masks_map.append(masks[bit_index])
            masks[bit_index] = masks[bit_index] >> 1

#             if mask & value != 0:
#                 vals[bit_index] += pow(2, bits_nums[bit_index])
#             mask = mask >> 1
#         return vals

    def get_location_via_curve_value(self, value, bit_distribution):
        vals = [0 for i in range(len(self.bits_nums))]
        mask = pow(2, len(bit_distribution) - 1)
        for i in range(len(bit_distribution)):
            bit_index = self.bit_index_map[i]
            if mask & value != 0:
                vals[bit_index] += pow(2, self.bits_nums_map[i])
            mask = mask >> 1
        return vals
#         bits_nums = copy.deepcopy(self.bits_nums)
#         bit_map = []
#         mask = pow(2, len(bit_distribution) - 1)
#         vals = [0 for i in range(len(bits_nums))]
#         for char in bit_distribution:
#             bit_index = bit_letters.index(char)
#             bits_nums[bit_index] -= 1
#             if mask & value != 0:
#                 vals[bit_index] += pow(2, bits_nums[bit_index])
#             mask = mask >> 1
#         return vals

#     def get_curve_value_via_location(location, bit_distribution, bits_nums):
#         res = 0
#         bit_current_location = len(bit_distribution) - 1
#         masks = [pow(2, bits_num - 1) for bits_num in bits_nums]
#         for char in bit_distribution:
#             bit_index = bit_letters.index(char)
#             if masks[bit_index] & location[bit_index] !=0:
#                 res += pow(2, bit_current_location)
#             masks[bit_index] = masks[bit_index] >> 1
#             bit_current_location -= 1
#         return res

    def get_curve_value_via_location(self, location, bit_distribution, bits_nums):
        res = 0
        bit_current_location = len(bit_distribution) - 1
        for i in range(len(bit_distribution)):
            bit_index = self.bit_index_map[i]
            if self.masks_map[i] & location[bit_index] != 0:
                res += pow(2, bit_current_location)
            bit_current_location -= 1
        return res

#     @timer
    def naive_local_cost(self, bit_distribution):
        #         start_time = perf_counter()
        start_time = time.time_ns()
        res = 0
        dim = len(self.bits_nums)
        self.get_map_bit_distribution(bit_distribution)
        for window in self.windows:
            low = self.get_curve_value_via_location(
                window.dimension_low, bit_distribution, self.bits_nums)
            high = self.get_curve_value_via_location(
                window.dimension_high, bit_distribution, self.bits_nums)
            is_in = True
            num = 1
#             print(low, high)
            for val in range(low, high + 1, 1):
                location = self.get_location_via_curve_value(
                    val, bit_distribution)
                flag = True
                for i in range(dim):
                    if location[i] < window.dimension_low[i] or location[i] > window.dimension_high[i]:
                        flag = False
                        break
                if flag:
                    if is_in:
                        continue
                    else:
                        is_in = True
                        num += 1
                else:
                    is_in = False
            res += num
#         end_time = perf_counter()
        end_time = time.time_ns()
        return res, end_time - start_time
