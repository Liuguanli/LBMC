import time
import copy
import sys
from time import perf_counter
import math

from utils import bit_letters

class LocalCost:
    def __init__(self, windows, bits_nums):
        self.windows = windows
        self.bits_nums = bits_nums
        self.dim = len(bits_nums)
        self.table = []
        self.row_lens = self.bits_nums
        self.column_lens = [sum(self.bits_nums) - self.bits_nums[i] + 1 for i in range(self.dim)]
        self.combinations = []
        self.volume = 0
        
        for window in self.windows: 
            window.gen_drop_patterns(self.bits_nums)
            window.gen_rise_patterns(self.bits_nums)
            self.volume += window.area
        if self.dim < 4:
            self.gen_pattern_tables()
            self.fill_out_tables()


    def gen_pattern_tables(self):

        if self.dim == 2:
            self.table.append([[0 for i in range(self.column_lens[0])] for i in range(self.row_lens[0])])
            self.table.append([[0 for i in range(self.column_lens[1])] for i in range(self.row_lens[1])])
            return

        for k in range(self.dim):
            self.table.append([[{} for i in range(self.column_lens[k])] for i in range(self.row_lens[k])])
        pass

    def fill_out_tables(self):
        start_time = perf_counter()
        
        if self.dim == 2:
            for i in range(self.dim):
                rise_dim = i
                drop_dim = self.dim - 1 - i
                for rise_index in range(self.row_lens[i]):
                    # cal rise pattern
                    for drop_index in range(self.column_lens[i]):
                        # cal drop pattern
                        for window in self.windows: 
                            rise_res = window.rise_patterns[rise_dim][rise_index]
                            drop_res = window.drop_patterns[drop_dim][drop_index]
                            self.table[rise_dim][rise_index][drop_index] += rise_res * drop_res
            return

        for i in range(self.dim):
            rise_dim = i
            candidate_dims = [i for i in range(self.dim) if i != rise_dim]
            for drop_index in range(self.column_lens[i]):
                self.combinations = []
                self.combination(rise_dim, candidate_dims, [], 0, drop_index)
                for rise_index in range(self.row_lens[i]):
                    for window in self.windows: 
                        rise_res = window.rise_patterns[rise_dim][rise_index]
                        for combs in self.combinations:
                            key = ""
                            drop_cost = 1
                            for i, dim_drop_index in enumerate(combs):
                                drop_dim = candidate_dims[i]
                                key += bit_letters[drop_dim] + str(dim_drop_index)
                                drop_cost *= window.drop_patterns[drop_dim][dim_drop_index]
                            if key not in self.table[rise_dim][rise_index][drop_index].keys():
                                self.table[rise_dim][rise_index][drop_index][key] = 0
                            self.table[rise_dim][rise_index][drop_index][key] += drop_cost * rise_res
        end_time = perf_counter()
        # print('{0} costs {1:.8f}s'.format("fill_out_tables", end_time - start_time))

    def combination(self, rise_dim, candidate_dims, temp, changed_len, drop_changed_length):

        # if len(candidate_dims) == len(temp):
        #     self.combinations.append(temp)
        if len(candidate_dims) - len(temp) == 1:
            remainder = drop_changed_length - changed_len
            if remainder >= 0 and remainder <= self.bits_nums[candidate_dims[-1]]:
                temp_copy = copy.deepcopy(temp)
                temp_copy.append(remainder)
                self.combinations.append(temp_copy)
        else:
            index = candidate_dims[len(temp)]

            maximal_remainder_changed_len = 0
            for dim in candidate_dims[len(temp) + 1:]:
                maximal_remainder_changed_len += self.bits_nums[dim]

            for dim_changed_len in range(self.bits_nums[index] + 1):
                current_changed_len = changed_len + dim_changed_len
                if current_changed_len > drop_changed_length:
                    break # already extend the maximal number of changed bits
                if current_changed_len + maximal_remainder_changed_len < drop_changed_length:
                    continue # not enough enough the other remaind dims are all changed

                temp_copy = copy.deepcopy(temp)
                temp_copy.append(dim_changed_len)
#                 if current_length + i <= self.config.dim_length[current_index + 1]:
                self.combination(rise_dim, candidate_dims, temp_copy, current_changed_len, drop_changed_length)
            
    def get_index_map_for_md(self, BMC: str):
        reading_map = [[] for i in range(self.dim)]
        dim_counter = [0 for i in range(self.dim)]

        for i in range(len(BMC) - 1, -1, -1):
            rise_index = bit_letters.index(BMC[i])
            dim_counter[rise_index] += 1
            drop_key = []
            changed_bits = 0
            for i in range(self.dim):
                if i == rise_index:
                    continue
                drop_key.append(dim_counter[i])
                
                changed_bits += dim_counter[i]
            reading_map[rise_index].append(drop_key)
        return reading_map
    
    def get_reading_map(self, BMC: str):
        reading_map = [[] for i in range(self.dim)]
        reading_map_md = [[] for i in range(self.dim)]
        dim_counter = [0 for i in range(self.dim)]
        if self.dim == 2:
            for i in range(len(BMC) - 1, -1, -1):
                rise_index = bit_letters.index(BMC[i])
                dim_counter[rise_index] += 1
                drop_key = 0
                for j in range(self.dim):
                    if j == rise_index:
                        continue
                    drop_key = dim_counter[j]
                reading_map[rise_index].append(drop_key)
            return reading_map, reading_map

        for i in range(len(BMC) - 1, -1, -1):
            rise_index = bit_letters.index(BMC[i])
            dim_counter[rise_index] += 1
            drop_key = ""
            changed_bits = 0
            for i in range(self.dim):
                if i == rise_index:
                    continue
                drop_key += bit_letters[i] + str(dim_counter[i])
                changed_bits += dim_counter[i]
            reading_map[rise_index].append(drop_key)
            reading_map_md[rise_index].append(changed_bits)
        return reading_map, reading_map_md
    
    def local_cost(self, BMC: str):
        start_time = time.time_ns()
        reading_map, changed_bits = self.get_reading_map(BMC)
        # print(reading_map)
        # print(changed_bits)
        # print(self.table[0][1])
        cost = 0


        if self.dim == 2:
            for i in range(self.dim):
                for rise_index in range(self.row_lens[i]):
                    cost += self.table[i][rise_index][changed_bits[i][rise_index]]
            end_time = time.time_ns()
            return self.volume - cost, end_time - start_time
        
        if self.dim > 3:
            reading_map = self.get_index_map_for_md(BMC)
            cost = 0
            for window in self.windows: 
                for i in range(self.dim):
                    rise_dim = i
                    for rise_index in range(self.bits_nums[i]):
                        rise_res = window.rise_patterns[rise_dim][rise_index]
                        drop_res = 1
                        drop_index = 0
                        for j in range(self.dim):
                            if j == rise_dim:
                                continue
                            drop_res *= window.drop_patterns[j][reading_map[rise_index][drop_index]]
                            drop_index += 1
                        cost += rise_res * drop_res
            end_time = time.time_ns()
            return self.volume - cost, end_time - start_time

        for i in range(self.dim):
            for rise_index in range(self.row_lens[i]):
                cost += self.table[i][rise_index][changed_bits[i][rise_index]][reading_map[i][rise_index]]
        end_time = time.time_ns()
        return self.volume - cost, end_time - start_time
    
    def get_curve_value_via_location(self, location, bit_distribution, bits_nums):
        res = 0
        bit_current_location = len(bit_distribution) - 1
        masks = [pow(2, bits_num - 1) for bits_num in bits_nums]
        for char in bit_distribution:
            bit_index = bit_letters.index(char)
            if masks[bit_index] & location[bit_index] !=0:
                res += pow(2, bit_current_location)
            masks[bit_index] = masks[bit_index] >> 1
            bit_current_location -= 1
        return res
    
            
    def get_location_via_curve_value(self, value, bit_distribution):
        bits_nums = copy.deepcopy(self.bits_nums)
        bit_map = []
        mask = pow(2, len(bit_distribution) - 1)
        vals = [0 for i in range(len(bits_nums))]
        for char in bit_distribution:
            bit_index = bit_letters.index(char)
            bits_nums[bit_index] -= 1
            if mask & value != 0:
                vals[bit_index] += pow(2, bits_nums[bit_index])
            mask = mask >> 1
        return vals

    def naive_local_cost(self, bit_distribution):
        start_time = time.time_ns()
        res = 0
        dim = len(self.bits_nums)
        for window in self.windows:
            low = self.get_curve_value_via_location(window.dimension_low, bit_distribution, self.bits_nums)
            high = self.get_curve_value_via_location(window.dimension_high, bit_distribution, self.bits_nums)
            is_in = True
            num = 1
#             print(low, high)
            for val in range(low, high + 1, 1):
                location = self.get_location_via_curve_value(val, bit_distribution)
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
        end_time = time.time_ns()
        return res , end_time - start_time

    def quilts_cost(self, bit_distribution):
        start_time_all = time.time_ns()
        dim = len(self.bits_nums)
        res = 0
        for window in self.windows:
            start_time = perf_counter()

            interval_lengths = []
            low = self.get_curve_value_via_location(window.dimension_low, bit_distribution, self.bits_nums)
            high = self.get_curve_value_via_location(window.dimension_high, bit_distribution, self.bits_nums)
            is_in = True
            interval_start = low
            for val in range(low, high + 1, 1):
                location = self.get_location_via_curve_value(val, bit_distribution)
                flag = True
                for i in range(dim):
                    if location[i] < window.dimension_low[i] or location[i] > window.dimension_high[i]:
                        flag = False
                        break
                if flag:
                    if is_in:
                        continue
                    else:
                        interval_lengths.append(val - interval_start)
                        is_in = True
                else:
                    if is_in == True:
                        interval_start = val
                    is_in = False
            c_g = sum(interval_lengths)
            c_t = c_g * math.log(c_g) - sum([i_l * math.log(i_l) for i_l in interval_lengths])
            res += c_g * c_t
            end_time = perf_counter()
            # print('{0} costs {1:.6f}s'.format("each window costs time:", end_time - start_time))
        end_time_all = time.time_ns()
        return res, end_time_all - start_time_all