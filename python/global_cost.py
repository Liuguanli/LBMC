import utils
import time
import copy
import sys
class GlobalCost:
    def __init__(self, windows, bits_nums):
        self.windows = windows
        self.bits_nums = copy.deepcopy(bits_nums)
        self.factor_map = {}
        self.dim = len(bits_nums)
        self.dim_counter = []
        self.each_curve_length_to_formula()

#     when given a distribution, use this one to calculate the cost
    def get_curve_value_via_location(self, location, bit_distribution, bits_nums):
        res = 0
        bit_current_location = len(bit_distribution) - 1
        masks = [pow(2, bits_num - 1) for bits_num in bits_nums]
        for char in bit_distribution:
            bit_index = utils.bit_letters.index(char)
            if masks[bit_index] & location[bit_index] != 0:
                res += pow(2, bit_current_location)
            masks[bit_index] = masks[bit_index] >> 1
            bit_current_location -= 1
        return res

    def cla_each_curve_length(self):
        res_temp_map = {}
        for window in self.windows:
            l_range = window.dimension_low
            h_range = window.dimension_high
            dim = window.dim

            for i in range(dim):
                low_temp = l_range[i]
                high_temp = h_range[i]

                for j in range(self.bits_nums[i]):
                    key = utils.factor_letters[i] + str(j + 1)
                    res_temp_map[key] = res_temp_map.get(key, 0)
                    if (low_temp & 1):
                        #                     low_temp_map[key] = low_temp_map.get(key, 0) + 1
                        res_temp_map[key] = res_temp_map.get(key, 0) - 1

                    if (high_temp & 1):
                        # key = factor_letters[i] + str(j + 1)
                        #                     high_temp_map[key] = high_temp_map.get(key, 0) + 1
                        res_temp_map[key] = res_temp_map.get(key, 0) + 1

                    low_temp = low_temp >> 1
                    high_temp = high_temp >> 1
    #             low.append(low_temp_map)
    #             high.append(high_temp_map)
        #         res.append(res_temp_map)
        #     print(low)
        #     print(high)
        #     print(res)
        return res_temp_map

#     @timer
    def get_factor_value_via_bit_distribution(self, bit_distribution):
        #         start_time = perf_counter()
        start_time = time.time_ns()
        bits_nums = copy.deepcopy(self.bits_nums)
        index = len(bit_distribution) - 1
        res = {}
        for char in bit_distribution:
            for i in range(self.dim):
                if char == utils.bit_letters[i]:
                    res[utils.factor_letters[i]+str(bits_nums[i])] = index
                    bits_nums[i] -= 1
                    break
            index -= 1
        self.factor_value = res
#         return res
        if len(self.factor_map) == 0:
            self.each_curve_length_to_formula()
#         end_time = perf_counter()
        end_time = time.time_ns()
        return end_time - start_time

    def each_curve_length_to_formula(self):
        # if len(self.factor_map) == 0:
        self.factor_map = self.cla_each_curve_length()
        # print("self.bits_nums", self.bits_nums)
        if len(self.dim_counter) == 0:
            for index, bit_num in enumerate(self.bits_nums):
                counter = []
                for i in range(bit_num):
                    counter.append(
                        self.factor_map[utils.factor_letters[index] + str(i + 1)])
                self.dim_counter.append(counter)

        formula = ""
        dim = len(self.bits_nums)
        for i in range(dim):
            for j in range(self.bits_nums[i]):
                key = utils.factor_letters[i] + str(j + 1)
                factor = self.factor_map.get(key, 0)
                if factor != 0:
                    if i == 0 and j == 0:
                        if factor > 0:
                            formula += str(factor) + "*" + "2^" + key
                        else:
                            formula += str(factor) + "*" + "2^" + key
                    else:
                        if factor > 0:
                            formula += " + " + str(factor) + "*" + "2^" + key
                        else:
                            formula += " - " + str(-factor) + "*" + "2^" + key
    #     print(formula)
        return formula

#     @timer
    def global_cost(self, bit_distribution):
        #         factor_value =
        #         if len(self.factor_map) == 0:
        #             self.factor_map = self.cla_each_curve_length()
        #         start_time = perf_counter()
        start_time = time.time_ns()
        self.get_factor_value_via_bit_distribution(bit_distribution)
        res = 0
        for key in self.factor_map:
            if key in self.factor_value.keys():
                res += self.factor_map[key] * pow(2, self.factor_value[key])
#         end_time = perf_counter()
        end_time = time.time_ns()
        return res , end_time - start_time
#         return res

#     @timer
    def naive_global_cost(self, bit_distribution):
        #         start_time = perf_counter()
        start_time = time.time_ns()
        res = 0
        for window in self.windows:
            res += self.get_curve_value_via_location(
                window.dimension_high, bit_distribution, self.bits_nums)
            res -= self.get_curve_value_via_location(
                window.dimension_low, bit_distribution, self.bits_nums)
#         end_time = perf_counter()
        end_time = time.time_ns()
        return res, end_time - start_time

#     def get_global_optimal_curve(self):
#         dim_counter_accumulate_summary = []
#         for i in range(self.dim):
#             current = 0
#             for j in range(self.bits_nums[i]):
#                 current += self.dim_counter[i][self.bits_nums[i] - j - 1] / pow(2, j)
#             dim_counter_accumulate_summary.append(current)
#         cursors = [bit_num - 1 for bit_num in self.bits_nums]
#         length = sum(self.bits_nums)
#         res = ""
# #         print(self.dim_counter)
#         # print(dim_counter_accumulate_summary)
#         for i in range(length):
# #             print("----------")
#             min_index = 0
#             min_val = sys.maxsize
#             for index, cursor in enumerate(cursors):
#                 # print(self.dim_counter[index][cursor], dim_counter_accumulate_summary[index])
#                 if cursor >= 0:
#                     if self.dim_counter[index][cursor] < min_val:
#                         min_val = self.dim_counter[index][cursor]
#                         min_index = index
#                     elif self.dim_counter[index][cursor] == min_val:
#                         if dim_counter_accumulate_summary[index] < dim_counter_accumulate_summary[min_val]:
#                             min_val = self.dim_counter[index][cursor]
#                             min_index = index
#             cursors[min_index] -= 1
#             res += utils.bit_letters[min_index]
            
#             dim_counter_accumulate_summary[min_index] -= self.dim_counter[min_index][cursor] 
#             dim_counter_accumulate_summary[min_index] *= 2
#         return res   

    def get_global_optimal_curve(self):
        cursors = [bit_num - 1 for bit_num in self.bits_nums]
        length = sum(self.bits_nums)
        res = ""
#         print(self.dim_counter)
        for i in range(length):
#             print("----------")
            min_index = 0
            min_val = sys.maxsize
            for index, cursor in enumerate(cursors):
                if cursor >= 0:
                    if self.dim_counter[index][cursor] < min_val:
                        min_val = self.dim_counter[index][cursor]
                        min_index = index
#                         print("if :", min_val, min_index)
                    elif self.dim_counter[index][cursor] == min_val:
                       #  tie : self.dim_counter[min_index][cursors[min_index]] == self.dim_counter[index][cursor] 
                        pre_min_cursot = cursors[min_index]
                        while pre_min_cursot >= 0 and cursor >= 0:
                            if pre_min_cursot == 0 and cursor == 0:
                                break
                            if self.dim_counter[index][cursor] < self.dim_counter[min_index][pre_min_cursot]:
                                min_val = self.dim_counter[index][cursor]
                                min_index = index
                                break
                            elif self.dim_counter[index][cursor] > self.dim_counter[min_index][pre_min_cursot]:
                                break
                            else:
                                if pre_min_cursot > 0:
                                    pre_min_cursot -= 1
                                if cursor > 0:
                                    cursor -= 1
#                         print("elif :", min_val, min_index)
#             print(bit_letters[min_index])
                     
            cursors[min_index] -= 1
            res += utils.bit_letters[min_index]
        return res

#         -16*2^a1 - 25*2^a2 + 58*2^a3 + 39*2^a4 - 71*2^a5 - 11*2^a6 - 61*2^a7 + 6*2^a8 - 44*2^a9 - 23*2^a10 - 16*2^a11 + 24*2^a12 + 21*2^a13 + 38*2^a14 + 49*2^a15 + 240*2^a16 - 65*2^b1 - 6*2^b2 + 4*2^b3 - 20*2^b4 + 20*2^b5 + 2*2^b6 + 11*2^b7 - 35*2^b8 - 36*2^b9 - 16*2^b10 - 12*2^b11 + 35*2^b12 + 43*2^b13 + 25*2^b14 + 54*2^b15 + 237*2^b16
# ['AAAAAAAAAAAAAAAABBBBBBBBBBBBBBBB'] minimal global cost 594422556131
