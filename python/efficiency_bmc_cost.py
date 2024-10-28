import json
import random
import time

from global_cost import GlobalCost
from local_cost import LocalCost
from generators import random_sampling
from generators import BMC_generation
from generators import get_query_windows

from utils import Window
# import generator

random.seed(2023)

# bit_letters = ["A", "B", "C", "D", "E"]

# def Z_Curve_generation(bit=12, dim=2):
#     length = dim * bit
#     clockwise_ZC = ""

#     for i in range(length):
#         dim_index = i % dim
#         clockwise_ZC += bit_letters[dim_index]
        
#     anti_clockwise_ZC = ""
#     for i in range(dim - 1, length + dim - 1, 1):
#         dim_index = i % dim
#         anti_clockwise_ZC += bit_letters[dim_index]
    
#     return [clockwise_ZC, anti_clockwise_ZC]

# def BMC_generation(bit=12, dim=2, num=8):
#     '''
#     generage $num BMCs, each BMC has $dim dimensions, each dimension has $bits bits
#     '''
#     res = []
#     length = dim * bit
#     for j in range(num):
#         temp_SFC = ""
#         remained_dim_bits = [bit for i in range(dim)]

#         for i in range(length):
#             while True:
#                 dim_index = random.randint(0, dim - 1)
#                 if remained_dim_bits[dim_index] == 0:
#                     continue
#                 temp_SFC += bit_letters[dim_index]
#                 remained_dim_bits[dim_index] -= 1
#                 break
#         res.append(temp_SFC)
        
#     ZCs = Z_Curve_generation(bit, dim)
#     res.extend(ZCs)
#     return res


# def generate_a_window(bit, dim, ratio):
#     dim_len = 2 ** bit
#     lengths = [dim_len for i in range(dim)]

#     dimension_low = []
#     dimension_high = []
#     dimension_low_raw = []
#     dimension_high_raw = []
# #     random.seed(10)
#     for i in range(dim):
#         # set the random range [0, 1-dim_i_length]
#         start_dim_i = random.random() * (1 - ratio[i])
#         end_dim_i = start_dim_i + ratio[i]
#         dimension_low.append(start_dim_i * lengths[i])
#         dimension_high.append(end_dim_i * lengths[i])
#         dimension_low_raw.append(start_dim_i)
#         dimension_high_raw.append(end_dim_i)

#     window = Window(dimension_low, dimension_high,
#                     dimension_low_raw, dimension_high_raw)
#     return window


# def get_query_windows(bit=16, dim=2, num=1000, ratio=[0.01, 0.01]):
#     windows = []
#     for i in range(num):
#         windows.append(generate_a_window(
#             bit, dim, ratio))
#     return windows


# def random_sampling(ratio=0.1, windows=None):
#     if ratio == 1:
#         return windows
#     return random.sample(windows, int(len(windows) * ratio))


def measurement_pattern(BMCs, bit=12, dim=2, sampling_ratios=[0.01, 0.02, 0.04, 0.08, 0.1, 1.0], type="global", windows=None):
    """_summary_

    Args:
        BMCs (_type_): _description_
        windows (_type_): _description_
        bits_nums (_type_): _description_
        type (str, optional): _description_. Defaults to "global".
        sampling_ratios (list, optional): _description_. Defaults to [0.01, 0.02, 0.04, 0.08, 0.1, 1.0].

    Returns:
        _type_: _description_
    """
    bits_nums = [bit for i in range(dim)]
    res = {}
    for index, BMC in enumerate(BMCs):
        # print(BMC)
        '''
        For each BMC, we need to calculate the global cost and local cost based on the sampled windows and all windows.
        '''
        each_BMC_res = {}
        for ratio in sampling_ratios:
            sampled_windows = random_sampling(ratio=ratio, windows=windows)
            each_ratio_res = {}
            if type == "global":
                # print("global", dim, len(windows), len(sampled_windows), bits_nums)
                GC = GlobalCost(sampled_windows, bits_nums)
                each_ratio_res["naive"] = GC.naive_global_cost(BMC)[1]
                each_ratio_res["GC"] = GC.global_cost(BMC)[1]
            elif type == "local":
                # print("local BMC index:", index, dim, len(windows), len(sampled_windows), bits_nums)
                LC = LocalCost(sampled_windows, bits_nums)
                each_ratio_res["naive"] = LC.naive_local_cost(BMC)[1]
                each_ratio_res["LC"] = LC.local_cost(BMC)[1]
            each_BMC_res[ratio] = each_ratio_res
        res[BMC] = each_BMC_res
    return res
                

def varying_query_nums(BMCs, bit, dim, nums, ratio):
    # print('varying_query_nums', bit, dim, nums, ratio)
    global_varying_query_nums = {}
    local_varying_query_nums = {}
    for num in nums:
        windows = get_query_windows(bit, dim, num, ratio)
        global_varying_query_nums[num] = measurement_pattern(BMCs, bit, dim, sampling_ratios=[0.01, 0.02, 0.04, 0.08, 0.1, 1.0], type="global", windows=windows)
        local_varying_query_nums[num] = measurement_pattern(BMCs, bit, dim, sampling_ratios=[0.01, 0.02, 0.04, 0.08, 0.1, 1.0], type="local", windows=windows)
    json.dump(global_varying_query_nums, open("../result/efficiency/global_varying_query_nums.json", "w"))
    json.dump(local_varying_query_nums, open("../result/efficiency/local_varying_query_nums.json", "w"))


def varying_dims(bit, dims, num, ratios):
    # print('varying_dims', bit, dims, num, ratios)
    global_varying_dims = {}
    local_varying_dims = {}
    for dim, ratio in zip(dims, ratios):
        # print(bit, dim, ratio)
        BMCs = BMC_generation(bit, dim, num=6)
        windows = get_query_windows(bit, dim, num, ratio)
        global_varying_dims[dim] = measurement_pattern(BMCs, bit, dim, sampling_ratios=[0.01, 0.02, 0.04, 0.08, 0.1, 1.0], type="global", windows=windows)
        local_varying_dims[dim] = measurement_pattern(BMCs, bit, dim, sampling_ratios=[0.01, 0.02, 0.04, 0.08, 0.1, 1.0], type="local", windows=windows)
    json.dump(global_varying_dims, open("../result/efficiency/global_varying_dims.json", "w"))
    json.dump(local_varying_dims, open("../result/efficiency/local_varying_dims.json", "w"))

def varying_query_ratio(BMCs, bit, dim, num, ratios):
    # print('varying_query_ratio', bit, dim, num, ratios)
    global_varying_query_ratio = {}
    local_varying_query_ratio = {}
    for ratio in ratios:
        windows = get_query_windows(bit, dim, num, ratio)
        global_varying_query_ratio[ratio[0]] = measurement_pattern(BMCs, bit, dim, sampling_ratios=[0.01, 0.02, 0.04, 0.08, 0.1, 1.0], type="global", windows=windows)
        local_varying_query_ratio[ratio[0]] = measurement_pattern(BMCs, bit, dim, sampling_ratios=[0.01, 0.02, 0.04, 0.08, 0.1, 1.0], type="local", windows=windows)
    json.dump(global_varying_query_ratio, open("../result/efficiency/global_varying_query_ratio.json", "w"))
    json.dump(local_varying_query_ratio, open("../result/efficiency/local_varying_query_ratio.json", "w"))

def varying_bits(bits, dim, num, ratio):
    # print('varying_bits', bits, dim, num, ratio)
    global_varying_bits = {}
    local_varying_bits = {}
    for bit in bits:
        BMCs = BMC_generation(bit)
        windows = get_query_windows(bit, dim, num, ratio)
        global_varying_bits[bit] = measurement_pattern(BMCs, bit, dim, sampling_ratios=[0.01, 0.02, 0.04, 0.08, 0.1, 1.0], type="global", windows=windows)
        local_varying_bits[bit] = measurement_pattern(BMCs, bit, dim, sampling_ratios=[0.01, 0.02, 0.04, 0.08, 0.1, 1.0], type="local", windows=windows)
    json.dump(global_varying_bits, open("../result/efficiency/global_varying_bits.json", "w"))
    json.dump(local_varying_bits, open("../result/efficiency/local_varying_bits.json", "w"))
        
if __name__ == "__main__":
    default_bit = 8
    default_num = 256
    default_dim = 2
    default_ratio = [0.01, 0.01]
    bits = [8, 10, 12, 14, 16]
    dims = [2, 3]
    dim_ratios = [[0.01, 0.01], [0.01, 0.01, 0.01]]
    nums=[256, 512, 1024, 2048, 4096]
    ratios = [[0.01, 0.01], [0.02, 0.02], [0.04, 0.04], [0.08, 0.08], [0.16, 0.16]]
    BMCs = BMC_generation(default_bit, default_dim, num=6)
    varying_query_nums(BMCs, bit=default_bit, dim=default_dim, nums=nums, ratio=default_ratio)
    varying_query_ratio(BMCs, bit=default_bit, dim=default_dim, num=default_num, ratios=ratios)
    varying_bits(bits=bits, dim=default_dim, num=default_num, ratio=default_ratio)
    varying_dims(bit=default_bit, dims=dims, num=default_num, ratios=dim_ratios)
    
    
   