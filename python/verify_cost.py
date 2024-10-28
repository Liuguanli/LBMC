# this file is used to verify the cost measurements


from app1 import calculate_local_cost
from app1 import calculate_global_cost
from app1 import get_reading_map
from app1 import get_index_map


import random
from utils import Point
from utils import Window

from global_cost import GlobalCost
from local_cost import LocalCost

from app_query_gen import read_windows


def generate_a_window(unit_len, dim, ratio, dim_scalar):
    lengths = [unit_len * rat for rat in ratio]
    dimension_low = []
    dimension_high = []
    dimension_low_raw = []
    dimension_high_raw = []
#     random.seed(10)
    for i in range(dim):
        # set the random range [0, 1-dim_i_length]
        start_dim_i = random.random() * (1 - lengths[i])
        end_dim_i = start_dim_i + lengths[i]
        dimension_low.append(start_dim_i * dim_scalar[i])
        dimension_high.append(end_dim_i * dim_scalar[i])
        dimension_low_raw.append(start_dim_i)
        dimension_high_raw.append(end_dim_i)

    window = Window(dimension_low, dimension_high,
                    dimension_low_raw, dimension_high_raw)
    return window


def get_query_windows(unit_len, dim, ratios, nums, dim_scalar):
    windows = []
    for i in range(len(nums)):
        for j in range(nums[i]):
            windows.append(generate_a_window(
                unit_len, dim, ratios[i], dim_scalar))
    return windows


def get_2d(bits_nums, BMC, num=10):
    unit_len = 0.02
    dim = 2
#     ratios = [[1.0, 2.0], [1.0, 1.0],  [2.0, 1.0],[1.0, 4.0],[4.0, 1.0]]
    ratios = [[1.0, 2.0]]
    nums = [num]
    dim_scalar = [(pow(2, l) - 1) for l in bits_nums]
    windows = get_query_windows(unit_len, dim, ratios, nums, dim_scalar)
    return windows, bits_nums, dim

def get_3d(bits_nums, BMC, num=10):
    unit_len=0.2
    dim = 3
#     ratios = [[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [2.0, 1.0, 2.0],[1.0, 4.0, 1.0],[4.0, 1.0, 1.0]]
    ratios = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],[1.0, 1.0, 1.0],[1.0, 1.0, 1.0]]
#     nums = [num, num, num, num, num]
    nums = [num]
#     bits_nums = [8,8,8]
#     BMC='ABCABCABCABCABCABCABCABC'
    dim_scalar = [(pow(2,l) - 1) for l in bits_nums]
    windows = get_query_windows(unit_len, dim, ratios, nums, dim_scalar)
    return windows, bits_nums, dim


def verify_cost(dim: int):

    if dim == 2:
        BMC = "ABABABABABABABAB"
        windows, bits_nums, dim = get_2d(bits_nums=[8, 8], BMC=BMC, num=100)

    if dim == 3:
        BMC = 'ABCABCABCABCABCABCABCABC'
        windows, bits_nums, dim = get_3d(bits_nums = [8,8,8], BMC=BMC, num=10)


    reading_map = get_reading_map(BMC, dim)
    index_map = get_index_map(BMC)
    my_gc = 0
    my_lc = 0
    GC = GlobalCost(windows, bits_nums)
    LC = LocalCost(windows, bits_nums)

    for i, window in enumerate(windows):
        my_gc += calculate_global_cost(window, index_map, bits_nums)
        my_lc += calculate_local_cost(window, reading_map)

    naive_gc = GC.naive_global_cost(BMC)[0]
    naive_lc = LC.naive_local_cost(BMC)[0]

    my_gc_all = GC.global_cost(BMC)[0]
    my_lc_all = LC.local_cost(BMC)[0]
    assert my_gc == naive_gc, ("wrong global cost calculation my_gc:%d, naive_gc:%d", (my_gc, naive_gc))
    assert my_gc_all == naive_gc, ("wrong global cost calculation my_gc_all:%d, naive_gc:%d", (my_gc_all, naive_gc))
    assert my_lc == naive_lc, ("wrong local cost calculation my_lc:%d, naive_lc:%d", (my_lc, naive_lc))
    assert my_lc_all == naive_lc, ("wrong local cost calculation my_lc_all:%d, naive_lc:%d", (my_lc_all, naive_lc))
    print("Yeah! All correct")

if __name__ == "__main__":
    verify_cost(2)
    verify_cost(3)