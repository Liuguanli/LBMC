import json
import random
import time
import os

import numpy as np
import pandas as pd

from global_cost import GlobalCost
from local_cost import LocalCost

from utils import Window

random.seed(2023)

bit_letters = ["A", "B", "C", "D", "E"]


def generage_dataset(BMCs, bit, dim=2, num=10000, distribution="uniform", windows=None):

    dim_len = 2 ** bit
    bits_nums = [bit for i in range(dim)]
    lengths = [dim_len for i in range(dim)]
    
    GC = GlobalCost(windows, bits_nums)
    # GC.get_curve_value_via_location(location, BMC, bits_nums)
    for BMC in BMCs:
        # if os.path.exists('../data/{}_{}_{}_points.csv'.format(distribution, dim, BMC)):
        #     continue

        # Generate 100 uniform random numbers for the x and y coordinates
        x = np.random.uniform(low=0.0, high=1.0, size=num)
        y = np.random.uniform(low=0.0, high=1.0, size=num)
        curve_values = []
        for i in range(num):
            curve_values.append(GC.get_curve_value_via_location([int(x[i] * lengths[0]), int(y[i] * lengths[1])], BMC, bits_nums))
        # for i in range(dim):
        #         location.append(int(float(items[i + 1]) * lengths[i]))
        # Combine the x and y coordinates into a 2D array
        points = np.column_stack((x, y, curve_values))

        # Convert the points to a pandas DataFrame
        df = pd.DataFrame(points, columns=['x', 'y', 'curve_value'])
        df['curve_value'] = df['curve_value'].astype(int)
        # df = df.sort_values(by='curve_value', ascending=True)
        df = df.sort_values(by='curve_value', ascending=True).reset_index(drop=True)
        # df = df[['x', 'y']]

        # Save the DataFrame to a CSV file
        df.to_csv('../data/{}_{}_{}_points.csv'.format(distribution, dim, BMC), index=True, header=False)

def linear_order_generation(bit=12, dim=2):
    clockwise_LO = ""
    for i in range(dim):
        for j in range(bit):
            clockwise_LO += bit_letters[i]

    anti_clockwise_LO = ""
    for i in range(dim - 1, -1, -1):
        for j in range(bit):
            anti_clockwise_LO += bit_letters[i]

    return [clockwise_LO, anti_clockwise_LO]


def Z_Curve_generation(bit=12, dim=2):
    length = dim * bit
    clockwise_ZC = ""

    for i in range(length):
        dim_index = i % dim
        clockwise_ZC += bit_letters[dim_index]

    anti_clockwise_ZC = ""
    for i in range(dim - 1, length + dim - 1, 1):
        dim_index = i % dim
        anti_clockwise_ZC += bit_letters[dim_index]

    return [clockwise_ZC, anti_clockwise_ZC]


def BMC_generation(bit=12, dim=2, num=6):
    '''
    generage $num BMCs, each BMC has $dim dimensions, each dimension has $bits bits
    '''
    res = []
    length = dim * bit
    for j in range(num):
        temp_SFC = ""
        remained_dim_bits = [bit for i in range(dim)]

        for i in range(length):
            while True:
                dim_index = random.randint(0, dim - 1)
                if remained_dim_bits[dim_index] == 0:
                    continue
                temp_SFC += bit_letters[dim_index]
                remained_dim_bits[dim_index] -= 1
                break
        res.append(temp_SFC)

    LOs = linear_order_generation(bit, dim)
    ZCs = Z_Curve_generation(bit, dim)
    res.extend(LOs)
    res.extend(ZCs)
    return res


def generate_a_window(bit, dim, ratio):
    dim_len = 2 ** bit
    lengths = [dim_len for i in range(dim)]

    dimension_low = []
    dimension_high = []
    dimension_low_raw = []
    dimension_high_raw = []
#     random.seed(10)
    for i in range(dim):
        # set the random range [0, 1-dim_i_length]
        start_dim_i = random.random() * (1 - ratio[i])
        end_dim_i = start_dim_i + ratio[i]
        dimension_low.append(start_dim_i * lengths[i])
        dimension_high.append(end_dim_i * lengths[i])
        dimension_low_raw.append(start_dim_i)
        dimension_high_raw.append(end_dim_i)

    window = Window(dimension_low, dimension_high,
                    dimension_low_raw, dimension_high_raw)
    return window


def get_query_windows(bit=16, dim=2, num=1000, ratio=[0.01, 0.01]):
    windows = []
    for i in range(num):
        windows.append(generate_a_window(
            bit, dim, ratio))
    return windows


def random_sampling(ratio=0.1, windows=None):
    if ratio == 1:
        return windows
    return random.sample(windows, int(len(windows) * ratio))
