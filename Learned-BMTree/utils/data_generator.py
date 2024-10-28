import json
import numpy as np
from scipy.stats import skewnorm

# Define the number of data points you want to generate


# Define the range for each dimension
min_val = 1
max_val = 2 ** 20 - 1

# Define the skewness parameter for the skewnorm distribution
# positive values mean the skew is to the right, negative values skew to the left
skewness = 3


def data_gen(data_info):
    data = data_info.split('_')
    num_data_points = int(data[1])
    distribution = data[0]
    if distribution == "uniform":

        # Generate uniformly distributed data
        uniform_data = [[np.random.randint(min_val, max_val) for _ in range(
            2)] for _ in range(num_data_points)]
        with open('./data/uniform_{}.json'.format(num_data_points), 'w') as f:
            json.dump(uniform_data, f)

    elif distribution == "normal":
        # Generate normally distributed data
        normal_data = [[int(max(min(np.random.normal(loc=(max_val + min_val) / 2, scale=(
            max_val - min_val) / 4), max_val), min_val)) for _ in range(2)] for _ in range(num_data_points)]

        with open('./data/normal_{}.json'.format(num_data_points), 'w') as f:
            json.dump(normal_data, f)

    elif distribution == "skewed":
        # Generate skewed data
        skewed_data = [[int(max(min(skewnorm.rvs(skewness, loc=(max_val + min_val) / 2, scale=(
            max_val - min_val) / 4), max_val), min_val)) for _ in range(2)] for _ in range(num_data_points)]

        # Write the data to a JSON file

        with open('./data/skewed_{}.json'.format(num_data_points), 'w') as f:
            json.dump(skewed_data, f)
