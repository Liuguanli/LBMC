import json
import numpy as np
import random

# Define the number of queries you want to generate
num_queries = 100

# Define the range for each dimension
min_val = 1
max_val = 1048575


def query_gen(dataset, name):
    queries = []
    name_list = name.split('_')
    n = int(name_list[1])
    width = int(name_list[2])
    height = int(name_list[3])
    
    with open('data/{}.json'.format(dataset), 'r') as f:
        dataset = json.load(f)
        while len(queries) < n:
            sample = random.choice(dataset)
            if sample[0] + width <= max_val and sample[1] + height <= max_val:
                queries.append([sample[0], sample[1], sample[0] + width, sample[1] + height])

    with open('query/{}.json'.format(name), 'w') as f:
        json.dump(queries, f)

# # Generate the queries
# queries = []
# for _ in range(num_queries):
#     x1 = np.random.randint(min_val, max_val - width)
#     x2 = x1 + width
#     y1 = np.random.randint(min_val, max_val - height)
#     y2 = y1 + height
#     queries.append([x1, x2, y1, y2])

# Write the queries to a JSON file
