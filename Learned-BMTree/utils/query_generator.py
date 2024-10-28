import json
import numpy as np
import random
import os
import pandas as pd

# Define the number of queries you want to generate
num_queries = 100

# Define the range for each dimension
min_val = 1
max_val = 1048575

# Define the width and height of the query range
width = 16384 * 2
height = width


def query_gen(dataset, name):
    queries = []
    queries_norm = []
    name_list = name.split('_')
    n = int(name_list[1])
    if len(name_list) >  3:
        width = int(name_list[2])
        height = int(name_list[3])

    with open('data/{}.json'.format(dataset), 'r') as f:
        dataset = json.load(f)
        while len(queries) < n:
            sample = random.choice(dataset)
            if sample[0] + width <= max_val and sample[1] + height <= max_val:
                queries.append(
                    [sample[0], sample[1], sample[0] + width, sample[1] + height])
                queries_norm.append([sample[0] / max_val, sample[1] / max_val,
                                     (sample[0] + width) / max_val, (sample[1] + height) / max_val])

    with open('query/{}.json'.format(name), 'w') as f:
        json.dump(queries, f)
    with open('query/{}_norm.json'.format(name), 'w') as f:
        json.dump(queries_norm, f)



def query_gen_mix(dataset_name, n, widths, heights):
    queries = []
    queries_norm = []

    dataset_name_prefix = dataset_name.split('_')[0]

    if not os.path.exists("query/" + 'query/{}_{}_mix.json'.format(dataset_name_prefix, n)) or \
        not os.path.exists("query/" + 'query/{}_{}_mix_norm.json'.format(dataset_name_prefix, n)):

        with open('data/{}.json'.format(dataset_name), 'r') as f:
            dataset = json.load(f)
            while len(queries) < n:
                width = random.choice(widths)
                height = random.choice(heights)
                sample = random.choice(dataset)
                if sample[0] + width <= max_val and sample[1] + height <= max_val:
                    queries.append(
                        [sample[0], sample[1], sample[0] + width, sample[1] + height])
                    queries_norm.append([sample[0] / max_val, sample[1] / max_val,
                                        (sample[0] + width) / max_val, (sample[1] + height) / max_val])

        with open('query/{}_{}_mix.json'.format(dataset_name_prefix, n), 'w') as f:
            json.dump(queries, f)
        with open('query/{}_{}_mix_norm.json'.format(dataset_name_prefix, n), 'w') as f:
            json.dump(queries_norm, f)

def query_gen_mix_distribution(dataset_name, ratio, widths, heights, n=2000):
    queries = []
    queries_norm = []

    dataset_name_prefix = dataset_name.split('_')[0]

    n2 = int(n / 100 * ratio)
    n1 = n - n2

    if not os.path.exists("query/" + 'query/{}_{}_mix{}.json'.format(dataset_name_prefix, n, ratio)) or \
        not os.path.exists("query/" + 'query/{}_{}_mix{}_norm.json'.format(dataset_name_prefix, n, ratio)):

        with open('data/{}.json'.format(dataset_name), 'r') as f:
            dataset = json.load(f)
            
            while len(queries) < n1:
                width = random.choice(widths)
                height = random.choice(heights)
                sample = random.choice(dataset)
                if sample[0] + width <= max_val and sample[1] + height <= max_val:
                    queries.append(
                        [sample[0], sample[1], sample[0] + width, sample[1] + height])
                    queries_norm.append([sample[0] / max_val, sample[1] / max_val,
                                        (sample[0] + width) / max_val, (sample[1] + height) / max_val])
            
            dataset = np.array(dataset)
            df = pd.DataFrame(dataset)
            while len(queries) < n2:
                width = random.choice(widths)
                height = random.choice(heights)

                mean = df.mean().values
                std = df.std().values
                sample = np.random.normal(loc=mean, scale=std, size=(1, 2))[0]
                # sample = random.choice(dataset)
                if sample[0] + width <= max_val and sample[1] + height <= max_val and sample[0] >= 0 and sample[1] >= 0:
                    queries.append(
                        [int(sample[0]), int(sample[1]), int(sample[0]) + width, int(sample[1]) + height])
                    queries_norm.append([sample[0] / max_val, sample[1] / max_val,
                                        (sample[0] + width) / max_val, (sample[1] + height) / max_val])

        with open('query/{}_{}_mix{}.json'.format(dataset_name_prefix, n, ratio), 'w') as f:
            json.dump(queries, f)
        with open('query/{}_{}_mix{}_norm.json'.format(dataset_name_prefix, n, ratio), 'w') as f:
            json.dump(queries_norm, f)


def query_gen_mix_distribution_incremental(dataset_name, widths, heights, ratios=[0, 25, 50, 75, 100], ns=[2000]):


    queries1 = []
    queries_norm1 = []

    queries2 = []
    queries_norm2 = []
    
    dataset_name_prefix = dataset_name.split('_')[0]

    for n in ns:
        # if not os.path.exists("query/" + 'query/{}_{}_incremental{}.json'.format(dataset_name_prefix, n, ratio)) or \
        #     not os.path.exists("query/" + 'query/{}_{}_norm_incremental{}.json'.format(dataset_name_prefix, n, ratio)):

        with open('data/{}.json'.format(dataset_name), 'r') as f:
            dataset = json.load(f)
            while len(queries1) < n:
                width = random.choice(widths)
                height = random.choice(heights)
                sample = random.choice(dataset)
                if sample[0] + width <= max_val and sample[1] + height <= max_val:
                    queries1.append(
                        [sample[0], sample[1], sample[0] + width, sample[1] + height])
                    queries_norm1.append([sample[0] / max_val, sample[1] / max_val,
                                        (sample[0] + width) / max_val, (sample[1] + height) / max_val])
            
            dataset = np.array(dataset)
            df = pd.DataFrame(dataset)
            while len(queries2) < n:
                width = random.choice(widths)
                height = random.choice(heights)

                mean = df.mean().values
                std = df.std().values
                sample = np.random.normal(loc=mean, scale=std, size=(1, 2))[0]
                # sample = random.choice(dataset)
                if sample[0] + width <= max_val and sample[1] + height <= max_val and sample[0] >= 0 and sample[1] >= 0:
                    queries2.append(
                        [int(sample[0]), int(sample[1]), int(sample[0]) + width, int(sample[1]) + height])
                    queries_norm2.append([sample[0] / max_val, sample[1] / max_val,
                                        (sample[0] + width) / max_val, (sample[1] + height) / max_val])




        for ratio in ratios:
            if ratio == 0:
                with open('query/{}_{}_incremental.json'.format(dataset_name_prefix, n), 'w') as f:
                    json.dump(queries2, f)
                with open('query/{}_{}_norm_incremental.json'.format(dataset_name_prefix, n), 'w') as f:
                    json.dump(queries_norm2, f)
            elif ratio == 100:
                with open('query/{}_{}_incremental{}.json'.format(dataset_name_prefix, n, ratio), 'w') as f:
                    json.dump(queries1, f)
                with open('query/{}_{}_norm_incremental{}.json'.format(dataset_name_prefix, n, ratio), 'w') as f:
                    json.dump(queries_norm1, f)
            else:
                n1 = int(n * ratio / 100)
                n2 = n - n1
                queries = []
                queries_norm = []
                queries.extend(queries2[:n2])
                queries.extend(queries1[:n1])
                queries_norm.extend(queries_norm2[:n2])
                queries_norm.extend(queries_norm1[:n1])

                with open('query/{}_{}_incremental{}.json'.format(dataset_name_prefix, n, ratio), 'w') as f:
                    json.dump(queries, f)
                with open('query/{}_{}_norm_incremental{}.json'.format(dataset_name_prefix, n, ratio), 'w') as f:
                    json.dump(queries_norm, f)


def query_gen_mix_incremental(dataset_name, ns, widths, heights):
    queries = []
    queries_norm = []
    
    dataset_name_prefix = dataset_name.split('_')[0]

    for n in ns:
        if not os.path.exists("query/" + 'query/{}_{}_incremental.json'.format(dataset_name_prefix, n)) or \
            not os.path.exists("query/" + 'query/{}_{}_norm_incremental.json'.format(dataset_name_prefix, n)):

            with open('data/{}.json'.format(dataset_name), 'r') as f:
                dataset = json.load(f)
                while len(queries) < n:
                    width = random.choice(widths)
                    height = random.choice(heights)
                    sample = random.choice(dataset)
                    if sample[0] + width <= max_val and sample[1] + height <= max_val:
                        queries.append(
                            [sample[0], sample[1], sample[0] + width, sample[1] + height])
                        queries_norm.append([sample[0] / max_val, sample[1] / max_val,
                                            (sample[0] + width) / max_val, (sample[1] + height) / max_val])

            with open('query/{}_{}_incremental.json'.format(dataset_name_prefix, n), 'w') as f:
                json.dump(queries, f)
            with open('query/{}_{}_norm_incremental.json'.format(dataset_name_prefix, n), 'w') as f:
                json.dump(queries_norm, f)

