import os
import pprint
import re


def read_LBMC_results(dataset_name, data_size, folder_name, width=1024, height=16384):
    base_path = f"./result/postgresql"

    dataset = '{}_{}_'.format(dataset_name, data_size)
    test_query = '{}_2000_{}_{}_dim2_norm_{}.txt'.format(dataset_name, width, height, folder_name)
    file_name = dataset + test_query
    
    folder_path = os.path.join(base_path, folder_name)            
        
    file_path = os.path.join(folder_path, file_name)
    
    if not os.path.exists(file_path):
        return None, None
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    
    time_usage_regex = r"time usage: (\d+\.\d+) s"
    query_time_usage_regex = r"finish running, time usage: (\d+\.\d+) ms"
    
    time_usage_value = None
    avg_block_access = None
    query_time_usage = None
    all_result_num = None
    
    for line in lines:
        if "avg block access:" in line:
            avg_block_access = float(line.split(":")[-1].strip())
            continue
        if "all_result_num:" in line:
            all_result_num = int(line.split(":")[-1].strip())
            continue
        
        match = re.search(time_usage_regex, line)
        if match:
            time_usage_value = float(match.group(1))
        match = re.search(query_time_usage_regex, line)
        if match:
            query_time_usage = float(match.group(1))
                
    return avg_block_access, time_usage_value, query_time_usage, all_result_num
    

def read_BMTree_results(dataset_name, data_size, width=1024, height=16384):
    base_path = "./Learned-BMTree/pgsql_result"
    dataset = '{}_{}'.format(dataset_name, data_size)
    file_name = f'{dataset}_{dataset_name}_2000_{width}_{height}_dim2_bmtree_mcts_bmtree_{dataset}_{dataset_name}_2000_{width}_{height}_dim2.txt'
    
    file_path = os.path.join(base_path, file_name)
    
    # print(file_path)
    
    if not os.path.exists(file_path):
        return None, None
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        if "avg block access:" in line:
            avg_block_access = float(line.split(":")[-1].strip())
            continue
        if "all_result_num:" in line:
            all_result_num = int(line.split(":")[-1].strip())
            continue
        if "finish running, time usage:" in line:
            matches = re.findall(r"finish running, time usage: (\d+\.\d+) ms", line)
            if matches:
                query_time_usage =float(matches[0])

            
    base_path = "./Learned-BMTree/fast_result"
    folder_name = f"{dataset_name}_{data_size}_{dataset_name}_1000_{width}_{height}_dim2/mcts/{int(data_size * 0.001)}_0_8_8_10"
    folder_path = os.path.join(base_path, folder_name)
    folder_names = os.listdir(folder_path)
    folder_values = [float(folder) for folder in folder_names if folder.replace(".", "", 1).isdigit()]
    max_value_folder = int(max(folder_values))
    file_path = os.path.join(folder_path, str(max_value_folder), "result_.txt")
    
    time_usage_regex = r"total use time: (\d+\.\d+) s"
    time_usage_value = None
    
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            match = re.search(time_usage_regex, line)
            if match:
                time_usage_value = float(match.group(1))
                
    return avg_block_access, time_usage_value, query_time_usage, all_result_num

def fig_1():
    datasets = ["OSM", "NYC", "uniform", "SKEW"]
    methods = ["BMTree", "LBMC", "QUILTS", "LC", "ZC", "HC"]
    data_sizes = [10000000]
    
    block_access_res = []
    time_cost_res = []
    query_time_cost_res = []
    all_result_num_res = []
    for method in methods:
        block_access_temp = []
        time_cost_temp = []
        query_time_temp = []
        all_result_num_temp = []
        for dataset in datasets:
            for data_size in data_sizes:
                if method == "BMTree":
                    block_access, time_cost, query_time_usage, all_result_num = read_BMTree_results(dataset, data_size)
                    block_access_temp.append(block_access)
                    time_cost_temp.append(time_cost)
                    query_time_temp.append(query_time_usage)
                    all_result_num_temp.append(all_result_num)
                else:
                    block_access, time_cost, query_time_usage, all_result_num = read_LBMC_results(dataset, data_size, method)
                    block_access_temp.append(block_access)
                    if method == "LBMC":
                        time_cost_temp.append(time_cost)
                    query_time_temp.append(query_time_usage)
                    all_result_num_temp.append(all_result_num)
                    
        block_access_res.append(block_access_temp)
        if method in ["BMTree", "LBMC"]:
            time_cost_res.append(time_cost_temp)
        query_time_cost_res.append(query_time_temp)
        all_result_num_res.append(all_result_num_temp)
    # pprint.pprint(block_access_res)
    # pprint.pprint(time_cost_res)
    return block_access_res, time_cost_res, query_time_cost_res, all_result_num_res
    
    
def fig_2():
    methods = ['BMTree', 'LBMC', 'QUILTS', 'LC', 'ZC', 'HC']
    datasets = ['OSM', 'SKEW']
    data_sizes = [10000, 100000, 1000000, 10000000, 100000000]
    dataset_block_access_res = {}
    dataset_time_cost_res = {}
    for dataset in datasets:
        # print(f"-------------dataset: {dataset}-----------------")
        block_access_res = []
        time_cost_res = []
        for method in methods:
            block_access_temp = []
            time_cost_temp = []
            for data_size in data_sizes:
                if method == "BMTree":
                    block_access, time_cost = read_BMTree_results(dataset, data_size)
                    block_access_temp.append(block_access)
                    time_cost_temp.append(time_cost)
                else:
                    block_access, time_cost = read_LBMC_results(dataset, data_size, method)
                    block_access_temp.append(block_access)
                    if method == "LBMC":
                        time_cost_temp.append(time_cost)
            block_access_res.append(block_access_temp)
            if method in ["BMTree", "LBMC"]:
                time_cost_res.append(time_cost_temp)
        dataset_block_access_res[dataset] = block_access_res
        dataset_time_cost_res[dataset] = time_cost_res
        # pprint.pprint(block_access_res)
        # pprint.pprint(time_cost_res)
    return dataset_block_access_res, dataset_time_cost_res

    

    
def fig_3():
    methods = ["BMTree", 'LBMC', 'QUILTS', 'LC', 'ZC', 'HC']
    datasets = ['OSM', 'SKEW']
    data_size = 10000000
    # widths = [1024, 1024, 1024, 1024, 1024]
    # heights = [1024, 4096, 16384, 65536, 262144]
    widths = [16384, 4096, 1024, 1024, 1024]
    heights = [1024, 1024, 1024, 4096, 16384]
    dataset_block_access_res = {}
    dataset_time_cost_res = {}
    for dataset in datasets:
        # print(f"-------------dataset: {dataset}-----------------")
        block_access_res = []
        time_cost_res = []
        for method in methods:
            block_access_temp = []
            time_cost_temp = []
            for width, height in zip(widths, heights):
                if method == "BMTree":
                    block_access, time_cost = read_BMTree_results(dataset, data_size, width, height)
                    block_access_temp.append(block_access)
                    time_cost_temp.append(time_cost)
                else:
                    block_access, time_cost = read_LBMC_results(dataset, data_size, method, width, height)
                    block_access_temp.append(block_access)
                    if method == "LBMC":
                        time_cost_temp.append(time_cost)
            block_access_res.append(block_access_temp)
            if method in ["BMTree", "LBMC"]:
                time_cost_res.append(time_cost_temp)
        dataset_block_access_res[dataset] = block_access_res
        dataset_time_cost_res[dataset] = time_cost_res
        # pprint.pprint(block_access_res)
        # pprint.pprint(time_cost_res)
    return dataset_block_access_res, dataset_time_cost_res
    
    
def fig_4():
    methods = ["BMTree", 'LBMC', 'QUILTS', 'LC', 'ZC', 'HC']
    datasets = ['OSM', 'SKEW']
    data_size = 10000000
    widths = [256, 512, 1024, 2048, 4096]
    heights = [4096, 8192, 16384, 32768, 65536]
    
    dataset_block_access_res = {}
    dataset_time_cost_res = {}
    
    for dataset in datasets:
        # print(f"-------------dataset: {dataset}-----------------")
        block_access_res = []
        time_cost_res = []
        for method in methods:
            block_access_temp = []
            time_cost_temp = []
            for width, height in zip(widths, heights):
                if method == "BMTree":
                    block_access, time_cost = read_BMTree_results(dataset, data_size, width, height)
                    block_access_temp.append(block_access)
                    time_cost_temp.append(time_cost)
                else:
                    block_access, time_cost = read_LBMC_results(dataset, data_size, method, width, height)
                    block_access_temp.append(block_access)
                    if method == "LBMC":
                        time_cost_temp.append(time_cost)
            block_access_res.append(block_access_temp)
            if method in ["BMTree", "LBMC"]:
                time_cost_res.append(time_cost_temp)
        dataset_block_access_res[dataset] = block_access_res
        dataset_time_cost_res[dataset] = time_cost_res
        # pprint.pprint(block_access_res)
        # pprint.pprint(time_cost_res)
    return dataset_block_access_res, dataset_time_cost_res
        
    
        
pprint.pprint(fig_1())
# fig_2()
# fig_3()
# fig_4()
