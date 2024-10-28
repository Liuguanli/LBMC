
import bisect
import math
from typing import List

class BlockQuery:
    def __init__(self, points, sfc_values):
        self.min_point = points[0]
        self.max_point = points[1]
        self.min_sfc_value = sfc_values[0]
        self.max_sfc_value = sfc_values[1]

    def inside(self, block):

        res = []
        
        for point in block.block_data:
            if self.point_inside(point):
                res.append(point)

        return res
    
    def point_inside(self, point):
        for i, dim_value in enumerate(point.dim_values):
            if not (self.min_point[i] <= dim_value <= self.max_point[i]):
                return False
        return True
    


class Point:
    def __init__(self, dim_values, sfc_value):
        self.dim_values = dim_values
        self.sfc_value = sfc_value


class Block:
    def __init__(self, block_id, block_data, block_type):
        self.block_id = block_id
        self.block_data = block_data
        self.block_type = block_type


class Storage:
    def __init__(self, data, block_size):

        self.block_size = block_size
        self.block_list = []

        # sort data by sfc value
        data.sort(key=lambda point: point.sfc_value)

        self.index_sfc = [point.sfc_value for point in data]

        blocks = [data[i:i + block_size] for i in range(0, len(data), block_size)]

        for index, b in enumerate(blocks):
            block = Block(index, b, 'data')
            self.block_list.append(block)


    def window_query_all(self, queries : List[BlockQuery]):
        query_res = []
        avg_block_acc_num = 0
        for query in queries:
            query_res, block_acc_num = self.window_query(query)
            avg_block_acc_num += block_acc_num
        return avg_block_acc_num / len(queries)

    def window_query(self, query : BlockQuery):
        
        query_res = []
        # binary search
        left_bound = bisect.bisect_left(self.index_sfc, query.min_sfc_value)
        right_bound = bisect.bisect_right(self.index_sfc, query.max_sfc_value)
        
        left_block_id = left_bound // self.block_size
        right_block_id = math.ceil(right_bound / self.block_size)

        for block in self.block_list[left_block_id:right_block_id + 1]:
            query_res.extend(query.inside(block))
        
        return query_res, right_block_id - left_block_id + 1

