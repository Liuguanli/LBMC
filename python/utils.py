from builtins import property
import math

floder = "SIGMOD2023"

bit_letters = ["A", "B", "C", "D", "E", "F", "G"]
factor_letters = ["a", "b", "c", "d", "e", "f", "g"]
logger_print = True

pow_of_two = [int(pow(2, i)) for i in range(128)]

class Config:
    def __init__(self, dim_length):
        self.dim_length = dim_length
        self.dim = len(dim_length)

class Point:
    def __init__(self, xs, value=0):
        self.xs = xs
        self.value = value
        self.dim = len(xs)

    def __str__(self):
        return "pos: " + " ".join(map(str, self.xs)) + " val: " + str(self.value) + "\n"

    def __repr__(self):
        return "pos: " + " ".join(map(str, self.xs)) + " val: " + str(self.value) + "\n"


class Window:
    def __init__(self, dimension_low, dimension_high, dimension_low_raw, dimension_high_raw):
        assert len(dimension_low) == len(
            dimension_high), "dimension_low and dimension_high should be same dimension"
        self.point_l = Point(dimension_low)
        self.point_h = Point(dimension_high)
        self.dimension_low = [int(_) for _ in dimension_low]
        self.dimension_high = [int(_) for _ in dimension_high]
        self.dimension_low_raw = dimension_low_raw
        self.dimension_high_raw = dimension_high_raw
        self.dim = len(dimension_low)
        self.ratio = 1
        self.area = self.get_area()

    def get_area(self):
        area = 1
        for high, low in zip(self.dimension_high, self.dimension_low):
            area *= (high - low + 1)
        return area

    def calculate_drop_pattern(self, drop_dim: int, drop_index: int):
        """
            a * 2 ^k - 1 -------> (a - 1) * 2^k
        """
        # range = [self.dimension_low[drop_dim], self.dimension_high[drop_dim]]
        if self.dimension_high[drop_dim] - self.dimension_low[drop_dim] < pow_of_two[drop_index]:
            return 0
        end = math.floor((self.dimension_high[drop_dim] + 1) / pow_of_two[drop_index])
        start = math.ceil(self.dimension_low[drop_dim] / pow_of_two[drop_index]) + 1
        # if end - start < 0:
        #     print(end - start, range, drop_index, end , start)
        return end - start + 1


    def calculate_rise_pattern(self, rise_dim: int, rise_index: int):
        """
            a * 2 ^k + (2^{k - 1} - 1) -------> a * 2 ^k + 2^{k - 1}
        """
        # range = [self.dimension_low[rise_dim], self.dimension_high[rise_dim]]
        # start = math.ceil((range[0] - pow(2, rise_index - 1) + 1) / pow(2, rise_index))
        # end = math.floor((range[1] - pow(2, rise_index - 1)) / pow(2, rise_index))
        start = math.ceil((self.dimension_low[rise_dim] - pow_of_two[rise_index - 1] + 1) / pow_of_two[rise_index])
        end = math.floor((self.dimension_high[rise_dim] - pow_of_two[rise_index - 1]) / pow_of_two[rise_index])
        return end - start + 1

    def gen_drop_patterns(self, bit_nums):
        self.drop_patterns = []
        for i in range(self.dim):
            dim_patterns = []
            for changed_bits in range(bit_nums[i] + 1):
                dim_patterns.append(self.calculate_drop_pattern(i, changed_bits))
            self.drop_patterns.append(dim_patterns)
        # return self.drop_patterns

    def gen_rise_patterns(self, bit_nums):
        self.rise_patterns = []
        for i in range(self.dim):
            dim_patterns = []
            for changed_bits in range(bit_nums[i]):
                dim_patterns.append(self.calculate_rise_pattern(i, changed_bits + 1))
            self.rise_patterns.append(dim_patterns)
        # return self.rise_patterns

    def __str__(self):
        return "pl: " + str(self.point_l) + " ph: " + str(self.point_h)

    def __repr__(self):
        # res = [self.point_h.xs[i] - self.point_l.xs[i] for i in range(self.dim)]
        # return str(res[-1] / (res[-2] + 1))
    
        return "pl: " + str(self.point_l) + " ph: " + str(self.point_h)
    


def ratio_to_pattern(ratios):
    res = ""
    for i in range(len(ratios) - 1):
        res += str(int(ratios[i])) + "_"
    res += str(int(ratios[-1]))
    return res