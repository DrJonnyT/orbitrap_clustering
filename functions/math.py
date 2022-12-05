# -*- coding: utf-8 -*-
import numpy as np
##E.g. round to nearest multiple of 2 would be rounded to 0, 2, 4, etc
def round_to_nearest_x_even(num,x):
    return np.round(num / x) * x
##E.g. round to nearest multiple of 2 would be rounded to 0, 2, 4, etc
def round_to_nearest_x_odd(num,x):
    return np.floor((num) / x) * x +x/2


def sqrt_sum_squares(x):
    x = np.array(x)
    return np.sqrt(np.sum(np.multiply(x,x)))