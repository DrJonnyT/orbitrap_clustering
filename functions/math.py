# -*- coding: utf-8 -*-
import numpy as np

##E.g. round to nearest multiple of 2 would be rounded to 0, 2, 4, etc
def round_to_nearest_x_even(num,x=2):
    """
    E.g. round to nearest multiple of 2 would be rounded to 0, 2, 4, etc
    Note it follows the logic of np.round, so for values in the middle it goes to the nearest even value
    In this context that would be values that are multiples of 4

    Parameters
    ----------
    num : variable or array
        
    x : integer
        

    Returns
    -------
    float or numpy array of floats
        num but rounded to nearest x odd number

    """
    if type(num) == list:
        num = np.array(num)
    
    return np.round(num / x) * x


def round_to_nearest_x_odd(num, x=2):
    """
    E.g. round to nearest multiple of 2 would be rounded to 1, 3, 5, etc
    Note that if x is not 2, the meaning here is unclear
    For values in the middle like 3,5 etc, this seems to always round up

    Parameters
    ----------
    num : variable or array
        
    x : integer
        

    Returns
    -------
    float or numpy array of floats
        num but rounded to nearest x odd number

    """
    if type(num) == list:
        num = np.array(num)
        
    return np.floor((num) / x) * x + x/2


def sqrt_sum_squares(x):
    """
    Sum an array in quadrature

    Parameters
    ----------
    x : array
        input

    Returns
    -------
    variable, probably a float
        The square root of (the sum of (the input array squared))

    """
    x = np.array(x)
    return np.sqrt(np.sum(np.multiply(x,x)))