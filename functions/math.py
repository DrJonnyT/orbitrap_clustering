# -*- coding: utf-8 -*-
import numpy as np
import pdb

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


def num_frac_above_val(x,val):
    """
    Calculate the number fraction of an array that is above val
    Does not work if x contains infinities

    Parameters
    ----------
    x : array of numbers
    
    val : number
        

    Returns
    -------
    frac : float
        Number fraction of x that was above val.

    """
    x = np.array(x)
    
    #Return nan if there's infinities in the array
    if np.isinf(x).sum() > 0:
        return np.nan
    
    #Return nan if val is not a normal number
    if np.isnan(val) or np.isinf(val):
        return np.nan
     
    #Select only non-nan data
    x = x[~np.isnan(x)].ravel()
    
    #Return nan if no data in array
    if(len(x)) == 0:
        return np.nan
    
    above_mean = np.greater_equal(x,val)
    frac = above_mean.sum() / len(x)
    return frac


def normdot(X,Y):
    """
    Normalised dot product of two arrays

    Parameters
    ----------
    X : array
        
    Y : array
        

    Returns
    -------
    float

    """
    return np.dot(X,Y) / np.sqrt( np.dot(X,X) * np.dot(Y,Y)   )

def normdot_1min(X,Y):
    """
    1 minus normalised dot product of two arrays

    Parameters
    ----------
    X : array
    
    Y : array

    Returns
    -------
    float

    """
    return 1 - normdot(X,Y)