# -*- coding: utf-8 -*-
import numpy as np
from pandas import Series
def avg_array_clusters(labels,data, **kwargs):
    """
    Average an array over cluster labels

    Parameters
    ----------
    labels : integer array
        Cluster labels.
    data : array
        Data to be averaged. Currently only 1-D arrays are supported

    Returns
    -------
    data_avg : array
        The average value of data for every unique value in labels

    """
    
    if len(np.shape(data)) > 1:
        raise ValueError("data needs to be a 1-D array")
    
    data = np.array(data)
    
    unique = np.unique(labels)
    
    data_avg = Series(index = unique, dtype=float)
    
    for label in unique:
        data_avg[label] = np.mean(data[labels == label],axis=0)
        
    return data_avg