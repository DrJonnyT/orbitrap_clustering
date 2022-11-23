# -*- coding: utf-8 -*-
import numpy as np
from pandas import Series
import pdb
def avg_array_clusters(labels,data,weights=None,removenans=True):
    """
    Average an array over cluster labels

    Parameters
    ----------
    labels : integer array
        Cluster labels.
    data : array
        Data to be averaged. Currently only 1-D arrays are supported
    weights : array
        Scale factor to scale data before it is summed, then divide by sum of weights
        Example usage would be if you wanted the data points with higher concentration to be scaled accordingly, weights would be concentration

    Returns
    -------
    data_avg : array
        The average value of data for every unique value in labels

    """
    
    #Error if data is not 1D
    if len(np.shape(data)) > 1:
        raise ValueError("data needs to be a 1-D array")
        
    if(np.shape(labels) != np.shape(data)):
        raise ValueError("labels and data need to be same dimensions")
     
    
    
    data = np.array(data)
    labels= np.array(labels)
    unique = np.unique(labels)
    
    #Create output series
    data_avg = Series(index = unique, dtype=float)
    
    if weights is None:
        weights = np.ones(data.shape)
    else:
        weights = np.array(weights)
        if(np.shape(data) != np.shape(weights)):
            raise ValueError("data and weights need to have the same dimensions")
    
    if(removenans):
        nonans_index = np.where(np.isfinite(labels) * np.isfinite(data) * np.isfinite(weights))[0]
        labels = labels[nonans_index]
        data = data[nonans_index]
        weights = weights[nonans_index]
        
    
    for label in unique:
        data_label = data[labels == label]
        weights_label = weights[labels == label]
        data_avg[label] = np.sum(weights_label * data_label) / np.sum(weights_label)
        
    return data_avg