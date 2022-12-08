# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from functions.math import num_frac_above_val
def cluster_nfrac_above_avg(data,cluster_labels,avgtype='mean'):
    """
    Calculate the number fraction of a data array that are above the average

    Parameters
    ----------
    data : array
        Data to average. Probably a 1D array. Must have same dimension as cluster_labels
    cluster_labels: int array

    cluster_type : string, optional
        Type of average. The default is 'mean'.

    Returns
    -------
    df_out : dataframe
        Matrix of number frac above the average value. Columns are the unique cluster labels

    """    
    data = np.array(data)
    
    #Calculate the average value
    if avgtype == 'mean':
        avg_val = np.mean(data)
    elif avgtype == 'median':
        avg_val = np.median(data)
    else:
        raise ValueError("Invalid value of avgtype")
    
    
    
    
    all_clusters = np.unique(cluster_labels)
    
    df_out = pd.Series(index = all_clusters,dtype='float')
    
    for cluster in all_clusters:
        data_thisclust= data[cluster_labels == cluster]
        df_out[cluster] = num_frac_above_val(data_thisclust,avg_val)
        
    return df_out
    
