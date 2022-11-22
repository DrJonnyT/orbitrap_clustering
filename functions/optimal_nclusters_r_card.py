# -*- coding: utf-8 -*-
import numpy as np
import warnings




import pdb


def optimal_nclusters_r_card(nclusters,maxr,mincard,maxr_threshold=0.95,mincard_threshold=10):
    """
    Work out the optimal number of clusters based on the r and the cardinality (num points) of the smallest cluster

    Parameters
    ----------
    nclusters : array
        number of clusters.
    maxr : array
        maximum Pearson's R between clusters.    
    mincard : array
        cardinality (number of points) of smallest cluster.
    maxr_threshold : variable, optional
        threshold for maxr. The default is 0.95.
    mincard_threshold : variable, optional
        threshold for mincard. The default is 10.

    Returns
    -------
    variable
        Largest number of clusters for which (maxr > maxr_threshold) or neither (mincard < mincard_threshold).

    """
    
    #Make sure you can index easily
    nclusters = np.array(nclusters)
    maxr = np.array(maxr)
    mincard = np.array(mincard)
    
    #Check if first number of clusters breaches thresholds
    if maxr[0] > maxr_threshold or mincard[0] < mincard_threshold:
        warnings.warn("optimal_nclusters_r_card() the first number of clusters breaches the threshold, so just returning nan")
        return np.nan
    else:
        for index in range(len(nclusters)):
            if maxr[index] > maxr_threshold or mincard[index] < mincard_threshold:
                return nclusters[index-1]
    
    #If you reach this point, no data points above threshold
    warnings.warn("optimal_nclusters_r_card() has not found any threshold-breaching values so returning nan")
    return np.nan