# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from functions.math import normdot


def correlate_cluster_profiles(cluster_profiles_mtx_norm, num_clusters_index, cluster_index):
    """
    Correlate cluster mass spectral profiles

    Parameters
    ----------
    cluster_profiles_mtx_norm : dataframe
        Normalised cluster profiles.
    num_clusters_index : array of int
        Number of clusters for each row of cluster_profiles_mtx_norm
    cluster_index : array of int
        Cluster index for each col of cluster_profiles_mtx_norm

    Returns
    -------
    df_cluster_corr_mtx : dataframe
        The correlation coefficient between each cluster's average mass spec
    df_prevcluster_corr_mtx : dataframe
        The correlation coefficient between each cluster's average mass spec for the previous number of clusters

    """
    
    df_cluster_corr_mtx = pd.DataFrame(np.NaN, index = num_clusters_index, columns = cluster_index)
    df_prevcluster_corr_mtx = pd.DataFrame(np.NaN, index = num_clusters_index, columns = cluster_index)
    

    #index is the number of clusters
    #columns is the cluster in question
    for x_idx in np.arange(cluster_profiles_mtx_norm.shape[0]):
        num_clusters = num_clusters_index[x_idx]
        #print(num_clusters)
        if(num_clusters>1):
            for this_cluster in np.arange(num_clusters):
                #Find correlations with other clusters from the same num_clusters
                this_cluster_profile = cluster_profiles_mtx_norm[x_idx,this_cluster,:]  
                other_clusters_profiles = cluster_profiles_mtx_norm[x_idx,cluster_index!=this_cluster,:]
                profiles_corr = np.zeros(other_clusters_profiles.shape[0])
                for y_idx in np.arange(num_clusters-1):
                    this_other_cluster_profile = other_clusters_profiles[y_idx,:]
                    profiles_corr[y_idx] = normdot(this_cluster_profile,this_other_cluster_profile)
                    #profiles_corr[y_idx] = pearsonr(this_cluster_profile,this_other_cluster_profile)[0]
                df_cluster_corr_mtx.loc[num_clusters,this_cluster] = profiles_corr.max()
        if(num_clusters>1 and x_idx > 0):
            for this_cluster in np.arange(num_clusters):
                #Find correlations with the clusters from the previous num_clusters
                this_cluster_profile = cluster_profiles_mtx_norm[x_idx,this_cluster,:]  
                prev_clusters_profiles = cluster_profiles_mtx_norm[x_idx-1,:,:]
                profiles_corr = np.zeros(prev_clusters_profiles.shape[0])
                for y_idx in np.arange(num_clusters-1):
                    this_prev_cluster_profile = prev_clusters_profiles[y_idx,:]
                    #pdb.set_trace()
                    profiles_corr[y_idx] = normdot(this_cluster_profile,this_prev_cluster_profile)
                    #profiles_corr[y_idx] = pearsonr(this_cluster_profile,this_prev_cluster_profile)[0]
                df_prevcluster_corr_mtx.loc[num_clusters,this_cluster] = profiles_corr.max()
            
            
    return df_cluster_corr_mtx, df_prevcluster_corr_mtx