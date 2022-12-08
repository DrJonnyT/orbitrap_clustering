# -*- coding: utf-8 -*-
#To avoid KMeans memory leak
import os
os.environ["OMP_NUM_THREADS"] = '1'

from sklearn.cluster import AgglomerativeClustering, KMeans
import numpy as np
import pdb
import pandas as pd

#sklearn_extra has some depracation warnings
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sklearn_extra.cluster import KMedoids
    
#%%
def cluster_n_times(data,min_clusters,max_clusters,cluster_type='agglom'):
    """
    Run Clustering a set number of times, scanning through different k number of clusters

    Parameters
    ----------
    data : array
        Data to cluster. Probably a 2D matrix. Ideally pandas dataframe so the index works in the output
    min_clusters : int
        Minimum number of clusters to plot.
    max_clusters : int
        Maximum number of clusters to plot.
    cluster_type : string, optional
        Type of clustering to be performed. The default is 'agglom'.

    Returns
    -------
    df_cluster_labels_mtx : dataframe
        Matrix of cluster labels. Index is the same as data if data was a dataframe. Columns are n_clusters

    """
    num_clusters_index = np.arange(min_clusters,max_clusters+1)
    cluster_labels_mtx = []
    
    for num_clusters in num_clusters_index:
        #First run the clustering
        if(cluster_type=='agglom'):
            cluster_obj = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
        elif(cluster_type=='kmeans' or cluster_type=='Kmeans' or cluster_type=='KMeans'):
            cluster_obj = KMeans(n_clusters = num_clusters)

            warnings.filterwarnings('ignore')
        elif(cluster_type=='kmedoids' or cluster_type == 'Kmedoids' or cluster_type=='KMedoids'):
            cluster_obj = KMedoids(n_clusters = num_clusters)
        else:
            raise ValueError("Invalid value of cluster_type")

        #Ignore KMeans warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clustering = cluster_obj.fit(data)
        
        cluster_labels_mtx.append(clustering.labels_)
        
    df_cluster_labels_mtx = pd.DataFrame(cluster_labels_mtx,index=num_clusters_index).T.rename_axis(columns="num_clusters")
    if type(data) == pd.core.frame.DataFrame:
        df_cluster_labels_mtx.index=data.index
    return df_cluster_labels_mtx



def cluster_n_times_fn(data,min_clusters,max_clusters,arg_dict,**kwargs):
    """
    Compare clustering metrics for a given dataset

    Parameters
    ----------
    data : array
        Data to cluster. Probably a 2D matrix
    min_clusters : int
        Minimum number of clusters to plot.
    max_clusters : int
        Maximum number of clusters to plot.
    arg_dict : dictionary
        A dictionary of arguments to pass into the clustering function
    **kwargs
        Required: one of "sklearn_clust_fn" or "scipy_clust_fn" which would be the clustering function
        Optional: Suptitle, title at the top of the plot

    Raises
    ------
    ValueError
        If input is not correct.

    Returns
    -------
    None.
    
    Example usage
    -------
    df = pd.DataFrame({
        "Feature1": [6.05, 5.1, 1, 2, 3.1, 4],
        "Feature2": [5.05, 6, 2.1, 1.01, 4.1, 3],
    })
    arg_dict = {"linkage": "ward"}
    cluster_n_times_fn(df[['Feature1', 'Feature2']],2,5,arg_dict=arg_dict,sklearn_clust_fn=AgglomerativeClustering)
    
    or
    
    arg_dict = {
        "criterion": "maxclust",
        "metric" : "euclidean",
        "method" : "ward"
    }
    cluster_n_times_fn(df[['Feature1', 'Feature2']],2,5,arg_dict=arg_dict,scipy_clust_fn=fclusterdata)

    """
    
    if 'sklearn_clust_fn' in kwargs:
        if 'scipy_clust_fn' in kwargs:
            raise ValueError('Cannot input both sklearn_clust_fn and scipy_clust_fn')
        else:
            sklearn_clust_fn = kwargs.get('sklearn_clust_fn')
            clust_type = 'sklearn'
    elif 'scipy_clust_fn' in kwargs:
        scipy_clust_fn = kwargs.get('scipy_clust_fn')
        clust_type = 'scipy'
        arg_dict['criterion'] = 'maxclust'
        
    
    if type(data) == pd.core.frame.DataFrame:
        data_to_cluster = data.values
    else:
        data_to_cluster = data
    
    
    num_clusters_index = range(min_clusters,(max_clusters+1),1)
    cluster_labels_mtx = []
    #pdb.set_trace()
        
    for num_clusters in num_clusters_index:
        if clust_type == 'sklearn':
            arg_dict['n_clusters'] = num_clusters
            clust = sklearn_clust_fn(**arg_dict)
            labels = clust.fit_predict(data_to_cluster)
    
        elif clust_type == 'scipy':
            arg_dict['t'] = num_clusters
            labels = scipy_clust_fn(data_to_cluster,**arg_dict)
            
        cluster_labels_mtx.append(labels)
    
    df_cluster_labels_mtx = pd.DataFrame(cluster_labels_mtx,index=num_clusters_index).T.rename_axis(columns="num_clusters")
    if type(data) == pd.core.frame.DataFrame:
        df_cluster_labels_mtx.index=data.index
    return df_cluster_labels_mtx
