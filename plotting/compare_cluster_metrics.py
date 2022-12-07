# -*- coding: utf-8 -*-
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.cluster import AgglomerativeClustering, KMeans
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pdb

#sklearn_extra has some depracation warnings
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sklearn_extra.cluster import KMedoids

#%%Compare clustering metrics for a given dataset
def compare_cluster_metrics(df_data,min_clusters,max_clusters,cluster_type='agglom',suptitle_prefix='', suptitle_suffix=''):
    num_clusters_index = range(min_clusters,(max_clusters+1),1)
    ch_score = np.empty(len(num_clusters_index))
    db_score = np.empty(len(num_clusters_index))
    silhouette_scores = np.empty(len(num_clusters_index))
    
    for num_clusters in num_clusters_index:
        if(cluster_type=='agglom'):
            cluster_obj = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
            suptitle_cluster_type = 'HCA'
        elif(cluster_type=='kmeans' or cluster_type=='Kmeans' or cluster_type=='KMeans'):
            cluster_obj = KMeans(n_clusters = num_clusters)
            suptitle_cluster_type = 'KMeans'
        elif(cluster_type=='kmedoids' or cluster_type == 'Kmedoids' or cluster_type=='KMedoids'):
            cluster_obj = KMedoids(n_clusters = num_clusters)
            suptitle_cluster_type = 'KMedoids'
        else:
            raise Exception("Incorrect cluster_type")
        
        clustering = cluster_obj.fit(df_data.values)
        ch_score[num_clusters-min_clusters] = calinski_harabasz_score(df_data.values, clustering.labels_)
        db_score[num_clusters-min_clusters] = davies_bouldin_score(df_data.values, clustering.labels_)
        silhouette_scores[num_clusters-min_clusters] = silhouette_score(df_data.values, clustering.labels_)
        
    #Plot results
    fig,ax1 = plt.subplots(figsize=(10,6))
    ax2=ax1.twinx()
    ax3=ax1.twinx()
    ax3.spines.right.set_position(("axes", 1.1))
    p1, = ax1.plot(num_clusters_index,ch_score,label="CH score")
    p2, = ax2.plot(num_clusters_index,db_score,c='red',label="DB score")
    p3, = ax3.plot(num_clusters_index,silhouette_scores,c='black',label="Silhouette score")
    ax1.set_xlabel("Num clusters")
    ax1.set_ylabel("CH score")
    ax2.set_ylabel("DB score")
    ax3.set_ylabel('Silhouette score')
    
    ax1.yaxis.label.set_color(p1.get_color())
    ax2.yaxis.label.set_color(p2.get_color())
    ax3.yaxis.label.set_color(p3.get_color())
    #pdb.set_trace()
    
    ax1.spines['left'].set_color(p1.get_color())
    ax1.spines.right.set_visible(False)
    ax1.tick_params(axis='y', colors=p1.get_color())
    
    ax2.spines['right'].set_color(p2.get_color())
    ax2.tick_params(axis='y', colors=p2.get_color())
    ax2.spines.left.set_visible(False)
    
    ax3.spines['right'].set_color(p3.get_color())
    ax3.tick_params(axis='y', colors=p3.get_color())
    ax3.spines.left.set_visible(False)
    
    ax1.legend(handles=[p1, p2, p3])
    
    #ax1.xaxis.set_major_locator(plticker.MultipleLocator(base=5.0))
    #ax1.xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.suptitle(suptitle_prefix + suptitle_cluster_type + suptitle_suffix)
    plt.show()


#%%Compare clustering metrics for a given dataset

def compare_cluster_metrics_fn(data,min_clusters,max_clusters,arg_dict,**kwargs):
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
    compare_cluster_metrics_fn(df[['Feature1', 'Feature2']],2,5,arg_dict=arg_dict,sklearn_clust_fn=AgglomerativeClustering)
    
    or
    
    arg_dict = {
        "criterion": "maxclust",
        "metric" : "euclidean",
        "method" : "ward"
    }
    compare_cluster_metrics_fn(df[['Feature1', 'Feature2']],2,5,arg_dict=arg_dict,scipy_clust_fn=fclusterdata)

    """
    
    #Sort out suptitle
    if 'suptitle' in kwargs:
        suptitle = kwargs.get('suptitle')
    else:
        suptitle = ''
        
    
           
    
    
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
        
    
    
    num_clusters_index = range(min_clusters,(max_clusters+1),1)
    ch_score = np.empty(len(num_clusters_index))
    db_score = np.empty(len(num_clusters_index))
    silhouette_scores = np.empty(len(num_clusters_index))
    
    for num_clusters in num_clusters_index:
        if clust_type == 'sklearn':
            arg_dict['n_clusters'] = num_clusters
            clust = sklearn_clust_fn(**arg_dict)
            labels = clust.fit_predict(data)
            ch_score[num_clusters-min_clusters] = calinski_harabasz_score(data, labels)
            db_score[num_clusters-min_clusters] = davies_bouldin_score(data, labels)
            silhouette_scores[num_clusters-min_clusters] = silhouette_score(data, labels)
        elif clust_type == 'scipy':
            arg_dict['t'] = num_clusters
            labels = scipy_clust_fn(data,**arg_dict)
            ch_score[num_clusters-min_clusters] = calinski_harabasz_score(data, labels)
            db_score[num_clusters-min_clusters] = davies_bouldin_score(data, labels)
            silhouette_scores[num_clusters-min_clusters] = silhouette_score(data, labels)
        
    #Plot results
    fig,ax1 = plt.subplots(figsize=(10,6))
    ax2=ax1.twinx()
    ax3=ax1.twinx()
    ax3.spines.right.set_position(("axes", 1.1))
    p1, = ax1.plot(num_clusters_index,ch_score,label="CH score")
    p2, = ax2.plot(num_clusters_index,db_score,c='red',label="DB score")
    p3, = ax3.plot(num_clusters_index,silhouette_scores,c='black',label="Silhouette score")
    ax1.set_xlabel("Num clusters")
    ax1.set_ylabel("CH score")
    ax2.set_ylabel("DB score")
    ax3.set_ylabel('Silhouette score')
    
    ax1.yaxis.label.set_color(p1.get_color())
    ax2.yaxis.label.set_color(p2.get_color())
    ax3.yaxis.label.set_color(p3.get_color())
    #pdb.set_trace()
    
    ax1.spines['left'].set_color(p1.get_color())
    ax1.spines.right.set_visible(False)
    ax1.tick_params(axis='y', colors=p1.get_color())
    
    ax2.spines['right'].set_color(p2.get_color())
    ax2.tick_params(axis='y', colors=p2.get_color())
    ax2.spines.left.set_visible(False)
    
    ax3.spines['right'].set_color(p3.get_color())
    ax3.tick_params(axis='y', colors=p3.get_color())
    ax3.spines.left.set_visible(False)
    
    ax1.legend(handles=[p1, p2, p3])
    
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.suptitle(suptitle)
    #plt.show()

