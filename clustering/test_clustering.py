# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fclusterdata
from clustering import molecule_type_pos_frac
from clustering.molecule_type_math import molecule_type_pos_frac_clusters
from clustering.cluster_n_times import cluster_n_times, cluster_n_times_fn
from clustering.cluster_nfrac_above_avg  import cluster_nfrac_above_avg
from clustering.correlate_cluster_profiles import correlate_cluster_profiles
from clustering.cluster_top_percentiles import cluster_top_percentiles



def test_molecule_type_pos_frac():
    mol_types = ['CHO','CHO','CHON','CHON','CHONS','CHOS']
    data = np.array([1,1,-1,1,0,-1])
    mols_list = np.unique(mol_types)
    assert np.array_equal( molecule_type_pos_frac(data,mol_types,mols_list=mols_list).to_numpy(), [1,0.5,1,0])
    
    
def test_molecule_type_pos_frac_clusters():
    mol_types = ['CHO','CHO','CHON','CHON','CHONS','CHOS']
    data = np.array([[1,1,-1,1,0,-1],[-1,-1,-1,1,0,-1]])
    mols_list = np.unique(mol_types)
    clusters = [0,1]
    
    assert np.array_equal( molecule_type_pos_frac_clusters(data,mol_types,clusters,mols_list=mols_list).to_numpy(), 
                          np.array([[1,0.5,1,0],
                          [0,0.5,1,0]]))
    
    

def test_compare_cluster_metrics():
    df = pd.DataFrame({
        "Sample Name": ["Sample "+str(i) for i in range(6)],
        "Feature1": [0, 1, 10, 1, 30, 31],
        "Feature2": [1, 0, 11, 10, 31, 30],
    })
    df = df.set_index('Sample Name')
    
    #Test cluster_n_times
    df_labels = cluster_n_times(df,2,3,cluster_type='agglom')
    assert np.array_equal(df_labels[2],[0,0,0,0,1,1])
    assert np.array_equal(df_labels[3],[2,2,0,0,1,1])
    assert np.array_equal(df_labels.index,df.index)
    df_labels = cluster_n_times(df.to_numpy(),2,3,cluster_type='agglom')
    assert np.array_equal(df_labels[2],[0,0,0,0,1,1])
    assert np.array_equal(df_labels[3],[2,2,0,0,1,1])
    
    df_labels = cluster_n_times(df,2,3,cluster_type='kmeans')
    assert df_labels[2]['Sample 0'] == df_labels[2]['Sample 1'] 
    assert df_labels[2]['Sample 0'] != df_labels[2]['Sample 5'] 
    assert df_labels[3]['Sample 0'] == df_labels[3]['Sample 1'] 
    assert df_labels[3]['Sample 0'] != df_labels[3]['Sample 5'] 
    
    
    df_labels = cluster_n_times(df.to_numpy(),2,3,cluster_type='kmedoids')
    assert np.array_equal(df_labels[2],[1,1,1,1,0,0])
    assert np.array_equal(df_labels[3],[2,2,1,1,0,0])
    
    
    
    
    #Test sklearn function in cluster_n_times_fn
    arg_dict = {
        "linkage": "ward"
    }

    df_labels = cluster_n_times_fn(df,2,3,arg_dict=arg_dict,sklearn_clust_fn=AgglomerativeClustering)
    assert np.array_equal(df_labels[2],[0,0,0,0,1,1])
    assert np.array_equal(df_labels[3],[2,2,0,0,1,1])
    assert np.array_equal(df_labels.index,df.index)
    
    #Test scipy function in cluster_n_times_fn
    arg_dict = {
        "criterion": "maxclust",
        "metric" : "euclidean",
        "method" : "ward"
    }

    cluster_n_times_fn(df,2,3,arg_dict=arg_dict,scipy_clust_fn=fclusterdata)
    assert np.array_equal(df_labels[2],[0,0,0,0,1,1])
    assert np.array_equal(df_labels[3],[2,2,0,0,1,1])
    assert np.array_equal(df_labels.index,df.index)
    
    
    
def test_cluster_nfrac_above_avg():
    data = [1,1,1,1,1,1,1,1,1,-50]
    labels = [0,0,0,0,0,1,1,1,1,1]
    df_nfrac = cluster_nfrac_above_avg(data,labels)
    assert df_nfrac[0] == 1.0
    assert df_nfrac[1] == 0.8
    
    
def test_cluster_top_percentiles():
    #Pretend clusters. high0 is high for cluster 0, and low0 is low for cluster0. Mid is in the middle
    arrays = [['high0', 'low0','mid'], ['1.5', '8.9','2.5']]
    index = pd.MultiIndex.from_arrays(arrays, names=('compound', 'RT'))
    data_high0 = [10,10,12,1,2]
    data_low0 = [1,0,2,14,15]
    data_mid = [5,5,5,5,5]
    cluster_labels = [0,0,0,1,1]
    
    df = pd.DataFrame(np.array([data_high0,data_low0,data_mid]).T,columns = index)
    df_top = cluster_top_percentiles(df,cluster_labels,2,highest=True)
    df_bottom = cluster_top_percentiles(df,cluster_labels,2,highest=False)
    
    assert df_top.shape == (2, 4)    
    
    assert np.array_equal(df_top[0,'(Formula/RT)'] , ['(high0, 1.5)','(mid, 2.5)'])
    assert np.array_equal(df_top[1,'(Formula/RT)'] , ['(low0, 8.9)','(mid, 2.5)'])
    
    assert np.array_equal(df_bottom[1,'(Formula/RT)'] , ['(high0, 1.5)','(mid, 2.5)'])
    assert np.array_equal(df_bottom[0,'(Formula/RT)'] , ['(low0, 8.9)','(mid, 2.5)'])
    
    
    
    
    
    
