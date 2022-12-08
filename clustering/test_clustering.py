# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fclusterdata
from clustering import molecule_type_pos_frac
from clustering.molecule_type_math import molecule_type_pos_frac_clusters
from clustering.cluster_n_times import cluster_n_times, cluster_n_times_fn



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
    