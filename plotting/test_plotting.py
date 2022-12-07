# -*- coding: utf-8 -*-
from plotting.cmap_EOS11 import cmap_EOS11
from plotting.compare_cluster_metrics import compare_cluster_metrics_fn
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fclusterdata
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import pandas as pd



def test_cmap_EOS11():
    cmap = cmap_EOS11()
    assert type(cmap) == LinearSegmentedColormap
    assert cmap.N == 11
    assert cmap_EOS11(50).N == 50
    
    
    
def test_compare_cluster_metrics_fn():
    df = pd.DataFrame({
        "Sample Name": ["Sample "+str(i) for i in range(6)],
        "Feature1": [6, 5, 1, 2, 3, 4],
        "Feature2": [5, 6, 2, 1, 4, 3],
    })
    
    arg_dict = {
        "linkage": "ward"
    }

    compare_cluster_metrics_fn(df[['Feature1', 'Feature2']],2,5,arg_dict=arg_dict,sklearn_clust_fn=AgglomerativeClustering)
    
    
    arg_dict = {
        "criterion": "maxclust",
        "metric" : "euclidean",
        "method" : "ward"
    }

    compare_cluster_metrics_fn(df[['Feature1', 'Feature2']],2,5,arg_dict=arg_dict,scipy_clust_fn=fclusterdata)

    
    #plt.close('all')
    
