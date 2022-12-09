# -*- coding: utf-8 -*-
from plotting.cmap_EOS11 import cmap_EOS11
from plotting.compare_cluster_metrics import compare_cluster_metrics_fn
from plotting.beijingdelhi import plot_cluster_heatmap_BeijingDelhi
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fclusterdata
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime


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
    

def test_plot_cluster_heatmap_BeijingDelhi():
    time_start = pd.concat([pd.date_range("2016-11-11", periods=20, freq="6h").to_series(),
                          pd.date_range("2017-05-20", periods=20, freq="6h").to_series(),
                          pd.date_range("2018-05-30", periods=20, freq="6h").to_series(),
                          pd.date_range("2018-10-12", periods=20, freq="6h").to_series()])
    time_mid = time_start + datetime.timedelta(hours=3)
    time_end = time_start + datetime.timedelta(hours=6)
    df_times = pd.DataFrame()
    df_times['date_start'] = time_start
    df_times['date_mid'] = time_mid
    df_times['date_end'] = time_end
    df_times = df_times.set_index(time_mid)
    
    #Just some labels
    labels = np.round(np.sin(np.linspace(0,50,len(df_times)))**2 * 5)
        
    plot_cluster_heatmap_BeijingDelhi(labels,df_times,'suptitle','ylabel')