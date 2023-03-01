"""
It's fairly tricky to test plotting functions with pytest, so really these only check if the plotting functions complete without error.
"""


from plotting.cmap_EOS11 import cmap_EOS11
from plotting.compare_cluster_metrics import compare_cluster_metrics_fn
from plotting.beijingdelhi import plot_cluster_heatmap_BeijingDelhi
from plotting.plot_windrose_percluster import plot_windrose_percluster

import pdb
from matplotlib.colors import LinearSegmentedColormap

import pandas as pd
import numpy as np
import datetime


def test_cmap_EOS11():
    cmap = cmap_EOS11()
    assert type(cmap) == LinearSegmentedColormap
    assert cmap.N == 11
    assert cmap_EOS11(50).N == 50
    
    

# def test_compare_cluster_metrics_fn():
#     df = pd.DataFrame({
#         "Sample Name": ["Sample "+str(i) for i in range(6)],
#         "Feature1": [6, 5, 1, 2, 3, 4],
#         "Feature2": [5, 6, 2, 1, 4, 3],
#     })
    
#     arg_dict = {
#         "linkage": "ward"
#     }

#     compare_cluster_metrics_fn(df[['Feature1', 'Feature2']],2,5,arg_dict=arg_dict,sklearn_clust_fn=AgglomerativeClustering)
    
    
#     arg_dict = {
#         "criterion": "maxclust",
#         "metric" : "euclidean",
#         "method" : "ward"
#     }

#     compare_cluster_metrics_fn(df[['Feature1', 'Feature2']],2,5,arg_dict=arg_dict,scipy_clust_fn=fclusterdata)

    
#     #plt.close('all')
    

def test_plot_cluster_heatmap_BeijingDelhi():
    time_start = pd.concat([pd.date_range("2016-11-11", periods=20, freq="7h").to_series(),
                          pd.date_range("2017-05-20", periods=20, freq="7h").to_series(),
                          pd.date_range("2018-05-30", periods=20, freq="7h").to_series(),
                          pd.date_range("2018-10-12", periods=20, freq="7h").to_series()])
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
    
    
    
def test_plot_windrose_percluster():
    time_index = pd.concat([pd.date_range("2016-11-11", periods=20, freq="7h").to_series(),
                          pd.date_range("2017-05-20", periods=20, freq="7h").to_series(),
                          pd.date_range("2018-05-30", periods=20, freq="7h").to_series(),
                          pd.date_range("2018-10-12", periods=20, freq="7h").to_series()])
    df_merge = pd.DataFrame(index=time_index)
    df_merge['ws_ms'] = np.random.normal(3,1,80).clip(min=0)
    df_merge['wd_deg'] = np.random.uniform(0,360,80)
    cluster_labels = np.random.randint(0,3,80)
    plot_windrose_percluster(df_merge,cluster_labels,quiet=True)
    
    