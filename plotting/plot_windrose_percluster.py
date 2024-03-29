from windrose import WindroseAxes  #You must leave this in as it adds the windrose projection
import numpy as np
from seaborn import set_context, reset_orig
import matplotlib.pyplot as plt
import pandas as pd
import pdb

import matplotlib.cm as cm

from functions.delhi_beijing_time import delhi_beijing_datetime_cat

def plot_windrose_percluster(df_merge,cluster_labels,**kwargs): 
    """
    Plot 2 wind roses for each unique cluster from cluster_labels, one for Beijing and one for Delhi
    df_merge must have wind speed as 'ws_ms' and wind direction as 'wd_deg'
    
    Parameters
    ----------
    df_merge : Pandas DataFrame
        Merge AQ/met data with columns for wind speed as 'ws_ms' and wind direction as 'wd_deg'
    cluster_labels : Array
        Cluster labels with same length as df_merge    

    Returns
    -------
    variable
        Largest number of clusters for which (maxr > maxr_threshold) or neither (mincard < mincard_threshold).
    
    
    
    """
    
    if kwargs.get('quiet', False):
        pass
    else:
        print("Plotting wind roses. This may take a minute or two")
    
    if "binsize" in kwargs:
        binsize = kwargs.get("binsize")
    else:
        binsize = 1
    
    if "maxspeed" in kwargs:
        maxspeed = kwargs.get("maxspeed")
    else:
        maxspeed = df_merge['ws_ms'].max()
    
    unique_labels = np.unique(cluster_labels)
    num_clusters = len(unique_labels)
    
    
    dataset_cat = delhi_beijing_datetime_cat(df_merge.index)
    
    #Create figure
    set_context("talk", font_scale=0.8)  
    fig = plt.figure(figsize=(num_clusters*5,10))
    
    #Wind bins
    bins = np.arange(0, maxspeed, binsize)
    
    for cluster in unique_labels:
        df_wind = pd.DataFrame({"speed": df_merge['ws_ms'].loc[cluster_labels==cluster], "direction": df_merge['wd_deg'].loc[cluster_labels==cluster]})
        df_wind_beijing = df_wind.loc[pd.DataFrame([(dataset_cat == 'Beijing_winter'), (dataset_cat == 'Beijing_summer')]).any()]
        df_wind_delhi = df_wind.loc[pd.DataFrame([(dataset_cat == 'Delhi_summer'), (dataset_cat == 'Delhi_autumn')]).any()]
        
        #Plot Beijing
        ax = fig.add_subplot(2,num_clusters,cluster+1, projection="windrose")
        ax.set_title('Cluster ' + str(cluster),fontweight='bold')
        if (not df_wind_beijing.empty):
            ax.bar(df_wind_beijing['direction'], df_wind_beijing['speed'], bins=bins, cmap=cm.hot,normed=True)
        
        #Plot Delhi
        ax = fig.add_subplot(2,num_clusters,(num_clusters+cluster+1), projection="windrose")
        if (not df_wind_delhi.empty):
            ax.bar(df_wind_delhi['direction'], df_wind_delhi['speed'], bins=bins, cmap=cm.hot,normed=True)
        
    #Legend to last plot
    ax.legend(bbox_to_anchor=(1.9, 1.5),loc='upper right',title=r'Wind speed (ms$^{-1}$)')
    
    if 'suptitle' in kwargs:
        fig.suptitle(kwargs.get("suptitle"),fontweight='bold')
    
    #Add text to rows
    fig.text(0.1,0.9,"Beijing",fontweight='bold')
    fig.text(0.1,0.48,"Delhi",fontweight='bold')
    
    
    reset_orig()