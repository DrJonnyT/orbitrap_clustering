# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pdb

def plot_cluster_count_hists(*args,**kwargs):
    """
    Plot histograms of counts per cluster, for a list of df_cluster_counts_mtx

    Parameters
    ----------
    *args : list of dataframes
    list of dataframes of cluster counts. Index is num_clusters, columns are cluster index

    Returns
    -------
    None.

    """
    
    
    num_subplots = len(args)
    
    # #Find max yscale
    # maxy = 0
    # for df in args:
    #     maxy = np.maximum(maxy,df.max().max())
        
    
    #Get data from kwargs
    if 'titles' in kwargs:
        titles = kwargs.get('titles')
    if 'colors' in kwargs:
        colors = kwargs.get('colors')
    if 'sharex' in kwargs:
        sharex = kwargs.get('sharex')
    else:
        sharex = False
    
    
    #Example dataframe
    df_cluster_counts_mtx0 = args[0]
          
    
    #Loop through the possible numbers of clusters
    for n_clusters in df_cluster_counts_mtx0.index:
        fig,ax = plt.subplots(num_subplots,1,figsize=(2+n_clusters*0.5,num_subplots*4),sharex=sharex,sharey=True)
        ax = ax.ravel()
        
        
        if n_clusters == 70:
            return None
        
        #Loop through df inputs
        for subplot in range(num_subplots):
            #pdb.set_trace()
            #Pick out data and sort in order of largest first
            data = args[subplot].loc[n_clusters].sort_values(ascending=False)
            try:
                bars = ax[subplot].bar(data.index.astype('str'),data.values,color=colors[subplot])    
            except:
                bars = ax[subplot].bar(data.index.astype('str'),data.values)    
            
            #ax[subplot].set_ylim(bottom=0,top=maxy)
            
           
            
            ax[subplot].xaxis.set_major_locator(plticker.MultipleLocator(base=1))
            ax[subplot].set_ylabel('Counts')
            ax[subplot].bar_label(bars)
            
            
            
            try:
                ax[subplot].set_title(titles[subplot])
            except:
                pass
            
        ax[0].set_ylim(0, 1.05*ax[0].get_ylim()[1])
        
        fig.suptitle(f'{n_clusters} clusters')
        plt.tight_layout()
        plt.show()
        
    
    
    