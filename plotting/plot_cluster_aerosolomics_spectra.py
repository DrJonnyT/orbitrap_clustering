# -*- coding: utf-8 -*-
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import percentileofscore
import pdb

def plot_cluster_aerosolomics_spectra(cluster_labels,df_aero_concs,**kwargs):
    """
    Box plots of aerosolomics sources, averaged for each cluster label. Plotted like a spectrum

    Parameters
    ----------
    cluster_labels : array of integers
        Cluster labels
    df_aero_concs : dataframe
        Concentrations of species from the different Aerosolomics sources

    kwargs : keyword arguments (optional)
        suptitle : string
            Plot suptitle
        avg : string, default: 'mean'
            If 'mean', plot the mean per cluster. If 'pct' plot the percentile score. If 'median' plot the median per cluster.
        ygrid : bool, default : True
            Add a y grid
        offset_zero : bool, default : False
            Offset the zero so e.g. in a yscale of 0 - 100, values will go up or down from 50
            
    Returns
    -------
    None.

    """
    sns.set_context("talk", font_scale=1)
    unique_labels = np.unique(cluster_labels)
    num_clust = len(unique_labels)
    
    if 'avg' in kwargs:
        avg = kwargs.get('avg')
    else:
        avg = 'mean'
        
    if 'ygrid' in kwargs:
        ygrid = kwargs.get('ygrid')
    else:
        ygrid = True
        
    if 'offset_zero' in kwargs:
        offset_zero = kwargs.get('offset_zero')
    else:
        offset_zero = False
        
    
    
    
    if num_clust <=4:
        fig,ax = plt.subplots(1,num_clust,figsize=(12,4))
    elif num_clust <= 8:
        fig,ax = plt.subplots(2,4,figsize=(14,8))
    ax=ax.ravel()
    
    
    df_aero_gb = df_aero_concs.groupby(cluster_labels)
    
    x_colors = ['tab:green','tab:green','tab:green','tab:green','k','tab:green','k','k','k']
    
    
    for cluster in unique_labels:
        
        if avg == 'mean':
            ylabel = 'µg$\,$m$^{-3}$'
            ds_toplot = df_aero_gb.mean().loc[cluster]
        elif avg == 'median':
            ylabel = 'µg$\,$m$^{-3}$'
            ds_toplot = df_aero_gb.median().loc[cluster]
        elif avg == 'pct':
            ylabel = 'Percentile'
            
            #Extract the percentile of the median of each molecule, for this cluster
            data_thisclust = df_aero_concs.loc[cluster_labels==cluster]
            ds_toplot = pd.Series([percentileofscore(df_aero_concs[source],data_thisclust[source].median()) for source in df_aero_concs.columns],index=df_aero_concs.columns, dtype='float')
            
            if(offset_zero):
                #Make it so the plotting starts halfway up the graph
                ds_toplot = ds_toplot - 50
                ax[cluster].set_ylim(-50,50)
                ax[cluster].set_yticks([-50,-25,0,25,50], [0,25,50,75,100])
            

            
        #Plot the data
        ds_toplot.plot.bar(ax=ax[cluster],color=x_colors)
        
        if(ygrid):
            ax[cluster].grid(axis='y',linestyle='--',alpha=0.5)
        
        ax[cluster].set_ylabel(ylabel)
        
    
    #Remove empty subplots
    for blank in range(unique_labels.max()+1,len(ax)):
        ax[blank].set_axis_off()

    if 'suptitle' in kwargs:
        plt.suptitle(kwargs.get('suptitle'))

    plt.tight_layout()
    sns.reset_orig()