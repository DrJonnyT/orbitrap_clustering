# -*- coding: utf-8 -*-
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import pandas as pd
from numpy import unique
import pdb



def bin_mol_data_for_plot(df_cluster_labels_mtx,df_data_moltypes):
    """
    Bin molecule type data by cluster labels

    Parameters
    ----------
    df_cluster_labels_mtx : pandas dataframe
        Output from clustering.cluster_n_times()
    df_data_moltypes : dataframe
        Index is time-like, columns are the different molecule types, normally CHO, CHON, CHOS, CHONS

    Returns
    -------
    df_mol_data_forplot : dataframe
        index is a dummy, time-like
        columns are the molecule types in df_data_moltypes.columns
        data are the mean for each molecule type per cluster, per num_clusters
        num_clusters and cluster_index (df_cluster_labels_mtx index and columns) are also columns that are in an easily-plottable format, rather than index and column headers

    """
    
    index = df_cluster_labels_mtx.columns
    columns = pd.Index(unique(df_cluster_labels_mtx),name="cluster_index")

    df_frac_percluster_CHO = pd.DataFrame(index=index,columns=columns)
    df_frac_percluster_CHOS = pd.DataFrame(index=index,columns=columns)
    df_frac_percluster_CHON = pd.DataFrame(index=index,columns=columns)
    df_frac_percluster_CHONS = pd.DataFrame(index=index,columns=columns)


    for n_clusters in index:
        cluster_labels = df_cluster_labels_mtx[n_clusters]
        for cluster in columns:
            #pdb.set_trace()
            try:
                moltypes_thiscluster =  df_data_moltypes.loc[cluster_labels == cluster]
            except:
                moltypes_thiscluster =  df_data_moltypes.reset_index(drop=True).loc[cluster_labels.reset_index(drop=True) == cluster]
    
            # df_frac_percluster_CHO[cluster][n_clusters] = num_frac_above_val(moltypes_thiscluster['CHO'],df_all_data_moltypes['CHO'].median())
            # df_frac_percluster_CHOS[cluster][n_clusters] = num_frac_above_val(moltypes_thiscluster['CHOS'],df_all_data_moltypes['CHOS'].median())
            # df_frac_percluster_CHON[cluster][n_clusters] = num_frac_above_val(moltypes_thiscluster['CHON'],df_all_data_moltypes['CHON'].median())
            # df_frac_percluster_CHONS[cluster][n_clusters] = num_frac_above_val(moltypes_thiscluster['CHONS'],df_all_data_moltypes['CHONS'].median())
            
            df_frac_percluster_CHO[cluster][n_clusters] = (moltypes_thiscluster['CHO'].mean() )#- df_all_data_moltypes['CHO'].mean()) / df_all_data_moltypes['CHO'].std() 
            df_frac_percluster_CHOS[cluster][n_clusters] = (moltypes_thiscluster['CHOS'].mean() )#- df_all_data_moltypes['CHOS'].mean()) / df_all_data_moltypes['CHOS'].std() 
            df_frac_percluster_CHON[cluster][n_clusters] = (moltypes_thiscluster['CHON'].mean() )#- df_all_data_moltypes['CHON'].mean()) / df_all_data_moltypes['CHON'].std() 
            df_frac_percluster_CHONS[cluster][n_clusters] = (moltypes_thiscluster['CHONS'].mean() )#- df_all_data_moltypes['CHONS'].mean()) / df_all_data_moltypes['CHONS'].std() 
            
    #pdb.set_trace()
    df_mol_data_forplot = pd.melt(df_frac_percluster_CHO.reset_index(),id_vars='num_clusters')
    df_mol_data_forplot = df_mol_data_forplot.rename(columns={'value': 'CHO'})
    df_mol_data_forplot['CHOS'] = pd.melt(df_frac_percluster_CHOS.reset_index(),id_vars='num_clusters')['value']
    df_mol_data_forplot['CHON'] = pd.melt(df_frac_percluster_CHON.reset_index(),id_vars='num_clusters')['value']
    df_mol_data_forplot['CHONS'] = pd.melt(df_frac_percluster_CHONS.reset_index(),id_vars='num_clusters')['value']
    
    return df_mol_data_forplot



def plot_binned_mol_data(*args,**kwargs):
    """
    Plot one x per cluster, with num_clusters on the x axis and CHO, CHOS, CHON and CHONS on y axes, for a number of dataframes

    Parameters
    ----------
    *args : list of dataframes
    list of dataframes, each of which is output from bin_mol_data_for_plot. Index is time-like(?), columns are num_clusters, cluster_index, CHO, CHOS, CHON, CHONS

    **kwargs : list of keyword arguments
        titles : List of subplot titles
        ylabels : List of 4 y labels
        suptitle : suptitle for final plot
        colors : list of color strings for each dataframe in *args
        sharex : do the subplot columns share x axes? Default : True
        sharey : do the subfig rows share y axes? Default : True
        vlines: list of vertical lines to go on the plots
        
    Returns
    -------
    None.

    """
        
    sns.set_context("talk", font_scale=0.7)
    
    
    num_rows = len(args)
    num_subplots = num_rows * 4
    
    # #Find max yscale
    # maxy = 0
    # for df in args:
    #     maxy = np.maximum(maxy,df.max().max())
        
    
    #Get data from kwargs
    if 'titles' in kwargs:
        titles = kwargs.get('titles')
    else:
        titles = [""] * num_rows
        
    if 'ylabels' in kwargs:
        ylabels = kwargs.get('ylabels')
    else:
        ylabels = [""] * num_rows
        
    if 'vlines' in kwargs:
        vlines = kwargs.get('vlines')
        
    if 'colors' in kwargs:
        colors = kwargs.get('colors')
    if 'suptitle' in kwargs:
        suptitle = kwargs.get('suptitle')
    else:
        suptitle = ""
    if 'sharex' in kwargs:
        sharex = kwargs.get('sharex')
    else:
        sharex = True
    if 'sharey' in kwargs:
        sharey = kwargs.get('sharey')
    else:
        sharey = True
        
    if sharey:
        #Calculate common y limits for each row
        CHO_minmax = [0.95*min([df['CHO'].min() for df in args]), 1.02*max([df['CHO'].max() for df in args])]
        CHON_minmax = [0.95*min([df['CHON'].min() for df in args]), 1.02*max([df['CHON'].max() for df in args])]
        CHOS_minmax = [0.95*min([df['CHOS'].min() for df in args]), 1.02*max([df['CHOS'].max() for df in args])]
        CHONS_minmax = [0.95*min([df['CHONS'].min() for df in args]), 1.02*max([df['CHONS'].max() for df in args])]
    
    
    
    
    
    fig = plt.figure(constrained_layout=True,figsize=(18,num_rows*4))
    fig.suptitle(suptitle)
    
    # create num_rows x 1 subfigs
    subfigs = fig.subfigures(nrows=num_rows, ncols=1)

    for row, subfig in enumerate(subfigs):
        
        #Get data for each row
        df_mol_data_forplot = args[row]
        
        subfig.suptitle(titles[row])
        
        try:
            color = colors[row]
        except:
            color = 'tab:blue'

        # create 1x4 subplots per subfig and plot the data for each molecule
        axs = subfig.subplots(nrows=1, ncols=4, sharex=sharex)
        axs[0].scatter(df_mol_data_forplot['num_clusters'],df_mol_data_forplot['CHO'],marker='x',s=25, c=color)
        #axs[0].set_ylabel(ylabels[0])
        axs[1].scatter(df_mol_data_forplot['num_clusters'],df_mol_data_forplot['CHON'],marker='x',s=25, c=color)
        #axs[1].set_ylabel(ylabels[1])
        axs[2].scatter(df_mol_data_forplot['num_clusters'],df_mol_data_forplot['CHOS'],marker='x',s=25, c=color)
        #axs[2].set_ylabel(y)
        axs[3].scatter(df_mol_data_forplot['num_clusters'],df_mol_data_forplot['CHONS'],marker='x',s=25, c=color)
        #axs[3].set_ylabel('CHONS')
        axs[0].xaxis.set_major_locator(plticker.MultipleLocator(2))
        axs[0].xaxis.set_minor_locator(plticker.MultipleLocator(1))
        
        for ax, ylabel in zip(axs, ylabels):
            ax.grid(axis='y',alpha=0.5)
            ax.set_ylabel(ylabel)
        
        if sharey:
            axs[0].set_ylim(CHO_minmax)
            axs[1].set_ylim(CHON_minmax)
            axs[2].set_ylim(CHOS_minmax)
            axs[3].set_ylim(CHONS_minmax)
            
        if 'vlines' in kwargs:
            for ax in axs:
                ax.axvline(vlines[row],0,1,alpha=0.25,linestyle='--')
        
        
            

    
    sns.reset_orig()