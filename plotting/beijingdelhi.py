# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator, IndexLocator
import numpy as np
import pandas as pd
import pdb

from plotting.cmap_EOS11 import cmap_EOS11
from functions.delhi_beijing_datetime_cat import delhi_beijing_datetime_cat


def plot_all_cluster_tseries_BeijingDelhi(df_cluster_labels_mtx,ds_dataset_cat, title_prefix='',title_suffix=''):
    num_plots_to_make = df_cluster_labels_mtx.shape[1]
    for num_clusters in df_cluster_labels_mtx.columns:
        c = df_cluster_labels_mtx[num_clusters]
        title = title_prefix + str(num_clusters) + ' clusters' + title_suffix
        plot_tseries_BeijingDelhi(c,ds_dataset_cat,title,'Cluster index',integer_labels=True)
        
#%% Plot the time series divided into 4 projects
#c is the time series of cluster index
#ds_dataset_cat is the categorical data series of which dataset there is
#suptitle is the title to go at the top of the plot
def plot_tseries_BeijingDelhi(c,ds_dataset_cat,suptitle,ylabel,integer_labels=False):
    fig,ax = plt.subplots(2,2,figsize=(9,9))
    ax=ax.ravel()
    ax0=ax[0]
    ax0.plot(ds_dataset_cat.index,c)
    ax0.set_xlim(ds_dataset_cat[ds_dataset_cat == "Beijing_winter"].index.min(),ds_dataset_cat[ds_dataset_cat == "Beijing_winter"].index.max())
    ax0.set_title('Beijing winter')
    ax0.set_ylabel(ylabel)

    ax1=ax[1]
    ax1.plot(ds_dataset_cat.index,c)
    ax1.set_xlim(ds_dataset_cat[ds_dataset_cat == "Beijing_summer"].index.min(),ds_dataset_cat[ds_dataset_cat == "Beijing_summer"].index.max())
    ax1.set_title('Beijing summer')
    ax1.set_ylabel(ylabel)

    ax2=ax[2]
    ax2.plot(ds_dataset_cat.index,c)
    ax2.set_xlim(ds_dataset_cat[ds_dataset_cat == "Delhi_summer"].index.min(),ds_dataset_cat[ds_dataset_cat == "Delhi_summer"].index.max())
    ax2.set_title('Delhi summer')
    ax2.set_ylabel(ylabel)

    ax3=ax[3]
    ax3.plot(ds_dataset_cat.index,c)
    ax3.set_xlim(ds_dataset_cat[ds_dataset_cat == "Delhi_autumn"].index.min(),ds_dataset_cat[ds_dataset_cat == "Delhi_autumn"].index.max())
    ax3.set_title('Delhi autumn')
    ax3.set_ylabel(ylabel)

    #x ticks
    myFmt = mdates.DateFormatter('%d/%m')
    ax0.xaxis.set_major_formatter(myFmt)
    ax1.xaxis.set_major_formatter(myFmt)
    ax2.xaxis.set_major_formatter(myFmt)
    ax3.xaxis.set_major_formatter(myFmt)
    ax0.tick_params(axis='x', labelrotation=45)
    ax1.tick_params(axis='x', labelrotation=45)
    ax2.tick_params(axis='x', labelrotation=45)
    ax3.tick_params(axis='x', labelrotation=45)
    
    #y ticks
    if(integer_labels == True):
        ax0.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax3.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle(suptitle)
    fig.subplots_adjust(top=0.88)
    fig.tight_layout()
    plt.show()
    
    
    
    

def plot_cluster_heatmap_BeijingDelhi(labels,df_times,suptitle,ylabel):
    """
    Plot a time series cluster heatmap, divided into 4 projects from Beijing and Delhi

    Parameters
    ----------
    labels : array of int
        Time series of cluster index
    df_times : pandas dataframe
        Needs to have columns 'date_start', 'date_mid', and 'date_end' of type time, and same num rows as len(c)
    suptitle : string
        Suptitle of the plot.
    ylabel : string
        Y label of the plot.

    Returns
    -------
    None.

    """
    
    fig,ax = plt.subplots(2,2,figsize=(12,9))
    ax=ax.ravel()
    
    myFmt = mdates.DateFormatter('%d/%m/%y')
    
    #c_orig = c
    
    #Not sure why this is needed to copy but it was being weird
    all_times = df_times.to_numpy().copy()
    
    #Now need to scan through and cater for when a filter goes over midnight
    #Whenever a filter goes over midnight, change it so you have one row that ends and midnight and a 
    #new one that starts at midnight
    idx = 0
    while idx < len(labels):
        #pdb.set_trace()
        time_start = all_times[idx][0]
        time_end = all_times[idx][2]
        
        #If filter goes over midnight
        if(time_end.astype('datetime64[D]') > time_start.astype('datetime64[D]')):
            midnight = time_end.astype('datetime64[D]')
            
            #Old row goes up to midnight
            new_time_mid = (midnight - time_start )/2 + time_start
            all_times[idx] = [time_start,new_time_mid,midnight]
            
            #Insert new row starts from midnight
            labels = np.insert(labels,idx+1,labels[idx])
            new_time_mid2 = (time_end - midnight)/2 + midnight
            all_times = np.insert(all_times,idx+1,[midnight,new_time_mid2,time_end],axis=0) 
            idx += 1
        idx += 1
            
       
    df_times = pd.DataFrame(all_times,columns=df_times.columns)
    df_times.set_index('date_mid',inplace=True)
    
    #Onehot map of the cluster labels
    df_labels_onehot = pd.get_dummies(labels).set_index(df_times.index)
    time_delta_frac = (df_times['date_end'] - df_times['date_start']) / np.timedelta64(1, 'D')
    df_labels_frachot = df_labels_onehot.multiply( time_delta_frac,axis=0)
    
    ds_dataset_cat = delhi_beijing_datetime_cat(df_labels_frachot.index)
    
    
    #The time for each culster per day, as a fraction of the total time sampled on that day
    df_labels_frachot_perday = df_labels_frachot.resample('D').sum().divide(df_labels_frachot.resample('D').sum().sum(axis=1),axis=0)
    
    
    
    
    for idx in range(4):
        #pdb.set_trace()
        cat = ds_dataset_cat.unique()[idx]
        df_labels_onehot_thiscat = df_labels_onehot.loc[ds_dataset_cat == cat]
        #ds_dataset_cat_thiscat = ds_dataset_cat[ds_dataset_cat == cat]
        
        #The time for each culster per day, as a fraction of the total time sampled on that day
        df_labels_frachot_perday = df_labels_onehot_thiscat.resample('D').sum().divide(df_labels_onehot_thiscat.resample('D').sum().sum(axis=1),axis=0)
        
    
        xbins = pd.date_range(df_labels_frachot_perday.index.floor("1D").min(),(df_labels_frachot_perday.index.ceil("1D").max()+np.timedelta64(1,'D')),freq="1D").to_numpy()
        ybins = np.unique(labels)
        ybins= np.append(ybins,ybins.max() + 1) -0.5
        
        im = ax[idx].pcolormesh(xbins,ybins,df_labels_frachot_perday.T)
        
        ax[idx].set_yticks(np.unique(labels))

        #Format dates on x labels
        ax[idx].xaxis.set_major_formatter(myFmt)
        ax[idx].tick_params(axis='x', labelrotation=45)

        ax[idx].set_ylabel(ylabel)
        ax[idx].set_title(cat)
        
        ax[idx].xaxis_date()
        
        #Set x tick locations
        myLocator = IndexLocator( base = 7, offset = 0)
        ax[idx].xaxis.set_major_locator(myLocator)
        myLocator2 = IndexLocator( base = 1, offset = 0)
        ax[idx].xaxis.set_minor_locator(myLocator2)
        
        #Add grid
        ax[idx].grid(axis='x',linestyle='--',alpha=0.6)
        
    

    fig.suptitle(suptitle)
    fig.subplots_adjust(top=0.88)
    fig.tight_layout()
    
    #Add colourbar AFTER tight layout
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.90, 0.17, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Fraction')
    #plt.show()
    
    
def plot_n_cluster_heatmaps_BeijingDelhi(df_labels,df_times,suptitle_prefix,ylabel):
    """
    Run plot_cluster_heatmap_BeijingDelhi many times for different cluster labels

    Parameters
    ----------
    df_labels : pandas dataframe of int
        Dataframe of cluster labels, of the type produced by cluster_n_times. Index is time and columns are different n_clusters
    df_times : pandas dataframe
    Needs to have columns 'date_start', 'date_mid', and 'date_end' of type time, and same num rows as len(c)
    suptitle_prefix : string
        Suptitle prefix of the plot.
    ylabel : string
        Y label of the plot.

    Returns
    -------
    None.

    """
    
    for n_clusters in df_labels.columns:
        suptitle = f"{suptitle_prefix} {n_clusters} clusters"
        #pdb.set_trace()
        plot_cluster_heatmap_BeijingDelhi(df_labels[n_clusters].to_numpy(),df_times,suptitle,ylabel)
    
    
    
#%% Plot the time series divided into 4 projects
#c is the time series of cluster index
#ds_dataset_cat is the categorical data series of which dataset there is
#suptitle is the title to go at the top of the plot
def plot_AMS_tseries_BeijingDelhi(df_AQ_all,ds_dataset_cat,df_all_data,suptitle,ylabel):
    fig,ax = plt.subplots(2,2,figsize=(9,9))
    ax=ax.ravel()
    ax0=ax[0]
    ax0.plot(df_all_data.index,df_AQ_all['AMS_NO3'],c='b')
    ax0.plot(df_all_data.index,df_AQ_all['AMS_SO4'],c='r')
    ax0.plot(df_all_data.index,df_AQ_all['AMS_NH4'],c='orange')
    ax0.plot(df_all_data.index,df_AQ_all['AMS_Chl'],c='pink')
    ax0.plot(df_all_data.index,df_AQ_all['AMS_Org'],c='g')
    ax0.set_xlim(ds_dataset_cat[ds_dataset_cat == "Beijing_winter"].index.min(),ds_dataset_cat[ds_dataset_cat == "Beijing_winter"].index.max())
    ax0.set_ylim(0,110)
    ax0.set_title('Beijing winter')
    ax0.set_ylabel(ylabel)

    ax1=ax[1]
    ax1.plot(df_all_data.index,df_AQ_all['AMS_NO3'],c='b')
    ax1.plot(df_all_data.index,df_AQ_all['AMS_SO4'],c='r')
    ax1.plot(df_all_data.index,df_AQ_all['AMS_NH4'],c='orange')
    ax1.plot(df_all_data.index,df_AQ_all['AMS_Chl'],c='pink')
    ax1.plot(df_all_data.index,df_AQ_all['AMS_Org'],c='g')
    ax1.set_xlim(ds_dataset_cat[ds_dataset_cat == "Beijing_summer"].index.min(),ds_dataset_cat[ds_dataset_cat == "Beijing_summer"].index.max())
    ax1.set_ylim(0,35)
    ax1.set_title('Beijing summer')
    ax1.set_ylabel(ylabel)

    ax2=ax[2]
    ax2.plot(df_all_data.index,df_AQ_all['AMS_NO3'],c='b')
    ax2.plot(df_all_data.index,df_AQ_all['AMS_SO4'],c='r')
    ax2.plot(df_all_data.index,df_AQ_all['AMS_NH4'],c='orange')
    ax2.plot(df_all_data.index,df_AQ_all['AMS_Chl'],c='pink')
    ax2.plot(df_all_data.index,df_AQ_all['AMS_Org'],c='g')
    ax2.set_xlim(ds_dataset_cat[ds_dataset_cat == "Delhi_summer"].index.min(),ds_dataset_cat[ds_dataset_cat == "Delhi_summer"].index.max())
    ax2.set_ylim(0,75)
    ax2.set_title('Delhi summer')
    ax2.set_ylabel(ylabel)

    ax3=ax[3]
    ax3.plot(df_all_data.index,df_AQ_all['AMS_NO3'],c='b')
    ax3.plot(df_all_data.index,df_AQ_all['AMS_SO4'],c='r')
    ax3.plot(df_all_data.index,df_AQ_all['AMS_NH4'],c='orange')
    ax3.plot(df_all_data.index,df_AQ_all['AMS_Chl'],c='pink')
    ax3.plot(df_all_data.index,df_AQ_all['AMS_Org'],c='g')
    ax3.set_xlim(ds_dataset_cat[ds_dataset_cat == "Delhi_autumn"].index.min(),ds_dataset_cat[ds_dataset_cat == "Delhi_autumn"].index.max())
    ax3.set_title('Delhi autumn')
    ax3.set_ylabel(ylabel)


    myFmt = mdates.DateFormatter('%d/%m')
    ax0.xaxis.set_major_formatter(myFmt)
    ax1.xaxis.set_major_formatter(myFmt)
    ax2.xaxis.set_major_formatter(myFmt)
    ax3.xaxis.set_major_formatter(myFmt)
    ax0.tick_params(axis='x', labelrotation=45)
    ax1.tick_params(axis='x', labelrotation=45)
    ax2.tick_params(axis='x', labelrotation=45)
    ax3.tick_params(axis='x', labelrotation=45)

    fig.suptitle(suptitle)
    fig.subplots_adjust(top=0.88)
    fig.tight_layout()
    plt.show()