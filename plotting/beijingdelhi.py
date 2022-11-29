# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator



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