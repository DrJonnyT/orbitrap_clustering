# -*- coding: utf-8 -*-
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import string
from functions.delhi_beijing_time import delhi_beijing_datetime_cat

def plot_orbitrap_ams_aqmet(cluster_labels,df_orbitrap_moltypes,df_merge,**kwargs):
    """
    Plot a large matrix with box and whiskers of orbitrap molecule types, AMS data, and AQ/met data, segregated by cluster label

    Parameters
    ----------
    cluster_labels : array
        Cluster labels
    df_orbitrap_moltypes : DataFrame
        Orbitrap data segregated by molecule types CHO, CHON, CHOS, CHONS. Index is time
    df_merge : DataFrame
        Merge air quality / met data. Index is time
    **kwargs : 
        Additional keyword arguments
    suptitle : suptitle for the figure

    Returns
    -------
    None.

    """
    sns.set_context("paper", font_scale=1)
    
    ylabels = ["CHO (µg m$^{-3}$)","CHON (µg m$^{-3}$)","CHOS (µg m$^{-3}$)","CHONS (µg m$^{-3}$)",
               "AMS OA (µg m$^{-3}$)", "AMS NO$_3^-$ (µg m$^{-3}$)", "AMS SO$_4^{2-}$ (µg m$^{-3}$)", "AMS PMF fraction",
               "AMS FFOA (µg m$^{-3}$)","AMS COA (µg m$^{-3}$)","AMS BBOA (µg m$^{-3}$)","AMS OOA (µg m$^{-3}$)",
               "CO (ppbv)", "NO$_2$ (ppbv)", "O$_3$ (ppbv)", "SO$_2$ (ppbv)",
               "Precip (mm)", "RH(%)","Wind speed (m s$^{-1}$",""
               ]
    
    
    whis=[5,95]
    
    
    #Make dataframes of the mean of each pmf factor for each cluster, as a fraction so all add up to 1
    df_merge_AMS_PMF_frac = df_merge[['AMS_FFOA','AMS_COA','AMS_BBOA','AMS_OOA']].groupby(cluster_labels).mean()
    df_merge_AMS_PMF_frac = df_merge_AMS_PMF_frac.div(df_merge_AMS_PMF_frac.sum(axis=1,skipna=False),axis=0)
    
        
    fig,ax = plt.subplots(5,4,figsize=(8,10),dpi=300)
    ax = ax.ravel()
    
    #Orbitrap data
    sns.boxplot(ax=ax[0], x=cluster_labels, y="CHO", data=df_orbitrap_moltypes,showfliers=False,color='tab:green',whis=whis)
    sns.boxplot(ax=ax[1], x=cluster_labels, y="CHON", data=df_orbitrap_moltypes,showfliers=False,color='tab:blue',whis=whis)
    sns.boxplot(ax=ax[2], x=cluster_labels, y="CHOS", data=df_orbitrap_moltypes,showfliers=False,color='tab:red',whis=whis)
    sns.boxplot(ax=ax[3], x=cluster_labels, y="CHONS", data=df_orbitrap_moltypes,showfliers=False,color='tab:gray',whis=whis)
    
    #AMS data
    sns.boxplot(ax=ax[4], x=cluster_labels, y="AMS_Org", data=df_merge,showfliers=False,color='tab:green')
    sns.boxplot(ax=ax[5], x=cluster_labels, y="AMS_NO3", data=df_merge,showfliers=False,color='tab:blue')
    sns.boxplot(ax=ax[6], x=cluster_labels, y="AMS_SO4", data=df_merge,showfliers=False,color='tab:red')
    df_merge_AMS_PMF_frac.plot(ax=ax[7],kind='bar', stacked=True,
                                           color=['dimgray', 'silver', 'tab:brown','mediumpurple'],
                                           legend=False,width=0.9)
    ax[7].set_ylabel('PMF fraction')
    sns.boxplot(ax=ax[8], x=cluster_labels, y="AMS_FFOA", data=df_merge,showfliers=False,color='dimgray')
    sns.boxplot(ax=ax[9], x=cluster_labels, y="AMS_COA", data=df_merge,showfliers=False,color='silver')
    sns.boxplot(ax=ax[10], x=cluster_labels, y="AMS_BBOA", data=df_merge,showfliers=False,color='tab:brown')
    sns.boxplot(ax=ax[11], x=cluster_labels, y="AMS_OOA", data=df_merge,showfliers=False,color='mediumpurple')
    
    
    #AQ and met data
    sns.boxplot(ax=ax[12], x=cluster_labels, y="co_ppbv", data=df_merge,showfliers=False,color='tab:gray',whis=whis)
    sns.boxplot(ax=ax[13], x=cluster_labels, y="no2_ppbv", data=df_merge,showfliers=False,color='tab:blue',whis=whis)
    sns.boxplot(ax=ax[14], x=cluster_labels, y="o3_ppbv", data=df_merge,showfliers=False,color='tab:green',whis=whis)
    sns.boxplot(ax=ax[15], x=cluster_labels, y="so2_ppbv", data=df_merge,showfliers=False,color='tab:red',whis=whis)
    sns.boxplot(ax=ax[16], x=cluster_labels, y="HYSPLIT_precip", data=df_merge,showfliers=False,color='tab:olive',whis=whis)
    sns.boxplot(ax=ax[17], x=cluster_labels, y="RH", data=df_merge,showfliers=False,color='tab:cyan',whis=whis)
    sns.boxplot(ax=ax[18], x=cluster_labels, y="ws_ms", data=df_merge,showfliers=False,color='tab:purple',whis=whis)
    ax[18].set_ylim(bottom=0)
    
    #Set axis labels
    [axis.set_xlabel('')  for axis in ax]
    [axis.set_ylabel(ylab,labelpad=0)  for axis, ylab in zip(ax,ylabels)]
    
    #Add letters in boxes for each subfigure
    trans = ScaledTranslation(2/72, -5/72, fig.dpi_scale_trans)
    
    for axis, letter in zip(ax,string.ascii_lowercase):
        axis.text(0.0, 1.0, ('(' + letter + ')'), transform=axis.transAxes + trans,
                fontsize='medium', verticalalignment='top', fontfamily='serif',
                bbox=dict(facecolor='1.0', edgecolor='none', pad=1.0))

    
    if 'suptitle' in kwargs:    
        plt.suptitle(kwargs.get('suptitle'))
    plt.tight_layout()
    
    sns.reset_orig()
    
    
    
    
def plot_orbitrap_ams_aqmet_time(cluster_labels,df_data,df_merge,ds_day_frac,**kwargs):
    """
    Plot a large matrix with box and whiskers of orbitrap molecule types, AMS data, and AQ/met data, segregated by cluster label

    Parameters
    ----------
    cluster_labels : array
        Cluster labels
    df_data : DataFrame
        Orbitrap data segregated by molecule types CHO, CHON, CHOS, CHONS. Index is time
    df_merge : Pandas DataFrame
        Merge air quality / met data. Index is time
    ds_day_frac : Pandas Series
        Daytime fraction per sample. Index is time
    **kwargs : 
        Additional keyword arguments
    suptitle : suptitle for the figure

    Returns
    -------
    None.

    """
    sns.set_context("paper", font_scale=1.2)
    
    ylabels = ["CO (ppbv)","O$_3$ (ppbv)","Orbitrap total (µg m$^{-3}$)","AMS OA (µg m$^{-3}$)",
               "AMS PMF fraction", "Precip (mm)", "Wind speed (m s$^{-1}$", "Fraction",
               "", "Fraction"
               ]
    
    
    whis=[5,95]
    
    
    #Make dataframes of the mean of each pmf factor for each cluster, as a fraction so all add up to 1
    df_merge_AMS_PMF_frac = df_merge[['AMS_FFOA','AMS_COA','AMS_BBOA','AMS_OOA']].groupby(cluster_labels).mean()
    df_merge_AMS_PMF_frac = df_merge_AMS_PMF_frac.div(df_merge_AMS_PMF_frac.sum(axis=1,skipna=False),axis=0)
    
        
    fig,ax = plt.subplots(2,5,figsize=(12,5),dpi=150)
    fig.subplots_adjust(wspace=0.6)
    ax = ax.ravel()
    
    
    #Plotting
    #CO and O3
    sns.boxplot(ax=ax[0], x=cluster_labels, y="co_ppbv", data=df_merge,showfliers=False,color='tab:gray',whis=whis)
    sns.boxplot(ax=ax[1], x=cluster_labels, y="o3_ppbv", data=df_merge,showfliers=False,color='lightsteelblue',whis=whis)
    
    
    #Orbitrap data
    sns.boxplot(ax=ax[2], x=cluster_labels, y=df_data.sum(axis=1), showfliers=False,color='tab:green',whis=whis)
    #AMS OA
    sns.boxplot(ax=ax[3], x=cluster_labels, y="AMS_Org", data=df_merge,showfliers=False,color='lime')
    #sns.boxplot(ax=ax[4], x=cluster_labels, y="AMS_inorg", data=df_merge,showfliers=False,color='tab:green')

    #AMS PMF
    df_merge_AMS_PMF_frac.plot(ax=ax[4],kind='bar', stacked=True,
                                           color=['dimgray', 'silver', 'tab:brown','mediumpurple'],
                                           legend=False,width=0.9)
        
    #Met data
    sns.boxplot(ax=ax[5], x=cluster_labels, y="HYSPLIT_precip", data=df_merge,showfliers=False,color='tab:olive',whis=whis)
    sns.boxplot(ax=ax[6], x=cluster_labels, y="ws_ms", data=df_merge,showfliers=False,color='tab:red',whis=whis)
    
    #Set axis bottoms to zero
    for axis in ax:
        axis.set_ylim(bottom=0)    
    # ax[4].set_ylim(bottom=0)
    # ax[5].set_ylim(bottom=0)
    # ax[6].set_ylim(bottom=0)
    
    
    
    
    
    #Prepare data for cat/counts plot
    ds_dataset_cat = delhi_beijing_datetime_cat(df_data.index)
    cluster_labels = np.array(cluster_labels)
    #Make data grouped by cluster
    a = pd.DataFrame(cluster_labels,columns=['clust'],index=ds_dataset_cat.index)
    df_cat_clust_counts = ds_dataset_cat.groupby(a['clust']).value_counts(normalize=True).unstack()
    
    #Day frac day and night grouped by cluster
    df_day_frac = pd.DataFrame()
    df_day_frac['Day'] = ds_day_frac
    df_day_frac['Night'] = 1 - ds_day_frac 
    
    
    df_cat_clust_counts.plot.bar(ax=ax[7],stacked=True,colormap='RdBu',width=0.8)
    df_day_frac.plot.bar(ax=ax[9],stacked=True,colormap='viridis_r',width=0.8)
    
    # #Set labels          
    # ax[0].set_xlabel('Cluster')
    # ax[0].set_ylabel('Fraction')
    # ax[1].set_xlabel('Cluster')
    # ax[1].set_ylabel('Fraction')


    #Set legend handles and size
    handles, labels = ax[4].get_legend_handles_labels()
    ax[4].legend(handles,['FFOA','COA','BBOA','OOA'],bbox_to_anchor=(1.0, 0.5),loc='center left',reverse=True)   
    handles, labels = ax[7].get_legend_handles_labels()
    ax[7].legend(handles, ['Beijing Winter','Beijing Summer','Delhi PreM', 'Delhi PostM'], bbox_to_anchor=(1.75, 1.),loc='upper center',ncol=1,handletextpad=0.4,reverse=True)            
    handles, labels = ax[9].get_legend_handles_labels()
    #ax[9].legend(handles,['Majority day','Majority night'],bbox_to_anchor=(-1.0, 0.0),loc='lower center') 
    ax[9].legend(handles,['Majority day','Majority night'],bbox_to_anchor=(1.0, 0.5),loc='center left',reverse=True) 
    
        
    
    
    
    
    
    #Set axis labels
    [axis.set_xlabel('')  for axis in ax]
    [axis.set_ylabel(ylab,labelpad=0)  for axis, ylab in zip(ax,ylabels)]
    
    #Add letters in boxes for each subfigure
    trans = ScaledTranslation(2/72, -5/72, fig.dpi_scale_trans)
    
    for axis, letter in zip(ax,string.ascii_lowercase):
        if letter == 'i':
            axis = ax[9]
            axis.text(0.0, 1.0, ('(' + letter + ')'), transform=axis.transAxes + trans,
                    fontsize='medium', verticalalignment='top', 
                    bbox=dict(facecolor='1.0', edgecolor='none', pad=1.0))
            break
        else:
            axis.text(0.0, 1.0, ('(' + letter + ')'), transform=axis.transAxes + trans,
                fontsize='medium', verticalalignment='top', 
                bbox=dict(facecolor='1.0', edgecolor='none', pad=1.0))


    #Delete empty subplot
    fig.delaxes(ax[8])
    
    if 'suptitle' in kwargs:    
        plt.suptitle(kwargs.get('suptitle'))
    #plt.tight_layout()
    
    sns.reset_orig()