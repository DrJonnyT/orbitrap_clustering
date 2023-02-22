# -*- coding: utf-8 -*-
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import string

def plot_orbitrap_ams_aqmet(cluster_labels,df_orbitrap_moltypes,df_merge,**kwargs):
    sns.set_context("paper", font_scale=1)
    
    ylabels = ["CHO (µg m$^{-3}$)","CHON (µg m$^{-3}$)","CHOS (µg m$^{-3}$)","CHONS (µg m$^{-3}$)",
               "AMS OA (µg m$^{-3}$)", "AMS NO$_3^-$ (µg m$^{-3}$)", "AMS SO$_4^{2-}$ (µg m$^{-3}$)", "AMS PMF fraction",
               "AMS FFOA (µg m$^{-3}$)","AMS COA (µg m$^{-3}$)","AMS BBOA (µg m$^{-3}$)","AMS OOA (µg m$^{-3}$)",
               "CO (ppbv)", "NO$_2$ (ppbv)", "O$_3$ (ppbv)", "SO$_2$ (ppbv)",
               "Precip (mm)", "RH(%)","",""
               ]
    
    
    whis=[5,95]
    
    
    #Make dataframes of the mean of each pmf factor for each cluster, as a fraction so all add up to 1
    df_merge_AMS_PMF_frac = df_merge[['AMS_FFOA','AMS_COA','AMS_BBOA','AMS_OOA']].groupby(cluster_labels).mean()
    df_merge_AMS_PMF_frac = df_merge_AMS_PMF_frac.div(df_merge_AMS_PMF_frac.sum(axis=1,skipna=False),axis=0)
    
    
    
    
    fig,ax = plt.subplots(5,4,figsize=(8,12))
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