# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from orbitrap_functions import cluster_extract_peaks
#%%Plot cluster profiles
def plot_all_cluster_profiles(df_all_data,cluster_profiles_mtx_norm, num_clusters_index,mz_columns,df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,df_cluster_corr_mtx,df_prevcluster_corr_mtx,df_cluster_counts_mtx=pd.DataFrame(),peaks_list=pd.DataFrame(columns=['Source']),title_prefix=''):
    for num_clusters in num_clusters_index:
        suptitle = title_prefix + str(int(num_clusters)) + ' clusters'
        plot_one_cluster_profile(df_all_data,cluster_profiles_mtx_norm, num_clusters_index,num_clusters, mz_columns,df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,df_cluster_corr_mtx,df_prevcluster_corr_mtx,df_cluster_counts_mtx,peaks_list,suptitle)
    
            
def plot_one_cluster_profile(df_all_data,cluster_profiles_mtx_norm, num_clusters_index, num_clusters, mz_columns,
                             df_clusters_HC_mtx=pd.DataFrame(),df_clusters_NC_mtx=pd.DataFrame(),
                             df_clusters_OC_mtx=pd.DataFrame(),df_clusters_SC_mtx=pd.DataFrame(),
                             df_cluster_corr_mtx=pd.DataFrame(),df_prevcluster_corr_mtx=pd.DataFrame(),
                             df_cluster_counts_mtx=pd.DataFrame(),peaks_list=pd.DataFrame(columns=['Source']),suptitle=''):
    
    sns.set_context("talk", font_scale=0.8)
    num_clusters_index = np.atleast_1d(num_clusters_index)
    
    if((num_clusters_index.shape[0])==1):    #Check if min number of clusters is 1
        if(num_clusters_index[0] == num_clusters):
            x_idx=0
        else:
            print("plot_cluster_profiles() error: nclusters is " + str(num_clusters) + " which is not in num_clusters_index")
    else:
        x_idx = np.searchsorted(num_clusters_index,num_clusters,side='left')
        if(x_idx == np.searchsorted(num_clusters_index,num_clusters,side='right')):
            print("plot_cluster_profiles() error: nclusters is " + str(num_clusters) + " which is not in num_clusters_index")
            return 0
    
    fig,axes = plt.subplots(num_clusters,2,figsize=(14,2.9*num_clusters),gridspec_kw={'width_ratios': [8, 4]},constrained_layout=True)
    fig.suptitle(suptitle)
    #cluster_profiles_2D = cluster_profiles_mtx_norm[x_idx,:,:]
    for y_idx in np.arange(num_clusters):
        this_cluster_profile = cluster_profiles_mtx_norm[x_idx,y_idx,:]
        ax = axes[-y_idx-1][0]
        ax.stem(mz_columns.to_numpy(),this_cluster_profile,markerfmt=' ')
        ax.set_xlim(left=100,right=400)
        ax.set_xlabel('m/z')
        ax.set_ylabel('Relative concentration')
        #ax.set_title('Cluster' + str(y_idx))
        ax.text(0.01, 0.95, 'Cluster ' + str(y_idx), transform=ax.transAxes, fontsize=12,
                verticalalignment='top')
        
        #Add in elemental ratios
        if(df_clusters_HC_mtx.empty == False ):
            ax.text(0.69, 0.95, 'H/C = ' + "{:.2f}".format(df_clusters_HC_mtx.loc[num_clusters][y_idx]), transform=ax.transAxes, fontsize=12,
                    verticalalignment='top')
        if(df_clusters_NC_mtx.empty == False ):
            ax.text(0.84, 0.95, 'N/C = ' + "{:.3f}".format(df_clusters_NC_mtx.loc[num_clusters][y_idx]), transform=ax.transAxes, fontsize=12,
                    verticalalignment='top')
        if(df_clusters_OC_mtx.empty == False ):
            ax.text(0.69, 0.85, 'O/C = ' + "{:.2f}".format(df_clusters_OC_mtx.loc[num_clusters][y_idx]), transform=ax.transAxes, fontsize=12,
                    verticalalignment='top')
        if(df_clusters_SC_mtx.empty == False ):
            ax.text(0.84, 0.85, 'S/C = ' + "{:.3f}".format(df_clusters_SC_mtx.loc[num_clusters][y_idx]), transform=ax.transAxes, fontsize=12,
                    verticalalignment='top')
        
        #Add in number of data points for this cluster
        if(df_cluster_counts_mtx.empty == False):
            #Find num samples in this cluster
            num_samples_this_cluster = df_cluster_counts_mtx.loc[num_clusters][y_idx]
            if(num_samples_this_cluster==1):
                ax.text(0.69, 0.75, str(int(num_samples_this_cluster)) + ' sample', transform=ax.transAxes, fontsize=12,
                verticalalignment='top')
            else:
                ax.text(0.69, 0.75, str(int(num_samples_this_cluster)) + ' samples', transform=ax.transAxes, fontsize=12,
                verticalalignment='top')
            
        #Add in best correlation
        if(df_cluster_corr_mtx.empty == False ):
            #Find best cluster correlation
            best_R = df_cluster_corr_mtx.loc[num_clusters][y_idx]
            ax.text(0.69, 0.65, 'Highest R = ' + str(round(best_R,2) ), transform=ax.transAxes, fontsize=12,
                    verticalalignment='top')
        if(df_prevcluster_corr_mtx.empty == False):
            #Find best cluster correlation
            best_R_prev = df_prevcluster_corr_mtx.loc[num_clusters][y_idx]
            if(best_R_prev < 0.9999999):
                ax.text(0.69, 0.55, 'Highest R_prev = ' + str(round(best_R_prev,2) ), transform=ax.transAxes, fontsize=12,
                    verticalalignment='top')
        
        
        
    
        #Add in a table with the top peaks
        ds_this_cluster_profile = pd.Series(this_cluster_profile,index=df_all_data.columns).T
        df_top_peaks = cluster_extract_peaks(ds_this_cluster_profile, df_all_data.T,10,dp=1,dropRT=False)
        #pdb.set_trace()
        df_top_peaks.index = df_top_peaks.index.get_level_values(0).str.replace(' ', '')
        ax2 = axes[-y_idx-1][1]
        cellText = pd.merge(df_top_peaks, peaks_list, how="left",left_index=True,right_index=True)[['RT','peak_pct','Source']]
        cellText.sort_values('peak_pct',inplace=True,ascending=False)
        cellText['Source'] = cellText['Source'].astype(str).replace(to_replace='nan',value='')
        cellText = cellText.reset_index().values
        the_table = ax2.table(cellText=cellText,loc='center',
                              colLabels=['Formula','RT','%','Potential source'],
                              cellLoc = 'left',
                              colLoc = 'left',
                              edges='open',colWidths=[0.3,0.1,0.1,0.5])
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(11)
        cells = the_table.properties()["celld"]

        #Set alignment of column headers
        cells[0,1].set_text_props(ha="right")
        cells[0,2].set_text_props(ha="right")
        #Set alignment of cells
        for i in range(1, 11):
            cells[i, 1].set_text_props(ha="right")
            cells[i, 2].set_text_props(ha="right")
        

        
    
    sns.reset_orig()