# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:45:01 2022

@author: mbcx5jt5
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.colors import ListedColormap
import numpy as np


from sklearn.preprocessing import StandardScaler, FunctionTransformer, MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
#from sklearn.decomposition import PCA
#from sklearn_extra.cluster import KMedoids
#from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score,silhouette_score,adjusted_rand_score
#from sklearn.manifold import TSNE

import seaborn as sns


import os
os.chdir('C:/Work/Python/Github/Orbitrap_clustering')
from ae_functions import *
from orbitrap_functions import *

from functions.combine_multiindex import combine_multiindex
from functions.prescale_whole_matrix import prescale_whole_matrix





#%%Load data from HDF
filepath = r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\PMF data\ORBITRAP_Data_Pre_PMF.h5"
df_all_data, df_all_err, ds_all_mz = Load_pre_PMF_data(filepath,join='inner')

df_all_sig_noise = (df_all_data / df_all_err).abs().fillna(0)

#Save data to CSV
#df_all_data.to_csv(r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\PMF data\df_all_data.csv")
#df_all_err.to_csv(r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\PMF data\df_all_err.csv")
#ds_all_mz.to_csv(r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\PMF data\ds_all_mz.csv",index=False,header=False)

#pd.DataFrame(df_all_data.columns.get_level_values(0),df_all_data.columns.get_level_values(1)).to_csv(r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\PMF data\RT_formula.csv",header=False)

#Load all time data, ie start/mid/end
df_all_times = pd.read_csv(r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\PMF data\Times_all.csv")
df_all_times['date_start'] = pd.to_datetime(df_all_times['date_start'],dayfirst=True)
df_all_times['date_mid'] = pd.to_datetime(df_all_times['date_mid'],dayfirst=True)
df_all_times['date_end'] = pd.to_datetime(df_all_times['date_end'],dayfirst=True)

df_all_times.set_index(df_all_times['date_mid'],inplace=True)
fuzzy_index = pd.merge_asof(pd.DataFrame(index=df_all_data.index),df_all_times,left_index=True,right_index=True,direction='nearest',tolerance=pd.Timedelta(hours=1.25))
df_all_times = df_all_times.loc[fuzzy_index['date_mid']]

dataset_cat = delhi_beijing_datetime_cat(df_all_data)
df_dataset_cat = pd.DataFrame(delhi_beijing_datetime_cat(df_all_data),columns=['dataset_cat'],index=df_all_data.index)
ds_dataset_cat = df_dataset_cat['dataset_cat']

time_cat = delhi_calc_time_cat(df_all_times)
df_time_cat = pd.DataFrame(delhi_calc_time_cat(df_all_times),columns=['time_cat'],index=df_all_times.index)
ds_time_cat = df_time_cat['time_cat']

#This is a list of peaks with Sari's description from her PMF
Sari_peaks_list = pd.read_csv(r'C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\Sari_Peaks_Sources.csv',index_col='Formula',na_filter=False)
Sari_peaks_list = Sari_peaks_list[~Sari_peaks_list.index.duplicated(keep='first')]


#%%Work out O:C, H:C, S:C, N:C ratios for all peaks
#df_element_ratios = df_all_data.columns.get_level_values(0).to_series().apply(lambda x: chemform_ratios(x)[0])
df_element_ratios = pd.DataFrame()
df_element_ratios['H/C'] = df_all_data.columns.get_level_values(0).to_series().apply(lambda x: chemform_ratios(x)[0])
df_element_ratios['O/C'] = df_all_data.columns.get_level_values(0).to_series().apply(lambda x: chemform_ratios(x)[1])
df_element_ratios['N/C'] = df_all_data.columns.get_level_values(0).to_series().apply(lambda x: chemform_ratios(x)[2])
df_element_ratios['S/C'] = df_all_data.columns.get_level_values(0).to_series().apply(lambda x: chemform_ratios(x)[3])







#%%Prescale data

#quantile transformer
qt = QuantileTransformer(output_distribution="normal",n_quantiles=df_all_data.shape[0])
df_all_qt = pd.DataFrame(qt.fit_transform(df_all_data.to_numpy()),index=df_all_data.index,columns=df_all_data.columns)

#MinMax transformer
minmax = MinMaxScaler()
df_all_minmax = pd.DataFrame(minmax.fit_transform(df_all_data.to_numpy()),index=df_all_data.index,columns=df_all_data.columns)

# compare_cluster_metrics(df_all_data,2,12,cluster_type='agglom',suptitle_prefix='Unscaled data', suptitle_suffix='')

# compare_cluster_metrics(df_qt,2,12,cluster_type='agglom',suptitle_prefix='Quantile transformed data', suptitle_suffix='')

# compare_cluster_metrics(df_minmax,2,12,cluster_type='agglom',suptitle_prefix='MinMax scaled data', suptitle_suffix='')

# compare_cluster_metrics(df_all_sig_noise,2,12,cluster_type='agglom',suptitle_prefix='Sig/noise data', suptitle_suffix='')


#%%Extract the n biggest peaks from the original data and corresponding compounds from the prescaled data
n_peaks = 8
df_top_peaks_list = cluster_extract_peaks(df_all_data.mean(), df_all_data.T,n_peaks,dp=1,dropRT=False)
df_top_peaks_unscaled = df_all_data[df_top_peaks_list.index]
df_top_peaks_qt = df_all_qt[df_top_peaks_list.index]
df_top_peaks_minmax = df_all_minmax[df_top_peaks_list.index]
df_top_peaks_sig_noise = df_all_sig_noise[df_top_peaks_list.index]

#df_top_peaks_unscaled.columns = df_top_peaks_unscaled.columns.get_level_values(0) + ", " +  df_top_peaks_unscaled.columns.get_level_values(1).astype(str)


df_top_peaks_unscaled.columns = combine_multiindex(df_top_peaks_unscaled.columns)
df_top_peaks_qt.columns = combine_multiindex(df_top_peaks_qt.columns)
df_top_peaks_minmax.columns = combine_multiindex(df_top_peaks_minmax.columns)
df_top_peaks_sig_noise.columns = combine_multiindex(df_top_peaks_sig_noise.columns)
    



#%%Pairplots distributions of these n biggest peaks
sns.set_context("talk", font_scale=1)

#Unscaled data
#g = sns.pairplot(df_top_peaks_unscaled,plot_kws=dict(marker="+", linewidth=1)).fig.suptitle("Unscaled data", y=1.01,fontsize=20)
g = sns.PairGrid(df_top_peaks_unscaled,corner=True)
g.fig.suptitle("Unscaled data", y=0.95,fontsize=26)
g.map_lower(sns.scatterplot, hue=ds_dataset_cat.cat.codes,palette = 'RdBu',linewidth=0.5,s=50)
g.map_diag(plt.hist, color='grey',edgecolor='black', linewidth=1.2)
g.add_legend(fontsize=26)
g.legend.get_texts()[0].set_text('Beijing Winter') # You can also change the legend title
g.legend.get_texts()[1].set_text('Beijing Summer')
g.legend.get_texts()[2].set_text('Delhi Summer')
g.legend.get_texts()[3].set_text('Delhi Autumn')
sns.move_legend(g, "center right", bbox_to_anchor=(0.8, 0.55), title='Dataset')
plt.setp(g.legend.get_title(), fontsize='24')
plt.setp(g.legend.get_texts(), fontsize='24')
plt.show()


#MinMax data
#sns.pairplot(df_top_peaks_minmax,plot_kws=dict(marker="+", linewidth=1)).fig.suptitle("MinMax data", y=1.01,fontsize=20)
g = sns.PairGrid(df_top_peaks_minmax,corner=True)
g.fig.suptitle("MinMax data", y=0.95,fontsize=26)
g.map_lower(sns.scatterplot, hue=ds_dataset_cat.cat.codes,palette = 'RdBu',linewidth=0.5,s=50)
g.map_diag(plt.hist, color='grey',edgecolor='black', linewidth=1.2)
g.add_legend(fontsize=26)
g.legend.get_texts()[0].set_text('Beijing Winter') # You can also change the legend title
g.legend.get_texts()[1].set_text('Beijing Summer')
g.legend.get_texts()[2].set_text('Delhi Summer')
g.legend.get_texts()[3].set_text('Delhi Autumn')
sns.move_legend(g, "center right", bbox_to_anchor=(0.8, 0.55), title='Dataset')
plt.setp(g.legend.get_title(), fontsize='24')
plt.setp(g.legend.get_texts(), fontsize='24')
plt.show()


#QT data
#sns.pairplot(df_top_peaks_qt,plot_kws=dict(marker="+", linewidth=1)).fig.suptitle("QT data", y=1.01,fontsize=20)
g = sns.PairGrid(df_top_peaks_qt,corner=True)
g.fig.suptitle("QuantileTansformer data", y=0.95,fontsize=26)
g.map_lower(sns.scatterplot, hue=ds_dataset_cat.cat.codes,palette = 'RdBu',linewidth=0.5,s=50)
g.map_diag(plt.hist, color='grey',edgecolor='black', linewidth=1.2)
g.add_legend(fontsize=26)
g.legend.get_texts()[0].set_text('Beijing Winter') # You can also change the legend title
g.legend.get_texts()[1].set_text('Beijing Summer')
g.legend.get_texts()[2].set_text('Delhi Summer')
g.legend.get_texts()[3].set_text('Delhi Autumn')
sns.move_legend(g, "center right", bbox_to_anchor=(0.8, 0.55), title='Dataset')
plt.setp(g.legend.get_title(), fontsize='24')
plt.setp(g.legend.get_texts(), fontsize='24')
plt.show()


#Sig/noise data
#sns.pairplot(df_top_peaks_sig_noise,plot_kws=dict(marker="+", linewidth=1)).fig.suptitle("Sig/noise data", y=1.01,fontsize=20)
g = sns.PairGrid(df_top_peaks_sig_noise,corner=True)
g.fig.suptitle("Sig/noise data", y=0.95,fontsize=26)
g.map_lower(sns.scatterplot, hue=ds_dataset_cat.cat.codes,palette = 'RdBu',linewidth=0.5,s=50)
g.map_diag(plt.hist, color='grey',edgecolor='black', linewidth=1.2)
g.add_legend(fontsize=26)
g.legend.get_texts()[0].set_text('Beijing Winter') # You can also change the legend title
g.legend.get_texts()[1].set_text('Beijing Summer')
g.legend.get_texts()[2].set_text('Delhi Summer')
g.legend.get_texts()[3].set_text('Delhi Autumn')
sns.move_legend(g, "center right", bbox_to_anchor=(0.8, 0.55), title='Dataset')
plt.setp(g.legend.get_title(), fontsize='24')
plt.setp(g.legend.get_texts(), fontsize='24')
plt.show()


#%%Clustering workflow - unscaled data
df_cluster_labels_mtx = cluster_n_times(df_all_data,10,min_num_clusters=2,cluster_type='agglom')
df_cluster_counts_mtx = count_cluster_labels_from_mtx(df_cluster_labels_mtx)

df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx = calc_cluster_elemental_ratios(df_cluster_labels_mtx,df_all_data,df_element_ratios)
plot_cluster_elemental_ratios(df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,'Unscaled data HCA elemental ratios')

cluster_profiles_mtx, cluster_profiles_mtx_norm, num_clusters_index, cluster_index = average_cluster_profiles(df_cluster_labels_mtx,df_all_data)
df_cluster_corr_mtx, df_prevcluster_corr_mtx = correlate_cluster_profiles(cluster_profiles_mtx_norm, num_clusters_index, cluster_index)
plot_cluster_profile_corrs(df_cluster_corr_mtx, df_prevcluster_corr_mtx,suptitle='Unscaled data HCA')

plot_all_cluster_tseries_BeijingDelhi(df_cluster_labels_mtx,ds_dataset_cat,title_prefix='Unscaled data HCA, ')

plot_all_cluster_profiles(df_all_data,cluster_profiles_mtx_norm,num_clusters_index,ds_all_mz,df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,
                          df_cluster_corr_mtx,df_prevcluster_corr_mtx,df_cluster_counts_mtx,Sari_peaks_list,title_prefix='Unscaled data HCA, ')
                        
df_clust_cat_counts, df_cat_clust_counts, df_clust_time_cat_counts,df_time_cat_clust_counts = count_clusters_project_time(
    df_cluster_labels_mtx,ds_dataset_cat,ds_time_cat,title_prefix='Unscaled data HCA, ',title_suffix='')
    
compare_cluster_metrics(df_all_data,2,12,'agglom','Unscaled data ',' metrics')


#%%Clustering workflow - MinMax data
df_cluster_labels_mtx = cluster_n_times(df_all_minmax,10,min_num_clusters=2,cluster_type='agglom')
df_cluster_counts_mtx = count_cluster_labels_from_mtx(df_cluster_labels_mtx)

df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx = calc_cluster_elemental_ratios(df_cluster_labels_mtx,df_all_data,df_element_ratios)
plot_cluster_elemental_ratios(df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,'MinMax data HCA elemental ratios')

cluster_profiles_mtx, cluster_profiles_mtx_norm, num_clusters_index, cluster_index = average_cluster_profiles(df_cluster_labels_mtx,df_all_data)
df_cluster_corr_mtx, df_prevcluster_corr_mtx = correlate_cluster_profiles(cluster_profiles_mtx_norm, num_clusters_index, cluster_index)
plot_cluster_profile_corrs(df_cluster_corr_mtx, df_prevcluster_corr_mtx,suptitle='MinMax data HCA')

plot_all_cluster_tseries_BeijingDelhi(df_cluster_labels_mtx,ds_dataset_cat,title_prefix='MinMaxdata HCA, ')

plot_all_cluster_profiles(df_all_data,cluster_profiles_mtx_norm,num_clusters_index,ds_all_mz,df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,
                          df_cluster_corr_mtx,df_prevcluster_corr_mtx,df_cluster_counts_mtx,Sari_peaks_list,title_prefix='MinMax data HCA, ')
                        
df_clust_cat_counts, df_cat_clust_counts, df_clust_time_cat_counts,df_time_cat_clust_counts = count_clusters_project_time(
    df_cluster_labels_mtx,ds_dataset_cat,ds_time_cat,title_prefix='MinMax data HCA, ',title_suffix='')
    
compare_cluster_metrics(df_all_minmax,2,12,'agglom','MinMax data ',' metrics')

#%%Clustering workflow - Quantile transformed data
df_cluster_labels_mtx = cluster_n_times(df_all_qt,10,min_num_clusters=2,cluster_type='agglom')
df_cluster_counts_mtx = count_cluster_labels_from_mtx(df_cluster_labels_mtx)

df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx = calc_cluster_elemental_ratios(df_cluster_labels_mtx,df_all_data,df_element_ratios)
plot_cluster_elemental_ratios(df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,'QT data HCA elemental ratios')

cluster_profiles_mtx, cluster_profiles_mtx_norm, num_clusters_index, cluster_index = average_cluster_profiles(df_cluster_labels_mtx,df_all_data)
df_cluster_corr_mtx, df_prevcluster_corr_mtx = correlate_cluster_profiles(cluster_profiles_mtx_norm, num_clusters_index, cluster_index)
plot_cluster_profile_corrs(df_cluster_corr_mtx, df_prevcluster_corr_mtx,suptitle='QT data HCA')

plot_all_cluster_tseries_BeijingDelhi(df_cluster_labels_mtx,ds_dataset_cat,title_prefix='QT data HCA, ')

plot_all_cluster_profiles(df_all_data,cluster_profiles_mtx_norm,num_clusters_index,ds_all_mz,df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,
                          df_cluster_corr_mtx,df_prevcluster_corr_mtx,df_cluster_counts_mtx,Sari_peaks_list,title_prefix='QT data HCA, ')
                        
df_clust_cat_counts, df_cat_clust_counts, df_clust_time_cat_counts,df_time_cat_clust_counts = count_clusters_project_time(
    df_cluster_labels_mtx,ds_dataset_cat,ds_time_cat,title_prefix='QT data HCA, ',title_suffix='')
    
compare_cluster_metrics(df_all_qt,2,12,'agglom','QT data ',' metrics')

#%%Clustering workflow - Sig/noise transformed data
df_cluster_labels_mtx = cluster_n_times(df_all_sig_noise,10,min_num_clusters=2,cluster_type='agglom')
df_cluster_counts_mtx = count_cluster_labels_from_mtx(df_cluster_labels_mtx)

df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx = calc_cluster_elemental_ratios(df_cluster_labels_mtx,df_all_data,df_element_ratios)
plot_cluster_elemental_ratios(df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,'Sig/noise data HCA elemental ratios')

cluster_profiles_mtx, cluster_profiles_mtx_norm, num_clusters_index, cluster_index = average_cluster_profiles(df_cluster_labels_mtx,df_all_data)
df_cluster_corr_mtx, df_prevcluster_corr_mtx = correlate_cluster_profiles(cluster_profiles_mtx_norm, num_clusters_index, cluster_index)
plot_cluster_profile_corrs(df_cluster_corr_mtx, df_prevcluster_corr_mtx,suptitle='Sig/noise data HCA')

plot_all_cluster_tseries_BeijingDelhi(df_cluster_labels_mtx,ds_dataset_cat,title_prefix='Sig/noise data HCA, ')

plot_all_cluster_profiles(df_all_data,cluster_profiles_mtx_norm,num_clusters_index,ds_all_mz,df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,
                          df_cluster_corr_mtx,df_prevcluster_corr_mtx,df_cluster_counts_mtx,Sari_peaks_list,title_prefix='Sig/noise data HCA, ')
                        
df_clust_cat_counts, df_cat_clust_counts, df_clust_time_cat_counts,df_time_cat_clust_counts = count_clusters_project_time(
    df_cluster_labels_mtx,ds_dataset_cat,ds_time_cat,title_prefix='Sig/noise data HCA, ',title_suffix='')
    
compare_cluster_metrics(df_all_sig_noise,2,12,'agglom','Sig/noise data ',' metrics')

