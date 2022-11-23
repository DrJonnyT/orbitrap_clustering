# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:45:01 2022

@author: mbcx5jt5
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.colors import ListedColormap

import matplotlib as mpl


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
from functions.optimal_nclusters_r_card import optimal_nclusters_r_card
from functions.avg_array_clusters import avg_array_clusters





#%%Load data from HDF
filepath = r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\PMF data\ORBITRAP_Data_Pre_PMF.h5"
df_all_data, df_all_err, ds_all_mz = Load_pre_PMF_data(filepath,join='inner')

df_all_signoise = (df_all_data / df_all_err).abs().fillna(0)

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



#Work out time series of these variables in the unscaled data
df_element_ratios_tseries = pd.DataFrame()
df_element_ratios_tseries['H/C'] = df_all_data.set_axis(df_all_data.columns.get_level_values(0),axis=1).multiply(df_element_ratios['H/C'],axis=1).sum(axis=1) / df_all_data.sum(axis=1)
df_element_ratios_tseries['O/C'] = df_all_data.set_axis(df_all_data.columns.get_level_values(0),axis=1).multiply(df_element_ratios['O/C'],axis=1).sum(axis=1) / df_all_data.sum(axis=1)
df_element_ratios_tseries['N/C'] = df_all_data.set_axis(df_all_data.columns.get_level_values(0),axis=1).multiply(df_element_ratios['N/C'],axis=1).sum(axis=1) / df_all_data.sum(axis=1)
df_element_ratios_tseries['S/C'] = df_all_data.set_axis(df_all_data.columns.get_level_values(0),axis=1).multiply(df_element_ratios['S/C'],axis=1).sum(axis=1) / df_all_data.sum(axis=1)



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

# compare_cluster_metrics(df_all_signoise,2,12,cluster_type='agglom',suptitle_prefix='Sig/noise data', suptitle_suffix='')


#%%Extract the n biggest peaks from the original data and corresponding compounds from the prescaled data
n_peaks = 8
df_top_peaks_list = cluster_extract_peaks(df_all_data.mean(), df_all_data.T,n_peaks,dp=1,dropRT=False)
df_top_peaks_unscaled = df_all_data[df_top_peaks_list.index]
df_top_peaks_qt = df_all_qt[df_top_peaks_list.index]
df_top_peaks_minmax = df_all_minmax[df_top_peaks_list.index]
df_top_peaks_signoise = df_all_signoise[df_top_peaks_list.index]

#df_top_peaks_unscaled.columns = df_top_peaks_unscaled.columns.get_level_values(0) + ", " +  df_top_peaks_unscaled.columns.get_level_values(1).astype(str)


df_top_peaks_unscaled.columns = combine_multiindex(df_top_peaks_unscaled.columns)
df_top_peaks_qt.columns = combine_multiindex(df_top_peaks_qt.columns)
df_top_peaks_minmax.columns = combine_multiindex(df_top_peaks_minmax.columns)
df_top_peaks_signoise.columns = combine_multiindex(df_top_peaks_signoise.columns)
    



#%%Pairplots distributions of these n biggest peaks

#Set Seaborn context so plots have better font sizes
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
#sns.pairplot(df_top_peaks_signoise,plot_kws=dict(marker="+", linewidth=1)).fig.suptitle("Sig/noise data", y=1.01,fontsize=20)
g = sns.PairGrid(df_top_peaks_signoise,corner=True)
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


#Reset seaborn context so matplotlib plots are not messed up
sns.reset_orig()


#%%Cluster metrics for four different data prescaling
df_cluster_labels_mtx_unscaled = cluster_n_times(df_all_data,10,min_num_clusters=2,cluster_type='agglom')
df_cluster_counts_mtx_unscaled = count_cluster_labels_from_mtx(df_cluster_labels_mtx_unscaled)

df_cluster_labels_mtx_minmax = cluster_n_times(df_all_minmax,10,min_num_clusters=2,cluster_type='agglom')
df_cluster_counts_mtx_minmax = count_cluster_labels_from_mtx(df_cluster_labels_mtx_minmax)

df_cluster_labels_mtx_qt = cluster_n_times(df_all_qt,10,min_num_clusters=2,cluster_type='agglom')
df_cluster_counts_mtx_qt = count_cluster_labels_from_mtx(df_cluster_labels_mtx_qt)

df_cluster_labels_mtx_signoise = cluster_n_times(df_all_signoise,10,min_num_clusters=2,cluster_type='agglom')
df_cluster_counts_mtx_signoise = count_cluster_labels_from_mtx(df_cluster_labels_mtx_signoise)

df_cluster_corr_mtx_unscaled, _ = correlate_cluster_profiles(*average_cluster_profiles(df_cluster_labels_mtx_unscaled,df_all_data)[1:])
df_cluster_corr_mtx_qt, _ = correlate_cluster_profiles(*average_cluster_profiles(df_cluster_labels_mtx_qt,df_all_data)[1:])
df_cluster_corr_mtx_minmax, _ = correlate_cluster_profiles(*average_cluster_profiles(df_cluster_labels_mtx_minmax,df_all_data)[1:])
df_cluster_corr_mtx_signoise, _ = correlate_cluster_profiles(*average_cluster_profiles(df_cluster_labels_mtx_signoise,df_all_data)[1:])





fig,ax = plt.subplots(2,1,figsize=(6,9))
ax[0].plot(df_cluster_labels_mtx_unscaled.columns,df_cluster_counts_mtx_unscaled.min(axis=1),label='Unscaled',linewidth=2)
ax[0].plot(df_cluster_labels_mtx_minmax.columns,df_cluster_counts_mtx_minmax.min(axis=1),label='MinMax',linewidth=2)
ax[0].plot(df_cluster_labels_mtx_qt.columns,df_cluster_counts_mtx_qt.min(axis=1),label='QT',linewidth=2)
ax[0].plot(df_cluster_labels_mtx_signoise.columns,df_cluster_counts_mtx_signoise.min(axis=1),label='Sig/noise',c='k',linewidth=2)
ax[0].legend(framealpha=1.)
ax[0].set_ylabel('Cardinality (num points) of smallest cluster',fontsize=12)
ax[0].set_xlabel('Num clusters',fontsize=12)
ax[0].set_yticks(np.arange(0, 230, 10))
ax[0].grid(axis='y')

ax[1].plot(df_cluster_labels_mtx_unscaled.columns,df_cluster_corr_mtx_unscaled.max(axis=1),label='Unscaled',linewidth=2)
ax[1].plot(df_cluster_labels_mtx_minmax.columns,df_cluster_corr_mtx_minmax.max(axis=1),label='MinMax',linewidth=2)
ax[1].plot(df_cluster_labels_mtx_qt.columns,df_cluster_corr_mtx_qt.max(axis=1),label='QT',linewidth=2)
ax[1].plot(df_cluster_labels_mtx_signoise.columns,df_cluster_corr_mtx_signoise.max(axis=1),label='Sig/noise',c='k',linewidth=2)
ax[1].legend(framealpha=1.)
ax[1].set_ylabel('Max R between clusters',fontsize=12)
ax[1].set_xlabel('Num clusters',fontsize=12)
ax[1].set_yticks(np.arange(0.35, 1.05, 0.05))
ax[1].grid(axis='y')

fig.suptitle('Cluster cardinality and similarity',fontsize=16)

plt.tight_layout()
plt.show()


#Optimal num clusters based on R and min cardinality
nclusters_unscaled = optimal_nclusters_r_card(df_cluster_labels_mtx_unscaled.columns.to_numpy(),df_cluster_corr_mtx_unscaled.max(axis=1),
                             df_cluster_counts_mtx_unscaled.min(axis=1))

nclusters_minmax = optimal_nclusters_r_card(df_cluster_labels_mtx_minmax.columns,df_cluster_corr_mtx_minmax.max(axis=1),
                             df_cluster_counts_mtx_minmax.min(axis=1))

nclusters_qt = optimal_nclusters_r_card(df_cluster_labels_mtx_qt.columns,df_cluster_corr_mtx_qt.max(axis=1),
                             df_cluster_counts_mtx_qt.min(axis=1))

nclusters_signoise = optimal_nclusters_r_card(df_cluster_labels_mtx_signoise.columns,df_cluster_corr_mtx_signoise.max(axis=1),
                             df_cluster_counts_mtx_signoise.min(axis=1))

#Optimal num clusters based just on R
nclusters_unscaled = optimal_nclusters_r_card(df_cluster_labels_mtx_unscaled.columns,
                                              df_cluster_corr_mtx_unscaled.max(axis=1),
                                              df_cluster_counts_mtx_unscaled.min(axis=1),mincard_threshold=1)

nclusters_minmax = optimal_nclusters_r_card(df_cluster_labels_mtx_minmax.columns,
                                            df_cluster_corr_mtx_minmax.max(axis=1),
                                            df_cluster_counts_mtx_minmax.min(axis=1),mincard_threshold=1)

nclusters_qt = optimal_nclusters_r_card(df_cluster_labels_mtx_qt.columns,
                                        df_cluster_corr_mtx_qt.max(axis=1),
                                        df_cluster_counts_mtx_qt.min(axis=1),mincard_threshold=1)

nclusters_signoise = optimal_nclusters_r_card(df_cluster_labels_mtx_signoise.columns,
                                              df_cluster_corr_mtx_signoise.max(axis=1),
                                              df_cluster_counts_mtx_signoise.min(axis=1),mincard_threshold=1)


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

plot_all_cluster_tseries_BeijingDelhi(df_cluster_labels_mtx,ds_dataset_cat,title_prefix='MinMax data HCA, ')

plot_all_cluster_profiles(df_all_data,cluster_profiles_mtx_norm,num_clusters_index,ds_all_mz,df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,
                          df_cluster_corr_mtx,df_prevcluster_corr_mtx,df_cluster_counts_mtx,Sari_peaks_list,title_prefix='MinMax data HCA, ')
                        
df_clust_cat_counts, df_cat_clust_counts, df_clust_time_cat_counts,df_time_cat_clust_counts = count_clusters_project_time(
    df_cluster_labels_mtx,ds_dataset_cat,ds_time_cat,title_prefix='MinMax data HCA, ',title_suffix='')
    
compare_cluster_metrics(df_all_minmax,2,12,'agglom','MinMax data ',' metrics')



#Plot the Minmax-scaled data
cluster_profiles_mtx, cluster_profiles_mtx_norm, num_clusters_index, cluster_index = average_cluster_profiles(df_cluster_labels_mtx,df_all_minmax)
df_cluster_corr_mtx, df_prevcluster_corr_mtx = correlate_cluster_profiles(cluster_profiles_mtx_norm, num_clusters_index, cluster_index)
plot_all_cluster_profiles(df_all_minmax,cluster_profiles_mtx_norm,num_clusters_index,ds_all_mz,df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,
                          df_cluster_corr_mtx,df_prevcluster_corr_mtx,df_cluster_counts_mtx,Sari_peaks_list,title_prefix='MinMax data HCA, scaled data, ')


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



#Plot the QT-scaled data
cluster_profiles_mtx, cluster_profiles_mtx_norm, num_clusters_index, cluster_index = average_cluster_profiles(df_cluster_labels_mtx,df_all_qt)
df_cluster_corr_mtx, df_prevcluster_corr_mtx = correlate_cluster_profiles(cluster_profiles_mtx_norm, num_clusters_index, cluster_index)
plot_all_cluster_profiles(df_all_qt,cluster_profiles_mtx_norm,num_clusters_index,ds_all_mz,df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,
                          df_cluster_corr_mtx,df_prevcluster_corr_mtx,df_cluster_counts_mtx,Sari_peaks_list,title_prefix='QT data HCA, scaled data, ')



#%%Clustering workflow - Sig/noise transformed data
df_cluster_labels_mtx = cluster_n_times(df_all_signoise,10,min_num_clusters=2,cluster_type='agglom')
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
    
compare_cluster_metrics(df_all_signoise,2,12,'agglom','Sig/noise data ',' metrics')


#Plot the QT-scaled data
cluster_profiles_mtx, cluster_profiles_mtx_norm, num_clusters_index, cluster_index = average_cluster_profiles(df_cluster_labels_mtx,df_all_signoise)
df_cluster_corr_mtx, df_prevcluster_corr_mtx = correlate_cluster_profiles(cluster_profiles_mtx_norm, num_clusters_index, cluster_index)
plot_all_cluster_profiles(df_all_signoise,cluster_profiles_mtx_norm,num_clusters_index,ds_all_mz,df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,
                          df_cluster_corr_mtx,df_prevcluster_corr_mtx,df_cluster_counts_mtx,Sari_peaks_list,title_prefix='Sig/Noise data HCA, scaled data, ')





#%%Now just doing the specified number of clusters
cluster_labels_unscaled = AgglomerativeClustering(4).fit_predict(df_all_data)
HC_clusters_unscaled = avg_array_clusters(cluster_labels_unscaled,df_element_ratios_tseries['H/C'],weights=df_all_data.sum(axis=1))
NC_clusters_unscaled = avg_array_clusters(cluster_labels_unscaled,df_element_ratios_tseries['N/C'],weights=df_all_data.sum(axis=1))
OC_clusters_unscaled = avg_array_clusters(cluster_labels_unscaled,df_element_ratios_tseries['O/C'],weights=df_all_data.sum(axis=1))
SC_clusters_unscaled = avg_array_clusters(cluster_labels_unscaled,df_element_ratios_tseries['S/C'],weights=df_all_data.sum(axis=1))

cluster_labels_minmax = AgglomerativeClustering(6).fit_predict(df_all_minmax)
HC_clusters_minmax = avg_array_clusters(cluster_labels_minmax,df_element_ratios_tseries['H/C'],weights=df_all_data.sum(axis=1))
NC_clusters_minmax = avg_array_clusters(cluster_labels_minmax,df_element_ratios_tseries['N/C'],weights=df_all_data.sum(axis=1))
OC_clusters_minmax = avg_array_clusters(cluster_labels_minmax,df_element_ratios_tseries['O/C'],weights=df_all_data.sum(axis=1))
SC_clusters_minmax = avg_array_clusters(cluster_labels_minmax,df_element_ratios_tseries['S/C'],weights=df_all_data.sum(axis=1))

cluster_labels_qt = AgglomerativeClustering(6).fit_predict(df_all_qt)
HC_clusters_qt = avg_array_clusters(cluster_labels_qt,df_element_ratios_tseries['H/C'],weights=df_all_data.sum(axis=1))
NC_clusters_qt = avg_array_clusters(cluster_labels_qt,df_element_ratios_tseries['N/C'],weights=df_all_data.sum(axis=1))
OC_clusters_qt = avg_array_clusters(cluster_labels_qt,df_element_ratios_tseries['O/C'],weights=df_all_data.sum(axis=1))
SC_clusters_qt = avg_array_clusters(cluster_labels_qt,df_element_ratios_tseries['S/C'],weights=df_all_data.sum(axis=1))

cluster_labels_signoise = AgglomerativeClustering(6).fit_predict(df_all_signoise)
HC_clusters_signoise = avg_array_clusters(cluster_labels_signoise,df_element_ratios_tseries['H/C'],weights=df_all_data.sum(axis=1))
NC_clusters_signoise = avg_array_clusters(cluster_labels_signoise,df_element_ratios_tseries['N/C'],weights=df_all_data.sum(axis=1))
OC_clusters_signoise = avg_array_clusters(cluster_labels_signoise,df_element_ratios_tseries['O/C'],weights=df_all_data.sum(axis=1))
SC_clusters_signoise = avg_array_clusters(cluster_labels_signoise,df_element_ratios_tseries['S/C'],weights=df_all_data.sum(axis=1))

#Average value of each, across the whole dataset
HC_all_avg = avg_array_clusters(np.zeros(len(df_element_ratios_tseries['H/C'])),df_element_ratios_tseries['H/C'],weights=df_all_data.sum(axis=1))[0]
NC_all_avg = avg_array_clusters(np.zeros(len(df_element_ratios_tseries['N/C'])),df_element_ratios_tseries['N/C'],weights=df_all_data.sum(axis=1))[0]
OC_all_avg = avg_array_clusters(np.zeros(len(df_element_ratios_tseries['O/C'])),df_element_ratios_tseries['O/C'],weights=df_all_data.sum(axis=1))[0]
SC_all_avg = avg_array_clusters(np.zeros(len(df_element_ratios_tseries['S/C'])),df_element_ratios_tseries['S/C'],weights=df_all_data.sum(axis=1))[0]


#%%Plot element ratio bar charts
sns.set_context("talk", font_scale=1)

#Make all the y scales the same
HC_scale_max = 1.05 * np.amax([HC_clusters_unscaled.max(),HC_clusters_minmax.max(),HC_clusters_qt.max(),HC_clusters_signoise.max()])
NC_scale_max = 1.05 * np.amax([NC_clusters_unscaled.max(),NC_clusters_minmax.max(),NC_clusters_qt.max(),NC_clusters_signoise.max()])
OC_scale_max = 1.05 * np.amax([OC_clusters_unscaled.max(),OC_clusters_minmax.max(),OC_clusters_qt.max(),OC_clusters_signoise.max()])
SC_scale_max = 1.05 * np.amax([SC_clusters_unscaled.max(),SC_clusters_minmax.max(),SC_clusters_qt.max(),SC_clusters_signoise.max()])

fig,ax = plt.subplots(2,2,figsize=(10,10))
ax = ax.ravel()
ax[0].bar(HC_clusters_unscaled.index.astype(str),HC_clusters_unscaled.values,color='gray')
ax[0].set_ylabel('H/C')
ax[0].set_ylim(None,HC_scale_max)
ax[0].axhline(y=HC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
ax[1].bar(NC_clusters_unscaled.index.astype(str),NC_clusters_unscaled.values)
ax[1].set_ylabel('N/C')
ax[1].set_ylim(None,NC_scale_max)
ax[1].axhline(y=NC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
ax[2].bar(OC_clusters_unscaled.index.astype(str),OC_clusters_unscaled.values,color='g')
ax[2].set_ylabel('O/C')
ax[2].set_ylim(None,OC_scale_max)
ax[2].axhline(y=OC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
ax[3].bar(SC_clusters_unscaled.index.astype(str),SC_clusters_unscaled.values,color='r')
ax[3].set_ylabel('S/C')
ax[3].set_ylim(None,SC_scale_max)
ax[3].axhline(y=SC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
plt.suptitle('Unscaled data, 4 clusters')
plt.tight_layout()
plt.show()

fig,ax = plt.subplots(2,2,figsize=(10,10))
ax = ax.ravel()
ax[0].bar(HC_clusters_minmax.index.astype(str),HC_clusters_minmax.values,color='gray')
ax[0].set_ylabel('H/C')
ax[0].set_ylabel('H/C')
ax[0].set_ylim(None,HC_scale_max)
ax[0].axhline(y=HC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
ax[1].bar(NC_clusters_minmax.index.astype(str),NC_clusters_minmax.values)
ax[1].set_ylabel('N/C')
ax[1].set_ylim(None,NC_scale_max)
ax[1].axhline(y=NC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
ax[2].bar(OC_clusters_minmax.index.astype(str),OC_clusters_minmax.values,color='g')
ax[2].set_ylabel('O/C')
ax[2].set_ylim(None,OC_scale_max)
ax[2].axhline(y=OC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
ax[3].bar(SC_clusters_minmax.index.astype(str),SC_clusters_minmax.values,color='r')
ax[3].set_ylabel('S/C')
ax[3].set_ylim(None,SC_scale_max)
ax[3].axhline(y=SC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
plt.suptitle('MinMax data, 6 clusters')
plt.tight_layout()
plt.show()

fig,ax = plt.subplots(2,2,figsize=(10,10))
ax = ax.ravel()
ax[0].bar(HC_clusters_qt.index.astype(str),HC_clusters_qt.values,color='gray')
ax[0].set_ylabel('H/C')
ax[0].set_ylabel('H/C')
ax[0].set_ylim(None,HC_scale_max)
ax[0].axhline(y=HC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
ax[1].bar(NC_clusters_qt.index.astype(str),NC_clusters_qt.values)
ax[1].set_ylabel('N/C')
ax[1].set_ylim(None,NC_scale_max)
ax[1].axhline(y=NC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
ax[2].bar(OC_clusters_qt.index.astype(str),OC_clusters_qt.values,color='g')
ax[2].set_ylabel('O/C')
ax[2].set_ylim(None,OC_scale_max)
ax[2].axhline(y=OC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
ax[3].bar(SC_clusters_qt.index.astype(str),SC_clusters_qt.values,color='r')
ax[3].set_ylabel('S/C')
ax[3].set_ylim(None,SC_scale_max)
ax[3].axhline(y=SC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
plt.suptitle('QT data, 6 clusters')
plt.tight_layout()
plt.show()

fig,ax = plt.subplots(2,2,figsize=(10,10))
ax = ax.ravel()
ax[0].bar(HC_clusters_signoise.index.astype(str),HC_clusters_signoise.values,color='gray')
ax[0].set_ylabel('H/C')
ax[0].set_ylabel('H/C')
ax[0].set_ylim(None,HC_scale_max)
ax[0].axhline(y=HC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
ax[1].bar(NC_clusters_signoise.index.astype(str),NC_clusters_signoise.values)
ax[1].set_ylabel('N/C')
ax[1].set_ylim(None,NC_scale_max)
ax[1].axhline(y=NC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
ax[2].bar(OC_clusters_signoise.index.astype(str),OC_clusters_signoise.values,color='g')
ax[2].set_ylabel('O/C')
ax[2].set_ylim(None,OC_scale_max)
ax[2].axhline(y=OC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
ax[3].bar(SC_clusters_signoise.index.astype(str),SC_clusters_signoise.values,color='r')
ax[3].set_ylabel('S/C')
ax[3].set_ylim(None,SC_scale_max)
ax[3].axhline(y=SC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
plt.suptitle('Sig/Noise data, 6 clusters')
plt.tight_layout()
plt.show()

sns.reset_orig()
    
    
