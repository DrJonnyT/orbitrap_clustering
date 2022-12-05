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


from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.cluster import AgglomerativeClustering


import seaborn as sns


import os
os.chdir('C:/Work/Python/Github/Orbitrap_clustering')
from ae_functions import *
from orbitrap_functions import load_pre_PMF_data

from functions.combine_multiindex import combine_multiindex
from functions.optimal_nclusters_r_card import optimal_nclusters_r_card
from functions.avg_array_clusters import avg_array_clusters
from file_loaders.load_beijingdelhi_merge import load_beijingdelhi_merge
from functions.delhi_beijing_datetime_cat import delhi_beijing_datetime_cat
from chem.chemform import ChemForm
from plotting.beijingdelhi import plot_all_cluster_tseries_BeijingDelhi, plot_cluster_heatmap_BeijingDelhi

from file_loaders.load_pre_PMF_data import load_pre_PMF_data



#%%Load data from HDF
filepath = r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\PMF data\ORBITRAP_Data_Pre_PMF.h5"
df_all_data, df_all_err, ds_all_mz = load_pre_PMF_data(filepath,join='inner')

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

#Now make sure the samples line but for the times and the previously-loaded data
df_all_times.set_index(df_all_times['date_mid'],inplace=True)
fuzzy_index = pd.merge_asof(pd.DataFrame(index=df_all_data.index),df_all_times,left_index=True,right_index=True,direction='nearest',tolerance=pd.Timedelta(hours=1.25))
df_all_times = df_all_times.loc[fuzzy_index['date_mid']]
#Fix the timestamp in the previously-loaded data
df_all_data.index = df_all_times.index
df_all_err.index = df_all_err.index


ds_dataset_cat = delhi_beijing_datetime_cat(df_all_data.index)


time_cat = delhi_calc_time_cat(df_all_times)
df_time_cat = pd.DataFrame(delhi_calc_time_cat(df_all_times),columns=['time_cat'],index=df_all_times.index)
ds_time_cat = df_time_cat['time_cat']

#This is a list of peaks with Sari's description from her PMF
Sari_peaks_list = pd.read_csv(r'C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\Sari_Peaks_Sources.csv',index_col='Formula',na_filter=False)
Sari_peaks_list = Sari_peaks_list[~Sari_peaks_list.index.duplicated(keep='first')]

#%%Classify molecules into types
#CHO/CHON/CHOS/CHNOS
molecule_types = np.array(list(ChemForm(mol).classify() for mol in df_all_data.columns.get_level_values(0)))


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


#%%Optimal num clusters based on R and min cardinality
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
HC_scale_max = 1.1 * np.amax([HC_clusters_unscaled.max(),HC_clusters_minmax.max(),HC_clusters_qt.max(),HC_clusters_signoise.max()])
NC_scale_max = 1.1 * np.amax([NC_clusters_unscaled.max(),NC_clusters_minmax.max(),NC_clusters_qt.max(),NC_clusters_signoise.max()])
OC_scale_max = 1.1 * np.amax([OC_clusters_unscaled.max(),OC_clusters_minmax.max(),OC_clusters_qt.max(),OC_clusters_signoise.max()])
SC_scale_max = 1.1 * np.amax([SC_clusters_unscaled.max(),SC_clusters_minmax.max(),SC_clusters_qt.max(),SC_clusters_signoise.max()])
HC_scale_min = 0.9 * np.amin([HC_clusters_unscaled.min(),HC_clusters_minmax.min(),HC_clusters_qt.min(),HC_clusters_signoise.min()])
NC_scale_min = 0.9 * np.amin([NC_clusters_unscaled.min(),NC_clusters_minmax.min(),NC_clusters_qt.min(),NC_clusters_signoise.min()])
OC_scale_min = 0.9 * np.amin([OC_clusters_unscaled.min(),OC_clusters_minmax.min(),OC_clusters_qt.min(),OC_clusters_signoise.min()])
SC_scale_min = 0.9 * np.amin([SC_clusters_unscaled.min(),SC_clusters_minmax.min(),SC_clusters_qt.min(),SC_clusters_signoise.min()])


fig,ax = plt.subplots(2,2,figsize=(10,10))
ax = ax.ravel()
ax[0].bar(HC_clusters_unscaled.index.astype(str),HC_clusters_unscaled.values,color='gray')
ax[0].set_ylabel('H/C')
ax[0].set_ylim(HC_scale_min,HC_scale_max)
ax[0].axhline(y=HC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
ax[1].bar(NC_clusters_unscaled.index.astype(str),NC_clusters_unscaled.values)
ax[1].set_ylabel('N/C')
ax[1].set_ylim(NC_scale_min,NC_scale_max)
ax[1].axhline(y=NC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
ax[2].bar(OC_clusters_unscaled.index.astype(str),OC_clusters_unscaled.values,color='g')
ax[2].set_ylabel('O/C')
ax[2].set_ylim(OC_scale_min,OC_scale_max)
ax[2].axhline(y=OC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
ax[3].bar(SC_clusters_unscaled.index.astype(str),SC_clusters_unscaled.values,color='r')
ax[3].set_ylabel('S/C')
ax[3].set_ylim(SC_scale_min,SC_scale_max)
ax[3].axhline(y=SC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
plt.suptitle('Unscaled data, 4 clusters')
plt.tight_layout()
plt.show()

fig,ax = plt.subplots(2,2,figsize=(10,10))
ax = ax.ravel()
ax[0].bar(HC_clusters_minmax.index.astype(str),HC_clusters_minmax.values,color='gray')
ax[0].set_ylabel('H/C')
ax[0].set_ylabel('H/C')
ax[0].set_ylim(HC_scale_min,HC_scale_max)
ax[0].axhline(y=HC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
ax[1].bar(NC_clusters_minmax.index.astype(str),NC_clusters_minmax.values)
ax[1].set_ylabel('N/C')
ax[1].set_ylim(NC_scale_min,NC_scale_max)
ax[1].axhline(y=NC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
ax[2].bar(OC_clusters_minmax.index.astype(str),OC_clusters_minmax.values,color='g')
ax[2].set_ylabel('O/C')
ax[2].set_ylim(OC_scale_min,OC_scale_max)
ax[2].axhline(y=OC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
ax[3].bar(SC_clusters_minmax.index.astype(str),SC_clusters_minmax.values,color='r')
ax[3].set_ylabel('S/C')
ax[3].set_ylim(SC_scale_min,SC_scale_max)
ax[3].axhline(y=SC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
plt.suptitle('MinMax data, 6 clusters')
plt.tight_layout()
plt.show()

fig,ax = plt.subplots(2,2,figsize=(10,10))
ax = ax.ravel()
ax[0].bar(HC_clusters_qt.index.astype(str),HC_clusters_qt.values,color='gray')
ax[0].set_ylabel('H/C')
ax[0].set_ylabel('H/C')
ax[0].set_ylim(HC_scale_min,HC_scale_max)
ax[0].axhline(y=HC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
ax[1].bar(NC_clusters_qt.index.astype(str),NC_clusters_qt.values)
ax[1].set_ylabel('N/C')
ax[1].set_ylim(NC_scale_min,NC_scale_max)
ax[1].axhline(y=NC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
ax[2].bar(OC_clusters_qt.index.astype(str),OC_clusters_qt.values,color='g')
ax[2].set_ylabel('O/C')
ax[2].set_ylim(OC_scale_min,OC_scale_max)
ax[2].axhline(y=OC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
ax[3].bar(SC_clusters_qt.index.astype(str),SC_clusters_qt.values,color='r')
ax[3].set_ylabel('S/C')
ax[3].set_ylim(SC_scale_min,SC_scale_max)
ax[3].axhline(y=SC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
plt.suptitle('QT data, 6 clusters')
plt.tight_layout()
plt.show()

fig,ax = plt.subplots(2,2,figsize=(10,10))
ax = ax.ravel()
ax[0].bar(HC_clusters_signoise.index.astype(str),HC_clusters_signoise.values,color='gray')
ax[0].set_ylabel('H/C')
ax[0].set_ylabel('H/C')
ax[0].set_ylim(HC_scale_min,HC_scale_max)
ax[0].axhline(y=HC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
ax[1].bar(NC_clusters_signoise.index.astype(str),NC_clusters_signoise.values)
ax[1].set_ylabel('N/C')
ax[1].set_ylim(NC_scale_min,NC_scale_max)
ax[1].axhline(y=NC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
ax[2].bar(OC_clusters_signoise.index.astype(str),OC_clusters_signoise.values,color='g')
ax[2].set_ylabel('O/C')
ax[2].set_ylim(OC_scale_min,OC_scale_max)
ax[2].axhline(y=OC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
ax[3].bar(SC_clusters_signoise.index.astype(str),SC_clusters_signoise.values,color='r')
ax[3].set_ylabel('S/C')
ax[3].set_ylim(SC_scale_min,SC_scale_max)
ax[3].axhline(y=SC_all_avg, linestyle='--',color=(0, 0, 0, 0.85))
plt.suptitle('Sig/Noise data, 6 clusters')
plt.tight_layout()
plt.show()

sns.reset_orig()
    


#%%Load air quality data


#df_all_merge = load_beijingdelhi_merge(newindex=df_all_data.index)

df_all_merge, df_merge_beijing_winter,df_merge_beijing_summer,df_merge_delhi_summer,df_merge_delhi_autumn = load_beijingdelhi_merge(newindex=df_all_data.index)

df_all_merge['cluster_labels_unscaled'] = cluster_labels_unscaled
df_all_merge['cluster_labels_minmax'] = cluster_labels_minmax
df_all_merge['cluster_labels_qt'] = cluster_labels_qt
df_all_merge['cluster_labels_signoise'] = cluster_labels_signoise


df_all_merge_grouped = pd.concat([df_all_merge]*4).groupby(np.concatenate([cluster_labels_unscaled,cluster_labels_minmax+10,cluster_labels_qt+20,cluster_labels_signoise+30]))




#%%Plot AQ data per cluster


#Make all the y scales the same
co_scale_max = 1.1 * df_all_merge_grouped['co_ppbv'].quantile(0.75).max()
no2_scale_max = 1.1 * df_all_merge_grouped['no2_ppbv'].quantile(0.75).max()
o3_scale_max = 1.1 * df_all_merge_grouped['o3_ppbv'].quantile(0.75).max()
so2_scale_max = 1.1 * df_all_merge_grouped['so2_ppbv'].quantile(0.75).max()

tempc_scale_max = 1.1 * df_all_merge_grouped['temp_C'].quantile(0.75).max()
tempc_scale_min = 1.1 * df_all_merge_grouped['temp_C'].quantile(0.25).min()
rh_scale_max = 1.1 * df_all_merge_grouped['RH'].quantile(0.75).max()
rh_scale_min = 1.1 * df_all_merge_grouped['RH'].quantile(0.25).min()

limits = [[0,co_scale_max],[0,no2_scale_max],[tempc_scale_min,tempc_scale_max],[0,o3_scale_max],[0,so2_scale_max],[rh_scale_min,rh_scale_max]]


sns.set_context("talk", font_scale=1)

#Unscaled data
fig,ax = plt.subplots(2,3,figsize=(10,10))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x='cluster_labels_unscaled', y="co_ppbv", data=df_all_merge,showfliers=False,color='tab:gray')
sns.boxplot(ax=ax[1], x='cluster_labels_unscaled', y="no2_ppbv", data=df_all_merge,showfliers=False,color='tab:blue')
sns.boxplot(ax=ax[3], x='cluster_labels_unscaled', y="o3_ppbv", data=df_all_merge,showfliers=False,color='tab:green')
sns.boxplot(ax=ax[4], x='cluster_labels_unscaled', y="so2_ppbv", data=df_all_merge,showfliers=False,color='tab:red')
sns.boxplot(ax=ax[2], x='cluster_labels_unscaled', y="temp_C", data=df_all_merge,showfliers=False,color='tab:olive')
sns.boxplot(ax=ax[5], x='cluster_labels_unscaled', y="RH", data=df_all_merge,showfliers=False,color='tab:cyan')
(ax[i].set_ylim(limits[i]) for i in range(len(limits)))
plt.suptitle('Unscaled data, 4 clusters')
plt.tight_layout()
plt.show()

#minmax data
fig,ax = plt.subplots(2,3,figsize=(10,10))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x='cluster_labels_minmax', y="co_ppbv", data=df_all_merge,showfliers=False,color='tab:gray')
sns.boxplot(ax=ax[1], x='cluster_labels_minmax', y="no2_ppbv", data=df_all_merge,showfliers=False,color='tab:blue')
sns.boxplot(ax=ax[3], x='cluster_labels_minmax', y="o3_ppbv", data=df_all_merge,showfliers=False,color='tab:green')
sns.boxplot(ax=ax[4], x='cluster_labels_minmax', y="so2_ppbv", data=df_all_merge,showfliers=False,color='tab:red')
sns.boxplot(ax=ax[2], x='cluster_labels_minmax', y="temp_C", data=df_all_merge,showfliers=False,color='tab:olive')
sns.boxplot(ax=ax[5], x='cluster_labels_minmax', y="RH", data=df_all_merge,showfliers=False,color='tab:cyan')
(ax[i].set_ylim(limits[i]) for i in range(len(limits)))
plt.suptitle('minmax data, 6 clusters')
plt.tight_layout()
plt.show()

#qt data
fig,ax = plt.subplots(2,3,figsize=(10,10))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x='cluster_labels_qt', y="co_ppbv", data=df_all_merge,showfliers=False,color='tab:gray')
sns.boxplot(ax=ax[1], x='cluster_labels_qt', y="no2_ppbv", data=df_all_merge,showfliers=False,color='tab:blue')
sns.boxplot(ax=ax[3], x='cluster_labels_qt', y="o3_ppbv", data=df_all_merge,showfliers=False,color='tab:green')
sns.boxplot(ax=ax[4], x='cluster_labels_qt', y="so2_ppbv", data=df_all_merge,showfliers=False,color='tab:red')
sns.boxplot(ax=ax[2], x='cluster_labels_qt', y="temp_C", data=df_all_merge,showfliers=False,color='tab:olive')
sns.boxplot(ax=ax[5], x='cluster_labels_qt', y="RH", data=df_all_merge,showfliers=False,color='tab:cyan')
(ax[i].set_ylim(limits[i]) for i in range(len(limits)))
plt.suptitle('qt data, 6 clusters')
plt.tight_layout()
plt.show()

#signoise data
fig,ax = plt.subplots(2,3,figsize=(10,10))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x='cluster_labels_signoise', y="co_ppbv", data=df_all_merge,showfliers=False,color='tab:gray')
sns.boxplot(ax=ax[1], x='cluster_labels_signoise', y="no2_ppbv", data=df_all_merge,showfliers=False,color='tab:blue')
sns.boxplot(ax=ax[3], x='cluster_labels_signoise', y="o3_ppbv", data=df_all_merge,showfliers=False,color='tab:green')
sns.boxplot(ax=ax[4], x='cluster_labels_signoise', y="so2_ppbv", data=df_all_merge,showfliers=False,color='tab:red')
sns.boxplot(ax=ax[2], x='cluster_labels_signoise', y="temp_C", data=df_all_merge,showfliers=False,color='tab:olive')
sns.boxplot(ax=ax[5], x='cluster_labels_signoise', y="RH", data=df_all_merge,showfliers=False,color='tab:cyan')
(ax[i].set_ylim(limits[i]) for i in range(len(limits)))
plt.suptitle('signoise data, 6 clusters')
plt.tight_layout()
plt.show()


sns.reset_orig()


#%%Now the same for the AMS data

#Make all the y scales the same
scale=1.1
limits = [
    [0,scale*df_all_merge_grouped['AMS_NH4'].quantile(0.75).max()],
    [0,scale*df_all_merge_grouped['AMS_NO3'].quantile(0.75).max()],
    [0,scale*df_all_merge_grouped['AMS_Chl'].quantile(0.75).max()],
    [0,scale*df_all_merge_grouped['AMS_Org'].quantile(0.75).max()],
    [0,scale*df_all_merge_grouped['AMS_SO4'].quantile(0.75).max()]]
    

sns.set_context("talk", font_scale=1)

#Unscaled data
fig,ax = plt.subplots(2,3,figsize=(10,10))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x='cluster_labels_unscaled', y="AMS_NH4", data=df_all_merge,showfliers=False,color='tab:orange')
sns.boxplot(ax=ax[1], x='cluster_labels_unscaled', y="AMS_NO3", data=df_all_merge,showfliers=False,color='tab:blue')
sns.boxplot(ax=ax[2], x='cluster_labels_unscaled', y="AMS_Chl", data=df_all_merge,showfliers=False,color='tab:pink')
sns.boxplot(ax=ax[3], x='cluster_labels_unscaled', y="AMS_Org", data=df_all_merge,showfliers=False,color='tab:green')
sns.boxplot(ax=ax[4], x='cluster_labels_unscaled', y="AMS_SO4", data=df_all_merge,showfliers=False,color='tab:red')
(ax[i].set_ylim(limits[i]) for i in range(len(limits)))
plt.suptitle('Unscaled data, 4 clusters')
plt.tight_layout()
plt.show()

#minmax data
fig,ax = plt.subplots(2,3,figsize=(10,10))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x='cluster_labels_minmax', y="AMS_NH4", data=df_all_merge,showfliers=False,color='tab:orange')
sns.boxplot(ax=ax[1], x='cluster_labels_minmax', y="AMS_NO3", data=df_all_merge,showfliers=False,color='tab:blue')
sns.boxplot(ax=ax[2], x='cluster_labels_minmax', y="AMS_Chl", data=df_all_merge,showfliers=False,color='tab:pink')
sns.boxplot(ax=ax[3], x='cluster_labels_minmax', y="AMS_Org", data=df_all_merge,showfliers=False,color='tab:green')
sns.boxplot(ax=ax[4], x='cluster_labels_minmax', y="AMS_SO4", data=df_all_merge,showfliers=False,color='tab:red')
(ax[i].set_ylim(limits[i]) for i in range(len(limits)))
plt.suptitle('minmax data, 6 clusters')
plt.tight_layout()
plt.show()

#qt data
fig,ax = plt.subplots(2,3,figsize=(10,10))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x='cluster_labels_qt', y="AMS_NH4", data=df_all_merge,showfliers=False,color='tab:orange')
sns.boxplot(ax=ax[1], x='cluster_labels_qt', y="AMS_NO3", data=df_all_merge,showfliers=False,color='tab:blue')
sns.boxplot(ax=ax[2], x='cluster_labels_qt', y="AMS_Chl", data=df_all_merge,showfliers=False,color='tab:pink')
sns.boxplot(ax=ax[3], x='cluster_labels_qt', y="AMS_Org", data=df_all_merge,showfliers=False,color='tab:green')
sns.boxplot(ax=ax[4], x='cluster_labels_qt', y="AMS_SO4", data=df_all_merge,showfliers=False,color='tab:red')
(ax[i].set_ylim(limits[i]) for i in range(len(limits)))
plt.suptitle('qt data, 6 clusters')
plt.tight_layout()
plt.show()

#signoise data
fig,ax = plt.subplots(2,3,figsize=(10,10))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x='cluster_labels_signoise', y="AMS_NH4", data=df_all_merge,showfliers=False,color='tab:orange')
sns.boxplot(ax=ax[1], x='cluster_labels_signoise', y="AMS_NO3", data=df_all_merge,showfliers=False,color='tab:blue')
sns.boxplot(ax=ax[2], x='cluster_labels_signoise', y="AMS_Chl", data=df_all_merge,showfliers=False,color='tab:pink')
sns.boxplot(ax=ax[3], x='cluster_labels_signoise', y="AMS_Org", data=df_all_merge,showfliers=False,color='tab:green')
sns.boxplot(ax=ax[4], x='cluster_labels_signoise', y="AMS_SO4", data=df_all_merge,showfliers=False,color='tab:red')
(ax[i].set_ylim(limits[i]) for i in range(len(limits)))
plt.suptitle('signoise data, 6 clusters')
plt.tight_layout()
plt.show()

sns.reset_orig()



#%%Classify by molecule type
df_all_data_classified = pd.DataFrame()

for mol in np.unique(molecule_types):
    df_all_data_classified[mol] = df_all_data.loc[:,molecule_types==mol].sum(axis=1)
    
#Unscaled data
fig,ax = plt.subplots(2,3,figsize=(10,10))
ax = ax.ravel()
sns.pieplot(ax=ax[0], y="CHO", data=df_all_data_classified,color='tab:orange')
sns.hist(ax=ax[1], x='cluster_labels_unscaled', y="CHN", data=df_all_data_classified,showfliers=False,color='tab:blue')
sns.hist(ax=ax[2], x='cluster_labels_unscaled', y="CHON", data=df_all_data_classified,showfliers=False,color='tab:pink')
sns.hist(ax=ax[3], x='cluster_labels_unscaled', y="CHOS", data=df_all_data_classified,showfliers=False,color='tab:green')
sns.hist(ax=ax[4], x='cluster_labels_unscaled', y="CHNOS", data=df_all_data_classified,showfliers=False,color='tab:red')
(ax[i].set_ylim(limits[i]) for i in range(len(limits)))
plt.suptitle('Unscaled data, 4 clusters')
plt.tight_layout()
plt.show()

df_all_data_classified[cluster_labels_unscaled==2].sum().clip(lower=0).plot.pie()


##TESTING THIS
plot_cluster_heatmap_BeijingDelhi(c,df_all_times,'Unscaled cluster heatmap',ylabel='Unscaled label')




nrows = 3
ncols = 5
Z = np.arange(nrows * ncols).reshape(nrows, ncols)
x = np.arange(ncols + 1)
y = np.arange(nrows + 1)

fig, ax = plt.subplots()
ax.pcolormesh(x, y, Z, shading='flat', vmin=Z.min(), vmax=Z.max())