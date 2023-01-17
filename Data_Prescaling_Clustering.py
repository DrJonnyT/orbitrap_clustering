# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:45:01 2022

@author: mbcx5jt5
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import scipy

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances, silhouette_score

import seaborn as sns
import pdb

import os
os.chdir('C:/Work/Python/Github/Orbitrap_clustering')


from orbitrap_functions import count_cluster_labels_from_mtx, cluster_extract_peaks
from orbitrap_functions import average_cluster_profiles, calc_cluster_elemental_ratios, plot_cluster_profile_corrs, count_clusters_project_time
from orbitrap_functions import plot_cluster_elemental_ratios

from functions.combine_multiindex import combine_multiindex
from functions.optimal_nclusters_r_card import optimal_nclusters_r_card
from functions.avg_array_clusters import avg_array_clusters
from functions.math import normdot, normdot_1min, num_frac_above_val
from file_loaders.load_beijingdelhi_merge import load_beijingdelhi_merge
from functions.delhi_beijing_datetime_cat import delhi_beijing_datetime_cat, delhi_calc_time_cat, calc_daylight_hours_BeijingDelhi, calc_daylight_deltat
from chem import ChemForm
from plotting.beijingdelhi import plot_all_cluster_tseries_BeijingDelhi, plot_cluster_heatmap_BeijingDelhi, plot_n_cluster_heatmaps_BeijingDelhi
from plotting.plot_cluster_count_hists import plot_cluster_count_hists

from file_loaders.load_pre_PMF_data import load_pre_PMF_data

from clustering.molecule_type_math import molecule_type_pos_frac_clusters_mtx
from clustering.cluster_n_times import cluster_n_times, cluster_n_times_fn
from clustering.correlate_cluster_profiles import correlate_cluster_profiles
from plotting.compare_cluster_metrics import compare_cluster_metrics, compare_cluster_metrics_fn
from plotting.plot_binned_mol_data import bin_mol_data_for_plot, plot_binned_mol_data
from plotting.plot_cluster_profiles import plot_all_cluster_profiles
from plotting.plot_cluster_aerosolomics_spectra import plot_cluster_aerosolomics_spectra


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


#Calculate daylight fraction for each filter
df_daytime = calc_daytime_frac_BeijingDelhi(df_all_times)



#This is a list of peaks with Sari's description from her PMF
Sari_peaks_list = pd.read_csv(r'C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\Sari_Peaks_Sources.csv',index_col='Formula',na_filter=False)
Sari_peaks_list = Sari_peaks_list[~Sari_peaks_list.index.duplicated(keep='first')]

#%%Classify molecules into types
#CHO/CHON/CHOS/CHNOS
molecule_types = np.array(list(ChemForm(mol).classify() for mol in df_all_data.columns.get_level_values(0)))

#Number of carbon atoms
molecule_Cx = np.array(list(ChemForm(mol).C for mol in df_all_data.columns.get_level_values(0)))

#Summed concentrations of all molecule types from ChemForm.classify()
#These sum to 1 so you can do the fraction as well
df_all_data_moltypes = df_all_data.groupby(molecule_types,axis=1).sum()
df_all_data_moltypes_frac = df_all_data_moltypes.clip(lower=0).div(df_all_data_moltypes.clip(lower=0).sum(axis=1), axis=0)

df_all_data_Cx = df_all_data.groupby(molecule_Cx,axis=1).sum()
df_all_data_Cx_frac = df_all_data_Cx.clip(lower=0).div(df_all_data_Cx.clip(lower=0).sum(axis=1), axis=0)


#Just these summed molecule types (do NOT sum to 1)
df_all_data_moltypes2 = pd.DataFrame(index = df_all_data.index)
df_all_data_moltypes2['CHOX'] = df_all_data_moltypes['CHO'] + df_all_data_moltypes['CHOS'] + df_all_data_moltypes['CHON'] + df_all_data_moltypes['CHONS']
df_all_data_moltypes2['CHNX'] = df_all_data_moltypes['CHN'] + df_all_data_moltypes['CHON'] + df_all_data_moltypes['CHONS'] + df_all_data_moltypes['CHNS']
df_all_data_moltypes2['CHSX'] = df_all_data_moltypes['CHOS'] + df_all_data_moltypes['CHS'] + df_all_data_moltypes['CHONS'] + df_all_data_moltypes['CHNS']




#%%Classify molecules based on Aerosolomics dataset
aerosolomics_path = "C:/Work/Orbitrap/data/Aerosolomics"
import glob
csv_files = glob.glob(aerosolomics_path + "/*.csv")

# Read each CSV file into DataFrame
# This creates a generator for dataframes in that folder
cols = ['CompoundName','ChemicalFormula','ExtractedMass','RT']
   
gen_df = (pd.read_csv(file,header=2,usecols=cols) for file in csv_files)
df_aerosolomics = pd.concat(gen_df, ignore_index=True)

#Source is just the first 2 or 3 letters, the molecule input in the chamber
df_aerosolomics['source'] = df_aerosolomics['CompoundName'].str.split('_').str[0]
df_aerosolomics = df_aerosolomics.set_index("ChemicalFormula")

#All possible sources from the Aerosolomics dataset
ds_aerosolomics_sources = df_aerosolomics['source'].unique()

df_aerosolomics_gbmol = df_aerosolomics.groupby(df_aerosolomics.index)

#This is now a series, index is molecular formula, data is a string of the molecules that 
#could be the source of that molecule
ds_aerosolomics_mol_sources = df_aerosolomics_gbmol['source'].unique().apply(lambda x: ';'.join(x))

#A series with index of molecular formula, same length as our data, with data as possible sources
ds_mol_aerosolomics = ds_aerosolomics_mol_sources.reindex(df_all_data.columns.get_level_values(0)).fillna('')

#The same but with no duplicates
ds_mol_aerosolomics_nodup = ds_mol_aerosolomics[~ds_mol_aerosolomics.index.duplicated(keep='first')]


#%%Sum molecules based on Aerosolomics sources
#Pick out columns that contain each of the sources

#The sums of known markers. NB these do not sum to the total of all aerosol as many markers can come from multiple sources
df_all_aerosolomics = pd.DataFrame(index=df_all_data.index)
#Unique markers, ie they appear in only one source
df_all_aerosolomics_uniquem = pd.DataFrame(index=df_all_data.index)

for source in ds_aerosolomics_sources:
    df_all_aerosolomics[source] = df_all_data.iloc[:,ds_mol_aerosolomics.str.contains(source).to_numpy()].sum(axis=1)
    df_all_aerosolomics_uniquem[source] = df_all_data.iloc[:,(ds_mol_aerosolomics.str.contains(source) * ~ds_mol_aerosolomics.str.contains(";") ).to_numpy()].sum(axis=1)

df_all_aerosolomics['Unknown'] = df_all_data.iloc[:,(ds_mol_aerosolomics == '').to_numpy()].sum(axis=1)

#ds_aerosolomics_unknown = df_all_data.iloc[:,(ds_mol_aerosolomics == '').to_numpy()].sum(axis=1)

ds_aerosolomics_known = df_all_data.iloc[:,~(ds_mol_aerosolomics == '').to_numpy()].sum(axis=1)



#Need to make it so we calculate the clusters that are particularly rich or not in unique markers for each source
#get the list of ds_mol_aerosolomics with no semicolons


# #%%Plot hist of unique markers
# fig,ax = plt.subplots(3,3)
# ax = ax.ravel()

# for num,source in enumerate(df_all_aerosolomics_uniquem.columns):
#     df_all_aerosolomics_uniquem[source].plot.hist(ax=ax[num])



#%%Work out O:C, H:C, S:C, N:C ratios for all peaks
#df_element_ratios = df_all_data.columns.get_level_values(0).to_series().apply(lambda x: chemform_ratios(x)[0])
df_element_ratios = pd.DataFrame()
df_element_ratios['H/C'] = df_all_data.columns.get_level_values(0).to_series().apply(lambda x: ChemForm(x).ratios()[0])
df_element_ratios['N/C'] = df_all_data.columns.get_level_values(0).to_series().apply(lambda x: ChemForm(x).ratios()[1])
df_element_ratios['O/C'] = df_all_data.columns.get_level_values(0).to_series().apply(lambda x: ChemForm(x).ratios()[2])
df_element_ratios['S/C'] = df_all_data.columns.get_level_values(0).to_series().apply(lambda x: ChemForm(x).ratios()[3])

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

# #MinMax transformer
# minmax = MinMaxScaler()
# df_all_minmax = pd.DataFrame(minmax.fit_transform(df_all_data.to_numpy()),index=df_all_data.index,columns=df_all_data.columns)

# compare_cluster_metrics(df_all_data,2,12,cluster_type='agglom',suptitle_prefix='Unscaled data', suptitle_suffix='')

# compare_cluster_metrics(df_qt,2,12,cluster_type='agglom',suptitle_prefix='Quantile transformed data', suptitle_suffix='')

# compare_cluster_metrics(df_minmax,2,12,cluster_type='agglom',suptitle_prefix='MinMax scaled data', suptitle_suffix='')

# compare_cluster_metrics(df_all_signoise,2,12,cluster_type='agglom',suptitle_prefix='Sig/noise data', suptitle_suffix='')

#Normalised so sum of every sample is 1
df_all_data_norm = df_all_data.divide(df_all_data.sum(axis=1),axis=0)


#%%Plot average mass spec for each dataset and all datasets
sns.set_context("talk", font_scale=1)
fig,ax = plt.subplots(5,1,figsize=(10,14),sharex=True,sharey=True)
ax = ax.ravel()

ax[0].stem(ds_all_mz,df_all_data_norm.to_numpy()[ds_dataset_cat == 'Beijing_winter'].mean(axis=0),markerfmt=' ')
ax[0].set_xlim(left=100,right=400)
ax[0].set_ylabel(r"µg m$^{-3}$")
ax[0].set_title('Beijing winter')

ax[1].stem(ds_all_mz,df_all_data_norm.to_numpy()[ds_dataset_cat == 'Beijing_summer'].mean(axis=0),markerfmt=' ')
ax[1].set_ylabel(r"µg m$^{-3}$")
ax[1].set_title('Beijing summer')

ax[2].stem(ds_all_mz,df_all_data_norm.to_numpy()[ds_dataset_cat == 'Delhi_summer'].mean(axis=0),markerfmt=' ')
ax[2].set_ylabel(r"µg m$^{-3}$")
ax[2].set_title('Delhi summer')

ax[3].stem(ds_all_mz,df_all_data_norm.to_numpy()[ds_dataset_cat == 'Delhi_autumn'].mean(axis=0),markerfmt=' ')
ax[3].set_ylabel(r"µg m$^{-3}$")
ax[3].set_title('Delhi autumn')

ax[4].stem(ds_all_mz,df_all_data_norm.to_numpy().mean(axis=0),markerfmt=' ')
ax[4].set_ylabel(r"µg m$^{-3}$")
ax[4].set_title('All projects')
ax[4].set_xlabel('m/z')

plt.suptitle('Average mass spectra per project')
plt.tight_layout()

sns.reset_orig()


#%%Plot average Cx frac for each dataset and all datasets
sns.set_context("talk", font_scale=1)
fig,ax = plt.subplots(5,1,figsize=(10,14),sharex=False,sharey=True)
ax = ax.ravel()

ax[0].stem(df_all_data_Cx_frac.columns,df_all_data_Cx_frac.to_numpy()[ds_dataset_cat == 'Beijing_winter'].mean(axis=0),markerfmt=' ')
ax[0].set_ylabel(r"µg m$^{-3}$")
ax[0].set_title('Beijing winter')

ax[1].stem(df_all_data_Cx_frac.columns,df_all_data_Cx_frac.to_numpy()[ds_dataset_cat == 'Beijing_summer'].mean(axis=0),markerfmt=' ')
ax[1].set_ylabel(r"µg m$^{-3}$")
ax[1].set_title('Beijing summer')

ax[2].stem(df_all_data_Cx_frac.columns,df_all_data_Cx_frac.to_numpy()[ds_dataset_cat == 'Delhi_summer'].mean(axis=0),markerfmt=' ')
ax[2].set_ylabel(r"µg m$^{-3}$")
ax[2].set_title('Delhi summer')

ax[3].stem(df_all_data_Cx_frac.columns,df_all_data_Cx_frac.to_numpy()[ds_dataset_cat == 'Delhi_autumn'].mean(axis=0),markerfmt=' ')
ax[3].set_ylabel(r"µg m$^{-3}$")
ax[3].set_title('Delhi autumn')

ax[4].stem(df_all_data_Cx_frac.columns,df_all_data_Cx_frac.to_numpy().mean(axis=0),markerfmt=' ')
ax[4].set_ylabel(r"µg m$^{-3}$")
ax[4].set_title('All projects')
ax[4].set_xlabel('m/z')

plt.suptitle('Average Cx per project')
plt.tight_layout()

sns.reset_orig()





#%%Calculate top 10 peaks for each project
n_peaks = 10
df_top_peaks_Beijing_winter = cluster_extract_peaks(df_all_data.loc[ds_dataset_cat == 'Beijing_winter'].mean(axis=0), df_all_data.T,n_peaks,dp=1,dropRT=False)
df_top_peaks_Beijing_summer = cluster_extract_peaks(df_all_data.loc[ds_dataset_cat == 'Beijing_summer'].mean(axis=0), df_all_data.T,n_peaks,dp=1,dropRT=False)
df_top_peaks_Delhi_summer = cluster_extract_peaks(df_all_data.loc[ds_dataset_cat == 'Delhi_summer'].mean(axis=0), df_all_data.T,n_peaks,dp=1,dropRT=False)
df_top_peaks_Delhi_autumn = cluster_extract_peaks(df_all_data.loc[ds_dataset_cat == 'Delhi_autumn'].mean(axis=0), df_all_data.T,n_peaks,dp=1,dropRT=False)
df_top_peaks_all = cluster_extract_peaks(df_all_data.mean(axis=0), df_all_data.T,n_peaks,dp=1,dropRT=False)

#With potential sources
list_df_toppeaks = [df_top_peaks_Beijing_winter,df_top_peaks_Beijing_summer,df_top_peaks_Delhi_summer,df_top_peaks_Delhi_autumn,df_top_peaks_all]

for df_top_peaks in list_df_toppeaks:
    df_top_peaks['source'] = ds_mol_aerosolomics_nodup.loc[df_top_peaks.index.get_level_values(0)].to_numpy()


#%%Extract the n biggest peaks from the original data and corresponding compounds from the prescaled data
n_peaks = 8
df_top_peaks_list = cluster_extract_peaks(df_all_data.mean(), df_all_data.T,n_peaks,dp=1,dropRT=False)
df_top_peaks_unscaled = df_all_data[df_top_peaks_list.index]
df_top_peaks_qt = df_all_qt[df_top_peaks_list.index]
#df_top_peaks_minmax = df_all_minmax[df_top_peaks_list.index]
#df_top_peaks_signoise = df_all_signoise[df_top_peaks_list.index]

#df_top_peaks_unscaled.columns = df_top_peaks_unscaled.columns.get_level_values(0) + ", " +  df_top_peaks_unscaled.columns.get_level_values(1).astype(str)


df_top_peaks_unscaled.columns = combine_multiindex(df_top_peaks_unscaled.columns)
df_top_peaks_qt.columns = combine_multiindex(df_top_peaks_qt.columns)
#df_top_peaks_minmax.columns = combine_multiindex(df_top_peaks_minmax.columns)
#df_top_peaks_signoise.columns = combine_multiindex(df_top_peaks_signoise.columns)
    



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


# #MinMax data
# #sns.pairplot(df_top_peaks_minmax,plot_kws=dict(marker="+", linewidth=1)).fig.suptitle("MinMax data", y=1.01,fontsize=20)
# g = sns.PairGrid(df_top_peaks_minmax,corner=True)
# g.fig.suptitle("MinMax data", y=0.95,fontsize=26)
# g.map_lower(sns.scatterplot, hue=ds_dataset_cat.cat.codes,palette = 'RdBu',linewidth=0.5,s=50)
# g.map_diag(plt.hist, color='grey',edgecolor='black', linewidth=1.2)
# g.add_legend(fontsize=26)
# g.legend.get_texts()[0].set_text('Beijing Winter') # You can also change the legend title
# g.legend.get_texts()[1].set_text('Beijing Summer')
# g.legend.get_texts()[2].set_text('Delhi Summer')
# g.legend.get_texts()[3].set_text('Delhi Autumn')
# sns.move_legend(g, "center right", bbox_to_anchor=(0.8, 0.55), title='Dataset')
# plt.setp(g.legend.get_title(), fontsize='24')
# plt.setp(g.legend.get_texts(), fontsize='24')
# plt.show()


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


# #Sig/noise data
# #sns.pairplot(df_top_peaks_signoise,plot_kws=dict(marker="+", linewidth=1)).fig.suptitle("Sig/noise data", y=1.01,fontsize=20)
# g = sns.PairGrid(df_top_peaks_signoise,corner=True)
# g.fig.suptitle("Sig/noise data", y=0.95,fontsize=26)
# g.map_lower(sns.scatterplot, hue=ds_dataset_cat.cat.codes,palette = 'RdBu',linewidth=0.5,s=50)
# g.map_diag(plt.hist, color='grey',edgecolor='black', linewidth=1.2)
# g.add_legend(fontsize=26)
# g.legend.get_texts()[0].set_text('Beijing Winter') # You can also change the legend title
# g.legend.get_texts()[1].set_text('Beijing Summer')
# g.legend.get_texts()[2].set_text('Delhi Summer')
# g.legend.get_texts()[3].set_text('Delhi Autumn')
# sns.move_legend(g, "center right", bbox_to_anchor=(0.8, 0.55), title='Dataset')
# plt.setp(g.legend.get_title(), fontsize='24')
# plt.setp(g.legend.get_texts(), fontsize='24')
# plt.show()


#Reset seaborn context so matplotlib plots are not messed up
sns.reset_orig()


#%%Cluster metrics for four different data prescaling

minclust = 2
maxclust = 20


#Naive HCA- default settings with Ward / Euclidean distance
df_cluster_labels_mtx_unscaled = cluster_n_times(df_all_data,minclust,maxclust,cluster_type='agglom')
df_cluster_counts_mtx_unscaled = count_cluster_labels_from_mtx(df_cluster_labels_mtx_unscaled)

#df_cluster_labels_mtx_minmax = cluster_n_times(df_all_minmax,2,13,min_num_clusters=2,cluster_type='agglom')
#df_cluster_counts_mtx_minmax = count_cluster_labels_from_mtx(df_cluster_labels_mtx_minmax)

#df_cluster_labels_mtx_qt = cluster_n_times(df_all_qt,2,13,cluster_type='agglom')

#QuantileTransformed data using Manhattan distance and complete linkage
arg_dict = {
    "affinity" : "manhattan",
    "linkage" : "complete"
    }
arg_dict={}
df_cluster_labels_mtx_qt = cluster_n_times_fn(df_all_qt,minclust,maxclust,arg_dict=arg_dict, sklearn_clust_fn = AgglomerativeClustering)
df_cluster_counts_mtx_qt = count_cluster_labels_from_mtx(df_cluster_labels_mtx_qt)


#Clustering based on normalised dot product
# Method to calculate distances between all sample pairs

# def normdot_1min_affinity(X):
#     return pairwise_distances(X, metric=normdot_1min)

def dot_1over(X,Y):
    return np.reciprocal(np.dot(X,Y))

# def dot_1over_affinity(X):
#     return pairwise_distances(X, metric=dot_1over)

arg_dict = {
    "affinity" : 'precomputed',
    "linkage" : "complete",
    }

distance_matrix = pairwise_distances(df_all_data, metric=normdot_1min)
df_cluster_labels_mtx_normdot = cluster_n_times_fn(distance_matrix,minclust,maxclust,arg_dict=arg_dict, sklearn_clust_fn = AgglomerativeClustering)
df_cluster_labels_mtx_normdot = df_cluster_labels_mtx_normdot.set_index(df_all_data.index)
df_cluster_counts_mtx_normdot = count_cluster_labels_from_mtx(df_cluster_labels_mtx_normdot)

# df_cluster_labels_mtx_signoise = cluster_n_times(df_all_signoise,2,10,min_num_clusters=2,cluster_type='agglom')
# df_cluster_counts_mtx_signoise = count_cluster_labels_from_mtx(df_cluster_labels_mtx_signoise)

#Correlations between unscaled data, using the different cluster labels
#This tells you about the clusters meaningfulness in the real world
df_cluster_corr_mtx_unscaled, _ = correlate_cluster_profiles(*average_cluster_profiles(df_cluster_labels_mtx_unscaled,df_all_data)[1:])
#df_cluster_corr_mtx_minmax, _ = correlate_cluster_profiles(*average_cluster_profiles(df_cluster_labels_mtx_minmax,df_all_data)[1:])
df_cluster_corr_mtx_qt, _ = correlate_cluster_profiles(*average_cluster_profiles(df_cluster_labels_mtx_qt,df_all_data)[1:])
#df_cluster_corr_mtx_signoise, _ = correlate_cluster_profiles(*average_cluster_profiles(df_cluster_labels_mtx_signoise,df_all_data)[1:])
df_cluster_corr_mtx_normdot, _ = correlate_cluster_profiles(*average_cluster_profiles(df_cluster_labels_mtx_normdot,df_all_data)[1:])


#Correlations between SCALED data, using the different cluster labels
#This tells you about the clustering itself
#df_cluster_corr_mtx_minmax_s, _ = correlate_cluster_profiles(*average_cluster_profiles(df_cluster_labels_mtx_minmax,df_all_minmax)[1:])
df_cluster_corr_mtx_qt_s, _ = correlate_cluster_profiles(*average_cluster_profiles(df_cluster_labels_mtx_qt,df_all_qt)[1:])
#df_cluster_corr_mtx_signoise_s, _ = correlate_cluster_profiles(*average_cluster_profiles(df_cluster_labels_mtx_signoise,df_all_signoise)[1:])


###Calculate Silhouette scores
Silhouette_scores_unscaled = []
Silhouette_scores_qt = []
Silhouette_scores_normdot = []

for n_clusters in df_cluster_labels_mtx_unscaled:
    labels_unscaled = df_cluster_labels_mtx_unscaled[n_clusters]
    labels_qt = df_cluster_labels_mtx_qt[n_clusters]
    labels_normdot = df_cluster_labels_mtx_normdot[n_clusters]
    
    Silhouette_scores_unscaled.append(silhouette_score(df_all_data,labels_unscaled))
    Silhouette_scores_qt.append(silhouette_score(df_all_qt,labels_qt))
    Silhouette_scores_normdot.append(silhouette_score(distance_matrix,labels_normdot,metric='precomputed'))



#%%Plot cluster metrics
sns.set_context("talk", font_scale=1)
fig,ax = plt.subplots(3,1,figsize=(9,12),sharex=True)
ax[0].plot(df_cluster_labels_mtx_unscaled.columns,df_cluster_counts_mtx_unscaled.min(axis=1),label='Naive',linewidth=2,c='k')
ax[0].plot(df_cluster_labels_mtx_unscaled.columns,df_cluster_counts_mtx_normdot.min(axis=1),label='NormDot',linewidth=2,c='tab:blue')
ax[0].plot(df_cluster_labels_mtx_qt.columns,df_cluster_counts_mtx_qt.min(axis=1),label='QT',linewidth=2,c='tab:red')
#ax[0].plot(df_cluster_labels_mtx_signoise.columns,df_cluster_counts_mtx_signoise.min(axis=1),label='Sig/noise',c='k',linewidth=2)
ax[0].set_title('Cardinality of smallest cluster')
ax[0].set_ylabel('Number of samples')
ax[0].set_xlabel('Num clusters')
ax[0].yaxis.set_major_locator(plticker.MultipleLocator(2))
ax[0].yaxis.set_minor_locator(plticker.MultipleLocator(1))
ax[0].grid(axis='y')
ax[0].set_ylim([0,20])
ax[0].label_outer()
ax[0].xaxis.set_tick_params(labelbottom=True)


ax[1].plot(df_cluster_labels_mtx_unscaled.columns,df_cluster_corr_mtx_unscaled.max(axis=1),label='Naive',linewidth=2,c='k')
ax[1].plot(df_cluster_labels_mtx_unscaled.columns,df_cluster_corr_mtx_normdot.max(axis=1),label='Normdot',linewidth=2,c='tab:blue')
ax[1].plot(df_cluster_labels_mtx_qt.columns,df_cluster_corr_mtx_qt.max(axis=1),label='QT',linewidth=2,c='tab:red')
#ax[1].plot(df_cluster_labels_mtx_signoise.columns,df_cluster_corr_mtx_signoise.max(axis=1),label='Sig/noise (unscaled)',c='k',linewidth=2)

#ax[1].plot(df_cluster_labels_mtx_minmax.columns,df_cluster_corr_mtx_minmax_s.max(axis=1),label='MinMax (scaled ms data)',linewidth=2,linestyle='--',c='tab:blue')
#ax[1].plot(df_cluster_labels_mtx_qt.columns,df_cluster_corr_mtx_qt_s.max(axis=1),label='QT (scaled ms data)',linewidth=2,linestyle='--',c='tab:red')
#ax[1].plot(df_cluster_labels_mtx_signoise.columns,df_cluster_corr_mtx_signoise_s.max(axis=1),label='Sig/noise (scaled)',c='k',linewidth=2,linestyle='--')

ax[1].legend(title='Cluster labels',framealpha=1.,loc='center right',bbox_to_anchor=(1.5, 0.5))
ax[1].set_ylabel('Correlation')
ax[1].set_title('Max normdot between mean unscaled cluster profiles')
ax[1].set_ylim(0.7)
ax[1].yaxis.set_major_locator(plticker.MultipleLocator(0.05))
ax[1].yaxis.set_minor_locator(plticker.MultipleLocator(0.025))

ax[1].grid(axis='y')
ax[1].xaxis.set_major_locator(plticker.MaxNLocator(integer=True))
ax[1].xaxis.set_minor_locator(plticker.MultipleLocator(1))

ax[1].label_outer()
ax[1].xaxis.set_tick_params(labelbottom=True)


ax[2].plot(df_cluster_labels_mtx_unscaled.columns,Silhouette_scores_unscaled,label='Naive',linewidth=2,c='k')
ax[2].plot(df_cluster_labels_mtx_qt.columns,Silhouette_scores_normdot,label='normdot',linewidth=2,c='tab:blue')
#ax2t = ax[2].twinx()
#ax2t.plot(df_cluster_labels_mtx_unscaled.columns,Silhouette_scores_qt,label='QT',linewidth=2,c='tab:red')
ax[2].plot(df_cluster_labels_mtx_unscaled.columns,Silhouette_scores_qt,label='QT',linewidth=2,c='tab:red')
ax[2].set_title('Silhouette score in data used for clustering')
ax[2].set_ylabel('Silhouette score')
ax[2].set_xlabel('Num clusters')
ax[2].grid(axis='y')
ax[2].set_ylim(0.13,0.6)
ax[2].yaxis.set_major_locator(plticker.MultipleLocator(0.1))
ax[2].yaxis.set_minor_locator(plticker.MultipleLocator(0.05))

#ax2t.set_yticks(np.arange(0.13,0.23,0.02))


#ax2t.yaxis.set_minor_locator(plticker.MultipleLocator(0.025))
#ax2t.set_ylabel('Silhouette score (QT data)')
#ax2t.set_ylim(0.14,0.21)

fig.suptitle('Cluster cardinality and similarity')

plt.tight_layout()
plt.show()
sns.reset_orig()

#%%compare cluster metrics

# sns.set_context("notebook",font_scale=1.2)
# compare_cluster_metrics_fn(df_all_data,df_cluster_labels_mtx_unscaled,suptitle='Naive clustering metrics')



# ##QT data
# compare_cluster_metrics_fn(df_all_qt,df_cluster_labels_mtx_qt,suptitle='QT clustering metrics')

# ##Normdot data
# arg_dict = {
#     "affinity" : 'precomputed',
#     "linkage" : "complete",
#     }
# compare_cluster_metrics_fn(df_all_data,df_cluster_labels_mtx_qt,suptitle='Normdot clustering metrics')

# sns.reset_orig()

#%%Plot stacked bar charts of cluster counts
sns.set_context("talk", font_scale=1)
plot_cluster_count_hists(df_cluster_counts_mtx_unscaled,df_cluster_counts_mtx_normdot,df_cluster_counts_mtx_qt,
                         titles=['Unscaled','Normdot','QuantileTransformer'],
                         colors=['tab:gray','tab:blue','tab:red'])

sns.reset_orig()







#%%Plot cluster number fraction above median


#Just plot up to a max of 10 clusters now

#Bin the data to find the mean of each molecule type, per cluster, per num_clusters
df_mol_data_forplot_unscaled = bin_mol_data_for_plot(df_cluster_labels_mtx_unscaled.loc[:,2:10],df_all_data_moltypes)
df_mol_data_forplot_qt = bin_mol_data_for_plot(df_cluster_labels_mtx_qt.loc[:,2:10],df_all_data_moltypes)
#df_mol_data_forplot_qt_scaled = bin_mol_data_for_plot(df_cluster_labels_mtx_qt,df_all_data_moltypes_qt)
df_mol_data_forplot_normdot = bin_mol_data_for_plot(df_cluster_labels_mtx_normdot.loc[:,2:10],df_all_data_moltypes)


#Calculate data as a fraction of the total 
df_mol_data_forplot_unscaled_frac = df_mol_data_forplot_unscaled.clip(lower=0).drop(['cluster_index','num_clusters'],axis=1)
df_mol_data_forplot_unscaled_frac = df_mol_data_forplot_unscaled_frac.div(df_mol_data_forplot_unscaled_frac.sum(axis=1,skipna=False), axis=0)
df_mol_data_forplot_unscaled_frac[['num_clusters','cluster_index']] = df_mol_data_forplot_unscaled[['num_clusters','cluster_index']]

df_mol_data_forplot_qt_frac = df_mol_data_forplot_qt.clip(lower=0).drop(['cluster_index','num_clusters'],axis=1)
df_mol_data_forplot_qt_frac = df_mol_data_forplot_qt_frac.div(df_mol_data_forplot_qt_frac.sum(axis=1,skipna=False), axis=0)
df_mol_data_forplot_qt_frac[['num_clusters','cluster_index']] = df_mol_data_forplot_qt[['num_clusters','cluster_index']]

df_mol_data_forplot_normdot_frac = df_mol_data_forplot_normdot.clip(lower=0).drop(['cluster_index','num_clusters'],axis=1)
df_mol_data_forplot_normdot_frac = df_mol_data_forplot_normdot_frac.div(df_mol_data_forplot_normdot_frac.sum(axis=1,skipna=False), axis=0)
df_mol_data_forplot_normdot_frac[['num_clusters','cluster_index']] = df_mol_data_forplot_normdot[['num_clusters','cluster_index']]




#Plot the binned cluster molecule data
plot_binned_mol_data(df_mol_data_forplot_unscaled_frac,df_mol_data_forplot_qt_frac,df_mol_data_forplot_normdot_frac,
    titles=['Unscaled','QuantileTransformer','Normdot'],
    colors=['tab:gray','tab:blue','tab:red'],
    ylabels=['CHO fracion','CHON fracion','CHOS fracion','CHONS fraction'],
    vlines=[4,8,7]
    )




#%%Plot cluster heatmaps

sns.set_context("talk", font_scale=1)
plot_n_cluster_heatmaps_BeijingDelhi(df_cluster_labels_mtx_unscaled.loc[:,4:4],df_all_times,"Unscaled data, ","Cluster label")
plot_n_cluster_heatmaps_BeijingDelhi(df_cluster_labels_mtx_normdot.loc[:,8:8],df_all_times,"Normdot data, ","Cluster label")
plot_n_cluster_heatmaps_BeijingDelhi(df_cluster_labels_mtx_qt.loc[:,7:7],df_all_times,"QT data, ","Cluster label")

sns.reset_orig()




#%%Plot all cluster profile

def plot_all_cluster_profiles_workflow(df_cluster_labels_mtx,title_prefix):
    df_cluster_counts_mtx = count_cluster_labels_from_mtx(df_cluster_labels_mtx)

    df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx = calc_cluster_elemental_ratios(df_cluster_labels_mtx,df_all_data,df_element_ratios)

    cluster_profiles_mtx, cluster_profiles_mtx_norm, num_clusters_index, cluster_index = average_cluster_profiles(df_cluster_labels_mtx,df_all_data)

    df_cluster_corr_mtx, df_prevcluster_corr_mtx = correlate_cluster_profiles(cluster_profiles_mtx_norm, num_clusters_index, cluster_index)

    plot_all_cluster_profiles(df_all_data,cluster_profiles_mtx_norm,num_clusters_index,ds_all_mz,df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,
                              df_cluster_corr_mtx,df_prevcluster_corr_mtx,df_cluster_counts_mtx,Sari_peaks_list,title_prefix=title_prefix)
          
    df_clust_cat_counts, df_cat_clust_counts, df_clust_time_cat_counts,df_time_cat_clust_counts = count_clusters_project_time(
        df_cluster_labels_mtx,ds_dataset_cat,ds_time_cat,title_prefix=title_prefix)


plot_all_cluster_profiles_workflow(df_cluster_labels_mtx_unscaled.loc[:,4:4],'Unscaled data, ')
#plot_all_cluster_profiles_workflow(df_cluster_labels_mtx_unscaled.loc[:,8:8],'Unscaled data (absolute max nclusters), ')
plot_all_cluster_profiles_workflow(df_cluster_labels_mtx_qt.loc[:,7:7],'QT data, ')
plot_all_cluster_profiles_workflow(df_cluster_labels_mtx_normdot.loc[:,8:8],'Normdot data, ')






cluster_labels_unscaled = df_cluster_labels_mtx_unscaled.loc[:,4:4].to_numpy().ravel()
cluster_labels_qt = df_cluster_labels_mtx_qt.loc[:,7:7].to_numpy().ravel()
cluster_labels_normdot = df_cluster_labels_mtx_normdot.loc[:,8:8].to_numpy().ravel()


#%%Plot CHO etc mols per cluster, for the accepted cluster numbers


# #Make all the y scales the same
# co_scale_max = 1.05 * df_all_merge_grouped['co_ppbv'].quantile(0.95,interpolation='lower').max()
# no2_scale_max = 1.1 * df_all_merge_grouped['no2_ppbv'].quantile(0.95,interpolation='lower').max()
# o3_scale_max = 1.1 * df_all_merge_grouped['o3_ppbv'].quantile(0.95,interpolation='lower').max()
# so2_scale_max = 1.1 * df_all_merge_grouped['so2_ppbv'].quantile(0.95,interpolation='lower').max()

# tempc_scale_max = 1.1 * df_all_merge_grouped['temp_C'].quantile(0.95,interpolation='lower').max()
# tempc_scale_min = 0.9 * df_all_merge_grouped['temp_C'].quantile(0.05,interpolation='lower').min()
# rh_scale_max = 1.1 * df_all_merge_grouped['RH'].quantile(0.95,interpolation='lower').max()
# rh_scale_min = 0.9 * df_all_merge_grouped['RH'].quantile(0.05,interpolation='lower').min()

# limits = [[0,co_scale_max],[0,no2_scale_max],[tempc_scale_min,tempc_scale_max],[0,o3_scale_max],[0,so2_scale_max],[rh_scale_min,rh_scale_max]]


#df_moltype_gb_clust_unscaled = df_all_data_moltypes_frac.groupby(cluster_labels_unscaled)






whis=[5,95]
sns.set_context("talk", font_scale=1)


#Unscaled data
#fig,ax = plt.subplots(3,4,figsize=(10,10))
fig = plt.figure(constrained_layout=True,figsize=(14,12))
subfigs = fig.subfigures(nrows=1, ncols=4)

axs = subfigs[0].subplots(nrows=3, ncols=1, sharey=True)
sns.boxplot(ax=axs[0], x=cluster_labels_unscaled, y="CHO", data=df_all_data_moltypes_frac,showfliers=False,color='tab:green',whis=whis)
sns.boxplot(ax=axs[1], x=cluster_labels_normdot, y="CHO", data=df_all_data_moltypes_frac,showfliers=False,color='tab:green',whis=whis)
sns.boxplot(ax=axs[2], x=cluster_labels_qt, y="CHO", data=df_all_data_moltypes_frac,showfliers=False,color='tab:green',whis=whis)
axs[0].set_title('CHO')
[ax.set_ylabel("") for ax in axs]
[ax.grid(axis='y',alpha=0.5) for ax in axs]
axs[0].set_ylabel("Naive clustering")
axs[1].set_ylabel("Normdot clustering")
axs[2].set_ylabel("QT clustering")

axs = subfigs[1].subplots(nrows=3, ncols=1, sharey=True)
sns.boxplot(ax=axs[0], x=cluster_labels_unscaled, y="CHON", data=df_all_data_moltypes_frac,showfliers=False,color='tab:blue',whis=whis)
sns.boxplot(ax=axs[1], x=cluster_labels_normdot, y="CHON", data=df_all_data_moltypes_frac,showfliers=False,color='tab:blue',whis=whis)
sns.boxplot(ax=axs[2], x=cluster_labels_qt, y="CHON", data=df_all_data_moltypes_frac,showfliers=False,color='tab:blue',whis=whis)
axs[0].set_title('CHON')
[ax.set_ylabel("") for ax in axs]
[ax.grid(axis='y',alpha=0.5) for ax in axs]


axs = subfigs[2].subplots(nrows=3, ncols=1, sharey=True)
sns.boxplot(ax=axs[0], x=cluster_labels_unscaled, y="CHOS", data=df_all_data_moltypes_frac,showfliers=False,color='tab:red',whis=whis)
sns.boxplot(ax=axs[1], x=cluster_labels_normdot, y="CHOS", data=df_all_data_moltypes_frac,showfliers=False,color='tab:red',whis=whis)
sns.boxplot(ax=axs[2], x=cluster_labels_qt, y="CHOS", data=df_all_data_moltypes_frac,showfliers=False,color='tab:red',whis=whis)
axs[0].set_title('CHOS')
[ax.set_ylabel("") for ax in axs]
[ax.grid(axis='y',alpha=0.5) for ax in axs]

axs = subfigs[3].subplots(nrows=3, ncols=1, sharey=True)
sns.boxplot(ax=axs[0], x=cluster_labels_unscaled, y="CHONS", data=df_all_data_moltypes_frac,showfliers=False,color='tab:gray',whis=whis)
sns.boxplot(ax=axs[1], x=cluster_labels_normdot, y="CHONS", data=df_all_data_moltypes_frac,showfliers=False,color='tab:gray',whis=whis)
sns.boxplot(ax=axs[2], x=cluster_labels_qt, y="CHONS", data=df_all_data_moltypes_frac,showfliers=False,color='tab:gray',whis=whis)
axs[0].set_title('CHONS')
[ax.set_ylabel("") for ax in axs]
[ax.grid(axis='y',alpha=0.5) for ax in axs]

#[axis.set_ylim(lim) for axis,lim in zip(ax,limits)]
#plt.suptitle('Unscaled data, 4 clusters')





plt.show()


# #qt data
# fig,ax = plt.subplots(2,3,figsize=(10,10))
# ax = ax.ravel()
# sns.boxplot(ax=ax[0], x='cluster_labels_qt', y="co_ppbv", data=df_all_merge,showfliers=False,color='tab:gray',whis=whis)
# sns.boxplot(ax=ax[1], x='cluster_labels_qt', y="no2_ppbv", data=df_all_merge,showfliers=False,color='tab:blue',whis=whis)
# sns.boxplot(ax=ax[3], x='cluster_labels_qt', y="o3_ppbv", data=df_all_merge,showfliers=False,color='tab:green',whis=whis)
# sns.boxplot(ax=ax[4], x='cluster_labels_qt', y="so2_ppbv", data=df_all_merge,showfliers=False,color='tab:red',whis=whis)
# sns.boxplot(ax=ax[2], x='cluster_labels_qt', y="temp_C", data=df_all_merge,showfliers=False,color='tab:olive',whis=whis)
# sns.boxplot(ax=ax[5], x='cluster_labels_qt', y="RH", data=df_all_merge,showfliers=False,color='tab:cyan',whis=whis)
# [axis.set_ylim(lim) for axis,lim in zip(ax,limits)]
# plt.suptitle('qt data, 7 clusters')
# plt.tight_layout()
# plt.show()

# #normdot data
# fig,ax = plt.subplots(2,3,figsize=(10,10))
# ax = ax.ravel()
# sns.boxplot(ax=ax[0], x='cluster_labels_normdot', y="co_ppbv", data=df_all_merge,showfliers=False,color='tab:gray',whis=whis)
# sns.boxplot(ax=ax[1], x='cluster_labels_normdot', y="no2_ppbv", data=df_all_merge,showfliers=False,color='tab:blue',whis=whis)
# sns.boxplot(ax=ax[3], x='cluster_labels_normdot', y="o3_ppbv", data=df_all_merge,showfliers=False,color='tab:green',whis=whis)
# sns.boxplot(ax=ax[4], x='cluster_labels_normdot', y="so2_ppbv", data=df_all_merge,showfliers=False,color='tab:red',whis=whis)
# sns.boxplot(ax=ax[2], x='cluster_labels_normdot', y="temp_C", data=df_all_merge,showfliers=False,color='tab:olive',whis=whis)
# sns.boxplot(ax=ax[5], x='cluster_labels_normdot', y="RH", data=df_all_merge,showfliers=False,color='tab:cyan',whis=whis)
# [axis.set_ylim(lim) for axis,lim in zip(ax,limits)]
# plt.suptitle('normdot data, 8 clusters')
# plt.tight_layout()
# plt.show()


sns.reset_orig()

#%%Load air quality data

df_all_merge, df_merge_beijing_winter,df_merge_beijing_summer,df_merge_delhi_summer,df_merge_delhi_autumn = load_beijingdelhi_merge(newindex=df_all_data.index)

df_all_merge['cluster_labels_unscaled'] = cluster_labels_unscaled
df_all_merge['cluster_labels_qt'] = cluster_labels_qt
df_all_merge['cluster_labels_normdot'] = cluster_labels_normdot


df_all_merge_grouped = pd.concat([df_all_merge]*3).groupby(np.concatenate([cluster_labels_unscaled,cluster_labels_qt+10,cluster_labels_normdot+50]))





#%%Plot AQ data per cluster


#Make all the y scales the same
co_scale_max = 1.05 * df_all_merge_grouped['co_ppbv'].quantile(0.95,interpolation='lower').max()
no2_scale_max = 1.1 * df_all_merge_grouped['no2_ppbv'].quantile(0.95,interpolation='lower').max()
o3_scale_max = 1.1 * df_all_merge_grouped['o3_ppbv'].quantile(0.95,interpolation='lower').max()
so2_scale_max = 1.1 * df_all_merge_grouped['so2_ppbv'].quantile(0.95,interpolation='lower').max()

tempc_scale_max = 1.1 * df_all_merge_grouped['temp_C'].quantile(0.95,interpolation='lower').max()
tempc_scale_min = 0.9 * df_all_merge_grouped['temp_C'].quantile(0.05,interpolation='lower').min()
rh_scale_max = 1.1 * df_all_merge_grouped['RH'].quantile(0.95,interpolation='lower').max()
rh_scale_min = 0.9 * df_all_merge_grouped['RH'].quantile(0.05,interpolation='lower').min()

limits = [[0,co_scale_max],[0,no2_scale_max],[tempc_scale_min,tempc_scale_max],[0,o3_scale_max],[0,so2_scale_max],[rh_scale_min,rh_scale_max]]

whis=[5,95]
sns.set_context("talk", font_scale=1)

#Unscaled data
fig,ax = plt.subplots(2,3,figsize=(10,10))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x='cluster_labels_unscaled', y="co_ppbv", data=df_all_merge,showfliers=False,color='tab:gray',whis=whis)
sns.boxplot(ax=ax[1], x='cluster_labels_unscaled', y="no2_ppbv", data=df_all_merge,showfliers=False,color='tab:blue',whis=whis)
sns.boxplot(ax=ax[3], x='cluster_labels_unscaled', y="o3_ppbv", data=df_all_merge,showfliers=False,color='tab:green',whis=whis)
sns.boxplot(ax=ax[4], x='cluster_labels_unscaled', y="so2_ppbv", data=df_all_merge,showfliers=False,color='tab:red',whis=whis)
sns.boxplot(ax=ax[2], x='cluster_labels_unscaled', y="temp_C", data=df_all_merge,showfliers=False,color='tab:olive',whis=whis)
sns.boxplot(ax=ax[5], x='cluster_labels_unscaled', y="RH", data=df_all_merge,showfliers=False,color='tab:cyan',whis=whis)
[axis.set_ylim(lim) for axis,lim in zip(ax,limits)]
plt.suptitle('Unscaled data, 4 clusters')
plt.tight_layout()
plt.show()


#qt data
fig,ax = plt.subplots(2,3,figsize=(10,10))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x='cluster_labels_qt', y="co_ppbv", data=df_all_merge,showfliers=False,color='tab:gray',whis=whis)
sns.boxplot(ax=ax[1], x='cluster_labels_qt', y="no2_ppbv", data=df_all_merge,showfliers=False,color='tab:blue',whis=whis)
sns.boxplot(ax=ax[3], x='cluster_labels_qt', y="o3_ppbv", data=df_all_merge,showfliers=False,color='tab:green',whis=whis)
sns.boxplot(ax=ax[4], x='cluster_labels_qt', y="so2_ppbv", data=df_all_merge,showfliers=False,color='tab:red',whis=whis)
sns.boxplot(ax=ax[2], x='cluster_labels_qt', y="temp_C", data=df_all_merge,showfliers=False,color='tab:olive',whis=whis)
sns.boxplot(ax=ax[5], x='cluster_labels_qt', y="RH", data=df_all_merge,showfliers=False,color='tab:cyan',whis=whis)
[axis.set_ylim(lim) for axis,lim in zip(ax,limits)]
plt.suptitle('qt data, 7 clusters')
plt.tight_layout()
plt.show()

#normdot data
fig,ax = plt.subplots(2,3,figsize=(10,10))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x='cluster_labels_normdot', y="co_ppbv", data=df_all_merge,showfliers=False,color='tab:gray',whis=whis)
sns.boxplot(ax=ax[1], x='cluster_labels_normdot', y="no2_ppbv", data=df_all_merge,showfliers=False,color='tab:blue',whis=whis)
sns.boxplot(ax=ax[3], x='cluster_labels_normdot', y="o3_ppbv", data=df_all_merge,showfliers=False,color='tab:green',whis=whis)
sns.boxplot(ax=ax[4], x='cluster_labels_normdot', y="so2_ppbv", data=df_all_merge,showfliers=False,color='tab:red',whis=whis)
sns.boxplot(ax=ax[2], x='cluster_labels_normdot', y="temp_C", data=df_all_merge,showfliers=False,color='tab:olive',whis=whis)
sns.boxplot(ax=ax[5], x='cluster_labels_normdot', y="RH", data=df_all_merge,showfliers=False,color='tab:cyan',whis=whis)
[axis.set_ylim(lim) for axis,lim in zip(ax,limits)]
plt.suptitle('normdot data, 8 clusters')
plt.tight_layout()
plt.show()


sns.reset_orig()





#%%Plot different molecules by cluster

sns.set_context("talk", font_scale=1)

whis=[5,95]
#Unscaled data
fig,ax = plt.subplots(2,2,figsize=(8,8))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x=cluster_labels_unscaled, y="CHO", data=df_all_data_moltypes,showfliers=False,color='tab:green',whis=whis)
sns.boxplot(ax=ax[1], x=cluster_labels_unscaled, y="CHON", data=df_all_data_moltypes,showfliers=False,color='tab:blue',whis=whis)
sns.boxplot(ax=ax[2], x=cluster_labels_unscaled, y="CHOS", data=df_all_data_moltypes,showfliers=False,color='tab:red',whis=whis)
sns.boxplot(ax=ax[3], x=cluster_labels_unscaled, y="CHONS", data=df_all_data_moltypes,showfliers=False,color='tab:gray',whis=whis)

#[axis.set_ylim(lim) for axis,lim in zip(ax,limits)]
plt.suptitle('Unscaled data, 4 clusters')
plt.tight_layout()
plt.show()

#Unscaled data
fig,ax = plt.subplots(2,2,figsize=(8,8))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x=cluster_labels_unscaled, y="CHO", data=df_all_data_moltypes_frac,showfliers=False,color='tab:green',whis=whis)
sns.boxplot(ax=ax[1], x=cluster_labels_unscaled, y="CHON", data=df_all_data_moltypes_frac,showfliers=False,color='tab:blue',whis=whis)
sns.boxplot(ax=ax[2], x=cluster_labels_unscaled, y="CHOS", data=df_all_data_moltypes_frac,showfliers=False,color='tab:red',whis=whis)
sns.boxplot(ax=ax[3], x=cluster_labels_unscaled, y="CHONS", data=df_all_data_moltypes_frac,showfliers=False,color='tab:gray',whis=whis)

#[axis.set_ylim(lim) for axis,lim in zip(ax,limits)]
plt.suptitle('Unscaled data fraction, 4 clusters')
plt.tight_layout()
plt.show()


#qt data
fig,ax = plt.subplots(2,2,figsize=(8,8))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x=cluster_labels_qt, y="CHO", data=df_all_data_moltypes,showfliers=False,color='tab:green',whis=whis)
sns.boxplot(ax=ax[1], x=cluster_labels_qt, y="CHON", data=df_all_data_moltypes,showfliers=False,color='tab:blue',whis=whis)
sns.boxplot(ax=ax[2], x=cluster_labels_qt, y="CHOS", data=df_all_data_moltypes,showfliers=False,color='tab:red',whis=whis)
sns.boxplot(ax=ax[3], x=cluster_labels_qt, y="CHONS", data=df_all_data_moltypes,showfliers=False,color='tab:gray',whis=whis)

#[axis.set_ylim(lim) for axis,lim in zip(ax,limits)]
plt.suptitle('qt data, 7 clusters')
plt.tight_layout()
plt.show()

#qt data
fig,ax = plt.subplots(2,2,figsize=(8,8))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x=cluster_labels_qt, y="CHO", data=df_all_data_moltypes_frac,showfliers=False,color='tab:green',whis=whis)
sns.boxplot(ax=ax[1], x=cluster_labels_qt, y="CHON", data=df_all_data_moltypes_frac,showfliers=False,color='tab:blue',whis=whis)
sns.boxplot(ax=ax[2], x=cluster_labels_qt, y="CHOS", data=df_all_data_moltypes_frac,showfliers=False,color='tab:red',whis=whis)
sns.boxplot(ax=ax[3], x=cluster_labels_qt, y="CHONS", data=df_all_data_moltypes_frac,showfliers=False,color='tab:gray',whis=whis)

#[axis.set_ylim(lim) for axis,lim in zip(ax,limits)]
plt.suptitle('qt data fraction, 7 clusters')
plt.tight_layout()
plt.show()


#normdot data
fig,ax = plt.subplots(2,2,figsize=(8,8))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x=cluster_labels_normdot, y="CHO", data=df_all_data_moltypes,showfliers=False,color='tab:green',whis=whis)
sns.boxplot(ax=ax[1], x=cluster_labels_normdot, y="CHON", data=df_all_data_moltypes,showfliers=False,color='tab:blue',whis=whis)
sns.boxplot(ax=ax[2], x=cluster_labels_normdot, y="CHOS", data=df_all_data_moltypes,showfliers=False,color='tab:red',whis=whis)
sns.boxplot(ax=ax[3], x=cluster_labels_normdot, y="CHONS", data=df_all_data_moltypes,showfliers=False,color='tab:gray',whis=whis)

#[axis.set_ylim(lim) for axis,lim in zip(ax,limits)]
plt.suptitle('normdot data, 7 clusters')
plt.tight_layout()
plt.show()

#normdot data
fig,ax = plt.subplots(2,2,figsize=(8,8))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x=cluster_labels_normdot, y="CHO", data=df_all_data_moltypes_frac,showfliers=False,color='tab:green',whis=whis)
sns.boxplot(ax=ax[1], x=cluster_labels_normdot, y="CHON", data=df_all_data_moltypes_frac,showfliers=False,color='tab:blue',whis=whis)
sns.boxplot(ax=ax[2], x=cluster_labels_normdot, y="CHOS", data=df_all_data_moltypes_frac,showfliers=False,color='tab:red',whis=whis)
sns.boxplot(ax=ax[3], x=cluster_labels_normdot, y="CHONS", data=df_all_data_moltypes_frac,showfliers=False,color='tab:gray',whis=whis)

#[axis.set_ylim(lim) for axis,lim in zip(ax,limits)]
plt.suptitle('normdot data fraction, 7 clusters')
plt.tight_layout()
plt.show()

sns.reset_orig()

    



#%%Plot clusters by Aerosolomics source


# def plot_cluster_aerosolomics(cluster_labels,df_aero_concs):
#     """
#     Box plots of aerosolomics sources, averaged for each cluster label

#     Parameters
#     ----------
#     cluster_labels : array of integers
#         Cluster labels
#     df_aero_concs : dataframe
#         Concentrations of species from the different Aerosolomics sources

#     Returns
#     -------
#     None.

#     """
#     sns.set_context("talk", font_scale=1)
#     fig,ax = plt.subplots(2,5,figsize=(14,8))
#     ax=ax.ravel()
#     whis=[5,95]
    
#     for subp, source in enumerate(df_all_aerosolomics.columns):
#         sns.boxplot(ax=ax[subp],x=cluster_labels,y=df_all_aerosolomics[source],color='tab:gray',whis=whis,showfliers=False)
#         ax[subp].set_ylabel('')
#         ax[subp].set_title(source)

#     plt.tight_layout()
#     sns.reset_orig()
    
    


plot_cluster_aerosolomics_spectra(cluster_labels_unscaled,df_all_aerosolomics_uniquem,suptitle='Naive clustering, unique tracers',avg='pct',offset_zero=True)

plot_cluster_aerosolomics_spectra(cluster_labels_normdot,df_all_aerosolomics_uniquem,suptitle='Normdot clustering, unique tracers',avg='pct',offset_zero=True)
plot_cluster_aerosolomics_spectra(cluster_labels_qt,df_all_aerosolomics_uniquem,suptitle='QT clustering, unique tracers',avg='pct',offset_zero=True)




plot_cluster_aerosolomics_spectra(cluster_labels_unscaled,df_all_aerosolomics,suptitle='Naive clustering, non-unique tracers',avg='pct',offset_zero=True)

plot_cluster_aerosolomics_spectra(cluster_labels_normdot,df_all_aerosolomics,suptitle='Normdot clustering, non-unique tracers',avg='pct',offset_zero=True)
plot_cluster_aerosolomics_spectra(cluster_labels_qt,df_all_aerosolomics,suptitle='QT clustering, non-unique tracers',avg='pct',offset_zero=True)







#%%Extract the most unusually high and low molecules for each cluster

from scipy.stats import percentileofscore

def extract_top_percentiles(df_data,cluster_labels,num,highest=True,dropRT=False):
    unique_labels = np.unique(cluster_labels)
    num_clust = len(unique_labels)
    
    df_top_pct = pd.DataFrame(columns=unique_labels)
    df_top_pct.columns.rename('cluster',inplace=True)
    
    for cluster in unique_labels:
        data_thisclust = df_data.loc[cluster_labels==cluster]
        
        ds_pct = pd.Series([percentileofscore(df_data[mol],data_thisclust[mol].median()) for mol in df_data.columns],index=df_data.columns, dtype='float')
        
        #Extract the top num peaks
        if(highest):
            ds_pct_top = ds_pct.sort_values(ascending=False).iloc[0:num]
        else:
            ds_pct_top = ds_pct.sort_values(ascending=True).iloc[0:num]
            
        if(dropRT):
            df_top_pct[cluster] = ds_pct_top.index.get_level_values(0)
        else:
            df_top_pct[cluster] = ds_pct_top.index.values
        
    return df_top_pct
        
a = extract_top_percentiles(df_all_data,cluster_labels_unscaled,30,dropRT=True)

#%%A big gap
##Everything below here is old and probably wont make it into the final script







































#%%Optimal num clusters based on R and min cardinality
nclusters_unscaled = optimal_nclusters_r_card(df_cluster_labels_mtx_unscaled.columns.to_numpy(),df_cluster_corr_mtx_unscaled.max(axis=1),
                             df_cluster_counts_mtx_unscaled.min(axis=1))

nclusters_minmax = optimal_nclusters_r_card(df_cluster_labels_mtx_minmax.columns,df_cluster_corr_mtx_minmax.max(axis=1),
                             df_cluster_counts_mtx_minmax.min(axis=1))

nclusters_qt = optimal_nclusters_r_card(df_cluster_labels_mtx_qt.columns,df_cluster_corr_mtx_qt.max(axis=1),
                             df_cluster_counts_mtx_qt.min(axis=1))

# nclusters_signoise = optimal_nclusters_r_card(df_cluster_labels_mtx_signoise.columns,df_cluster_corr_mtx_signoise.max(axis=1),
#                              df_cluster_counts_mtx_signoise.min(axis=1))

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

# nclusters_signoise = optimal_nclusters_r_card(df_cluster_labels_mtx_signoise.columns,
#                                               df_cluster_corr_mtx_signoise.max(axis=1),
#                                               df_cluster_counts_mtx_signoise.min(axis=1),mincard_threshold=1)


#%%Clustering workflow - unscaled data

df_cluster_labels_mtx = cluster_n_times(df_all_data,2,10,min_num_clusters=2,cluster_type='agglom')
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




arg_dict = {
    "criterion": "maxclust",
    "metric" : "mahalanobis",
    "method" : "ward"
}

compare_cluster_metrics_fn(df_all_data.to_numpy(),2,12,arg_dict=arg_dict,scipy_clust_fn=scipy.cluster.hierarchy.fclusterdata)






#Clustering based on normalised dot product

# Method to calculate distances between all sample pairs
from sklearn.metrics import pairwise_distances
def sim_affinity(X):
    return pairwise_distances(X, metric=inv_normdot)
    #return pairwise_distances(X, metric='manhattan')
    

def inv_normdot(X,Y):
    return 1/normdot(X,Y)

def normdot_1min(X,Y):
    return 1 - normdot(X,Y)

cluster = AgglomerativeClustering(n_clusters=5, affinity=sim_affinity, linkage='complete')
d = cluster.fit(df_all_data.to_numpy()).labels_




fig,ax=plt.subplots(1,figsize=(10,5))
ax.plot(a,label='euclidean')
ax.plot(b,label='manhattan')
ax.plot(c,label='inv_norm')
ax.plot(d,label='mahalanobis')
ax.legend()


#%%Clustering workflow - MinMax data
df_cluster_labels_mtx = cluster_n_times(df_all_minmax,2,10,min_num_clusters=2,cluster_type='agglom')
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
df_cluster_labels_mtx = cluster_n_times(df_all_qt,2,10,cluster_type='agglom')
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





#Look at the fraction of each molecule that is positive in the QT data
a, b = molecule_type_pos_frac_clusters_mtx(df_all_qt,molecule_types,df_cluster_labels_mtx)
c = a[0]





#Plot the QT-scaled data
cluster_profiles_mtx, cluster_profiles_mtx_norm, num_clusters_index, cluster_index = average_cluster_profiles(df_cluster_labels_mtx,df_all_qt)
df_cluster_corr_mtx, df_prevcluster_corr_mtx = correlate_cluster_profiles(cluster_profiles_mtx_norm, num_clusters_index, cluster_index)
plot_all_cluster_profiles(df_all_qt,cluster_profiles_mtx_norm,num_clusters_index,ds_all_mz,df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,
                          df_cluster_corr_mtx,df_prevcluster_corr_mtx,df_cluster_counts_mtx,Sari_peaks_list,title_prefix='QT data HCA, scaled data, ')



#%%Clustering workflow - Sig/noise transformed data
df_cluster_labels_mtx = cluster_n_times(df_all_signoise,2,10,min_num_clusters=2,cluster_type='agglom')
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