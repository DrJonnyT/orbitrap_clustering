# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:45:01 2022

@author: mbcx5jt5
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.transforms as mtransforms
import numpy as np
import scipy

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances, silhouette_score
from scipy.stats import percentileofscore

import string
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
from functions.delhi_beijing_time import delhi_beijing_datetime_cat, delhi_calc_time_cat, calc_daylight_hours_BeijingDelhi, calc_daylight_deltat
from chem import ChemForm
from plotting.beijingdelhi import plot_all_cluster_tseries_BeijingDelhi, plot_cluster_heatmap_BeijingDelhi, plot_n_cluster_heatmaps_BeijingDelhi
from plotting.plot_cluster_count_hists import plot_cluster_count_hists
from plotting.plot_clusters_project_daylight import plot_clusters_project_daylight
#from plotting.windroseaxessubplot import WindroseAxesSubplot

from file_loaders.load_pre_PMF_data import load_pre_PMF_data

from clustering.molecule_type_math import molecule_type_pos_frac_clusters_mtx
from clustering.cluster_n_times import cluster_n_times, cluster_n_times_fn
from clustering.correlate_cluster_profiles import correlate_cluster_profiles
from clustering.cluster_top_percentiles import cluster_top_percentiles
from plotting.compare_cluster_metrics import compare_cluster_metrics, compare_cluster_metrics_fn
from plotting.plot_binned_mol_data import bin_mol_data_for_plot, plot_binned_mol_data
from plotting.plot_cluster_profiles import plot_all_cluster_profiles
from plotting.plot_cluster_aerosolomics_spectra import plot_cluster_aerosolomics_spectra
from plotting.plot_windrose_percluster import plot_windrose_percluster


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
df_daytime_hours = calc_daylight_hours_BeijingDelhi(df_all_times)


#This is a list of peaks with Sari's description from her PMF
Sari_peaks_list = pd.read_csv(r'C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\Sari_Peaks_Sources.csv',index_col='Formula',na_filter=False)
Sari_peaks_list = Sari_peaks_list[~Sari_peaks_list.index.duplicated(keep='first')]


#%%Classify molecules into types
#CHO/CHON/CHOS/CHNOS
molecule_types = np.array(list(ChemForm(mol).classify() for mol in df_all_data.columns.get_level_values(0)))

#Number of carbon atoms
molecule_Cx = np.array(list(ChemForm(mol).C for mol in df_all_data.columns.get_level_values(0)))

#Summed concentrations of all molecule types from ChemForm.classify()
df_all_data_moltypes = df_all_data.groupby(molecule_types,axis=1).sum()
df_all_data_moltypes_frac = df_all_data_moltypes.clip(lower=0).div(df_all_data_moltypes.clip(lower=0).sum(axis=1), axis=0)

#Concenctrations as a percentile of the whole distribution from all datasets
df_all_data_moltypes_pct = pd.DataFrame(index=df_all_data_moltypes.index,columns=df_all_data_moltypes.columns)
for mol in df_all_data_moltypes.columns:
    df_all_data_moltypes_pct[mol] = [percentileofscore(df_all_data_moltypes[mol],df_all_data_moltypes[mol].loc[time]) for time in df_all_data_moltypes.index]



#DATA BY CARBON NUMBER
df_all_data_Cx = df_all_data.groupby(molecule_Cx,axis=1).sum()
df_all_data_Cx_frac = df_all_data_Cx.clip(lower=0).div(df_all_data_Cx.clip(lower=0).sum(axis=1), axis=0)
#Concenctrations as a percentile of the whole distribution from all datasets
df_all_data_Cx_pct = pd.DataFrame(index=df_all_data_Cx.index,columns=df_all_data_Cx.columns)
for mol in df_all_data_Cx.columns:
    df_all_data_Cx_pct[mol] = [percentileofscore(df_all_data_Cx[mol],df_all_data_Cx[mol].loc[time]) for time in df_all_data_Cx.index]



#Just these summed molecule types (do NOT sum to 1)
df_all_data_moltypes2 = pd.DataFrame(index = df_all_data.index)
df_all_data_moltypes2['CHOX'] = df_all_data_moltypes['CHO'] + df_all_data_moltypes['CHOS'] + df_all_data_moltypes['CHON'] + df_all_data_moltypes['CHONS']
df_all_data_moltypes2['CHNX'] = df_all_data_moltypes['CHN'] + df_all_data_moltypes['CHON'] + df_all_data_moltypes['CHONS'] + df_all_data_moltypes['CHNS']
df_all_data_moltypes2['CHSX'] = df_all_data_moltypes['CHOS'] + df_all_data_moltypes['CHS'] + df_all_data_moltypes['CHONS'] + df_all_data_moltypes['CHNS']

#Concenctrations as a percentile of the whole distribution from all datasets
df_all_data_moltypes2_pct = pd.DataFrame(index=df_all_data_moltypes2.index,columns=df_all_data_moltypes2.columns)
for mol in df_all_data_moltypes2.columns:
    df_all_data_moltypes2_pct[mol] = [percentileofscore(df_all_data_moltypes2[mol],df_all_data_moltypes2[mol].loc[time]) for time in df_all_data_moltypes2.index]


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


df_top_peaks_unscaled.columns = combine_multiindex(df_top_peaks_unscaled.columns,nospaces=True)
df_top_peaks_qt.columns = combine_multiindex(df_top_peaks_qt.columns,nospaces=True)
#df_top_peaks_minmax.columns = combine_multiindex(df_top_peaks_minmax.columns)
#df_top_peaks_signoise.columns = combine_multiindex(df_top_peaks_signoise.columns)
    



#%%Pairplots distributions of these n biggest peaks

#Set Seaborn context so plots have better font sizes
sns.set_context("talk", font_scale=0.9)

#Unscaled data
g = sns.PairGrid(df_top_peaks_unscaled,corner=True)
g.fig.suptitle("No prescaling", y=0.95,fontsize=26)
g.map_lower(sns.scatterplot, hue=ds_dataset_cat.cat.codes,palette = 'RdBu',linewidth=0.5,s=50)
g.map_diag(plt.hist, color='grey',edgecolor='black', linewidth=1.2)
g.add_legend(fontsize=26,frameon=True)
g.legend.get_texts()[0].set_text('Beijing winter') # You can also change the legend title
g.legend.get_texts()[1].set_text('Beijing summer')
g.legend.get_texts()[2].set_text('Delhi pre-monsoon')
g.legend.get_texts()[3].set_text('Delhi post-monsoon')
sns.move_legend(g, "center right", bbox_to_anchor=(0.8, 0.55), title='Dataset')
plt.setp(g.legend.get_title(), fontsize='20')
plt.setp(g.legend.get_texts(), fontsize='20')
plt.show()



#QT data
#sns.pairplot(df_top_peaks_qt,plot_kws=dict(marker="+", linewidth=1)).fig.suptitle("QT data", y=1.01,fontsize=20)
g = sns.PairGrid(df_top_peaks_qt,corner=True)
g.fig.suptitle("Quantile transformer prescaling", y=0.95,fontsize=26)
g.map_lower(sns.scatterplot, hue=ds_dataset_cat.cat.codes,palette = 'RdBu',linewidth=0.5,s=50)
g.map_diag(plt.hist, color='grey',edgecolor='black', linewidth=1.2)
g.add_legend(fontsize=26,frameon=True)
g.legend.get_texts()[0].set_text('Beijing winter') # You can also change the legend title
g.legend.get_texts()[1].set_text('Beijing summer')
g.legend.get_texts()[2].set_text('Delhi pre-monsoon')
g.legend.get_texts()[3].set_text('Delhi post-monsoon')
sns.move_legend(g, "center right", bbox_to_anchor=(0.8, 0.55), title='Dataset')
plt.setp(g.legend.get_title(), fontsize='20')
plt.setp(g.legend.get_texts(), fontsize='20')
plt.show()


#Reset seaborn context so matplotlib plots are not messed up
sns.reset_orig()




#%%How many compounds are the highest compound at any given time in the dataset?

df_all_max = df_all_data.idxmax(axis=1)
df_all_max_qt = df_all_qt.idxmax(axis=1)

#11 compounds
np.unique(df_all_max)
#240 compounds
np.unique(df_all_max_qt)



#Top 8 at any given time?
#107
top8_at_any_point = df_all_data.apply(lambda s: s.abs().nlargest(8).index.tolist(), axis=1)
np.unique(np.hstack(top8_at_any_point.to_numpy()).ravel()).shape
#510
top8_at_any_point_qt = df_all_qt.apply(lambda s: s.abs().nlargest(8).index.tolist(), axis=1)
np.unique(np.hstack(top8_at_any_point_qt.to_numpy()).ravel()).shape





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
trans = mtransforms.ScaledTranslation(5/72, -5/72, fig.dpi_scale_trans)

sns.set_context("talk", font_scale=0.8)
fig,ax = plt.subplots(3,1,figsize=(8,12),sharex=True)
ax[0].plot(df_cluster_labels_mtx_unscaled.columns,df_cluster_counts_mtx_unscaled.min(axis=1),label='Naive',linewidth=2,c='k')
ax[0].plot(df_cluster_labels_mtx_qt.columns,df_cluster_counts_mtx_qt.min(axis=1),label='QT',linewidth=2,c='tab:red')
ax[0].plot(df_cluster_labels_mtx_unscaled.columns,df_cluster_counts_mtx_normdot.min(axis=1),label='NormDot',linewidth=2,c='tab:blue')
#ax[0].plot(df_cluster_labels_mtx_signoise.columns,df_cluster_counts_mtx_signoise.min(axis=1),label='Sig/noise',c='k',linewidth=2)
#ax[0].set_title('Cardinality of smallest cluster')
ax[0].set_ylabel('Minimum cardinality')
ax[0].set_xlabel('Num clusters')
ax[0].yaxis.set_major_locator(plticker.MultipleLocator(2))
ax[0].yaxis.set_minor_locator(plticker.MultipleLocator(1))
ax[0].grid(axis='y')
ax[0].set_ylim([0,20])
ax[0].label_outer()
ax[0].xaxis.set_tick_params(labelbottom=True)
ax[0].text(0.0, 1.0, '(a)', transform=ax[0].transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1.0', edgecolor='none', pad=1.0))




ax[1].plot(df_cluster_labels_mtx_unscaled.columns,df_cluster_corr_mtx_unscaled.max(axis=1),label='Naive',linewidth=2,c='k')
ax[1].plot(df_cluster_labels_mtx_qt.columns,df_cluster_corr_mtx_qt.max(axis=1),label='QT',linewidth=2,c='tab:red')
ax[1].plot(df_cluster_labels_mtx_unscaled.columns,df_cluster_corr_mtx_normdot.max(axis=1),label='Normdot',linewidth=2,c='tab:blue')
ax[1].legend(title='Cluster labels',framealpha=1.,loc='center right',bbox_to_anchor=(1.5, 0.5))
ax[1].set_ylabel('Maximum similarity')
ax[1].set_ylim(0.7)
ax[1].yaxis.set_major_locator(plticker.MultipleLocator(0.05))
ax[1].yaxis.set_minor_locator(plticker.MultipleLocator(0.025))
ax[1].grid(axis='y')
ax[1].xaxis.set_major_locator(plticker.MaxNLocator(integer=True))
ax[1].xaxis.set_minor_locator(plticker.MultipleLocator(1))
ax[1].label_outer()
ax[1].xaxis.set_tick_params(labelbottom=True)
ax[1].text(0.0, 1.0, '(b)', transform=ax[1].transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1.0', edgecolor='none', pad=1.0))

ax[2].plot(df_cluster_labels_mtx_unscaled.columns,Silhouette_scores_unscaled,label='Naive',linewidth=2,c='k')
ax[2].plot(df_cluster_labels_mtx_unscaled.columns,Silhouette_scores_qt,label='QT',linewidth=2,c='tab:red')
ax[2].plot(df_cluster_labels_mtx_qt.columns,Silhouette_scores_normdot,label='normdot',linewidth=2,c='tab:blue')
ax[2].set_ylabel('Silhouette score')
ax[2].set_xlabel('Number of clusters')
ax[2].grid(axis='y')
ax[2].set_ylim(0.13,0.65)
ax[2].yaxis.set_major_locator(plticker.MultipleLocator(0.1))
ax[2].yaxis.set_minor_locator(plticker.MultipleLocator(0.05))
ax[2].text(0.0, 1.0, '(c)', transform=ax[2].transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='1.0', edgecolor='none', pad=1.0))


fig.suptitle('Cluster cardinality and similarity',x=0.4)
plt.tight_layout()
plt.show()
sns.reset_orig()


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






#%%Work out total time per cluster of daylight vs night
def calc_daynight_frac_per_cluster(cluster_labels,df_daytime_hours):
    all_labels = np.unique(cluster_labels)
    ds_frac = pd.Series(index=all_labels,dtype='float')
    ds_frac = ds_frac.fillna('nan')  
    
    for label in all_labels:
        df_cluster = df_daytime_hours.loc[cluster_labels == label]
        daylight_frac = df_cluster['daylight_hours'].sum() / df_cluster.sum().sum()
        ds_frac.loc[label] = daylight_frac
            
    return ds_frac


#%%Plot all cluster profile
def plot_all_cluster_profiles_workflow(df_cluster_labels_mtx,df_daytime_hours,title_prefix):
    df_cluster_counts_mtx = count_cluster_labels_from_mtx(df_cluster_labels_mtx)

    df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx = calc_cluster_elemental_ratios(df_cluster_labels_mtx,df_all_data,df_element_ratios)

    cluster_profiles_mtx, cluster_profiles_mtx_norm, num_clusters_index, cluster_index = average_cluster_profiles(df_cluster_labels_mtx,df_all_data)

    df_cluster_corr_mtx, df_prevcluster_corr_mtx = correlate_cluster_profiles(cluster_profiles_mtx_norm, num_clusters_index, cluster_index)

    plot_all_cluster_profiles(df_all_data,cluster_profiles_mtx_norm,num_clusters_index,ds_all_mz,df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,
                              df_cluster_corr_mtx,df_prevcluster_corr_mtx,df_cluster_counts_mtx,Sari_peaks_list,title_prefix=title_prefix)
          
    df_clust_cat_counts, df_cat_clust_counts, df_clust_time_cat_counts,df_time_cat_clust_counts = count_clusters_project_time(
        df_cluster_labels_mtx,ds_dataset_cat,ds_time_cat,title_prefix=title_prefix)
    
    
    #Work out daylight hours
    #pdb.set_trace()
    ds_day_frac = calc_daynight_frac_per_cluster(df_cluster_labels_mtx.iloc[:,0],df_daytime_hours)
    plot_clusters_project_daylight(
        df_cluster_labels_mtx,ds_dataset_cat,ds_day_frac,title_prefix=title_prefix)


plot_all_cluster_profiles_workflow(df_cluster_labels_mtx_unscaled.loc[:,4:4],df_daytime_hours,'Unscaled data, ')
#plot_all_cluster_profiles_workflow(df_cluster_labels_mtx_unscaled.loc[:,8:8],'Unscaled data (absolute max nclusters), ')


plot_all_cluster_profiles_workflow(df_cluster_labels_mtx_qt.loc[:,7:7],df_daytime_hours,'QT data, ')
plot_all_cluster_profiles_workflow(df_cluster_labels_mtx_normdot.loc[:,8:8],df_daytime_hours,'Normdot data, ')






cluster_labels_unscaled = df_cluster_labels_mtx_unscaled.loc[:,4:4].to_numpy().ravel()
cluster_labels_qt = df_cluster_labels_mtx_qt.loc[:,7:7].to_numpy().ravel()
cluster_labels_normdot = df_cluster_labels_mtx_normdot.loc[:,8:8].to_numpy().ravel()




#%%Plot cluster elemental ratios
df_HC_mtx = pd.DataFrame()
df_OC_mtx = pd.DataFrame()
df_SC_mtx = pd.DataFrame()
df_NC_mtx = pd.DataFrame()


#normdot cluster labels
df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx = calc_cluster_elemental_ratios(df_cluster_labels_mtx_normdot.loc[:,8:8],df_all_data,df_element_ratios)
df_HC_mtx['normdot'] = df_clusters_HC_mtx.loc[8]
df_OC_mtx['normdot'] = df_clusters_OC_mtx.loc[8]
df_SC_mtx['normdot'] = df_clusters_SC_mtx.loc[8]
df_NC_mtx['normdot'] = df_clusters_NC_mtx.loc[8]


#Unscaled cluster labels
df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx = calc_cluster_elemental_ratios(df_cluster_labels_mtx_unscaled.loc[:,4:4],df_all_data,df_element_ratios)
df_HC_mtx['Unscaled'] = df_clusters_HC_mtx.loc[4]
df_OC_mtx['Unscaled'] = df_clusters_OC_mtx.loc[4]
df_SC_mtx['Unscaled'] = df_clusters_SC_mtx.loc[4]
df_NC_mtx['Unscaled'] = df_clusters_NC_mtx.loc[4]

#qt cluster labels
df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx = calc_cluster_elemental_ratios(df_cluster_labels_mtx_qt.loc[:,7:7],df_all_data,df_element_ratios)
df_HC_mtx['qt'] = df_clusters_HC_mtx.loc[7]
df_OC_mtx['qt'] = df_clusters_OC_mtx.loc[7]
df_SC_mtx['qt'] = df_clusters_SC_mtx.loc[7]
df_NC_mtx['qt'] = df_clusters_NC_mtx.loc[7]


df_HC_mtx = df_HC_mtx.stack()
df_OC_mtx = df_OC_mtx.stack()
df_SC_mtx = df_SC_mtx.stack()
df_NC_mtx = df_NC_mtx.stack()




color_dict = { 'Unscaled':'black', 'normdot':'blue', 'qt':'red'}

color=[ color_dict[i] for i in df_HC_mtx.index.get_level_values(1) ]

from matplotlib.markers import MarkerStyle
text_style = dict(horizontalalignment='right', verticalalignment='center',
                  fontsize=12, fontfamily='monospace')
marker_style = dict(linestyle=':', color='0.8', markersize=10,
                    markerfacecolor="tab:blue", markeredgecolor="tab:blue")
marker_style.update(markeredgecolor="none", markersize=15)



markers="$" + df_HC_mtx.index.get_level_values(0).astype(str) + "$"




fig,ax = plt.subplots(1,3,figsize=(10,5))

#ax[0].scatter(df_HC_mtx,df_OC_mtx,c=color,marker=df_HC_mtx.index.get_level_values(0))


    # Escape dollars so that the text is written "as is", not as mathtext.
ax[0].text(df_HC_mtx, df_OC_mtx, repr(markers).replace("$", r"\$"), **text_style)
ax[0].plot(df_HC_mtx,df_OC_mtx, marker=markers.to_numpy())
#format_axes(ax)

plt.show()

            
# ds_day_frac_unscaled = calc_daynight_frac_per_cluster(cluster_labels_unscaled,df_daytime_hours)
# ds_day_frac_qt = calc_daynight_frac_per_cluster(cluster_labels_qt,df_daytime_hours)
# ds_day_frac_normdot = calc_daynight_frac_per_cluster(cluster_labels_normdot,df_daytime_hours)


#%%
fig, ax = plt.subplots()
fig.subplots_adjust(left=0.4)

marker_style.update(mec="None", markersize=15)
markers = ["$1$", r"$\frac{1}{2}$", "$f$", "$\u266B$", r"$\mathcal{A}$"]


for y, marker in enumerate(markers):
    # Escape dollars so that the text is written "as is", not as mathtext.
    ax.text(-0.5, y, repr(marker).replace("$", r"\$"), **text_style)
    ax.plot(y * points, marker=marker, **marker_style)
format_axes(ax)
fig.suptitle('mathtext markers', fontsize=14)

plt.show()


#%%Plot CHO etc mols per cluster, for the accepted cluster numbers


whis=[5,95]
sns.set_context("talk", font_scale=1)


#Unscaled data
#fig,ax = plt.subplots(3,4,figsize=(10,10))
fig = plt.figure(constrained_layout=True,figsize=(14,12))
subfigs = fig.subfigures(nrows=1, ncols=4)

axs = subfigs[0].subplots(nrows=3, ncols=1, sharey=True)
sns.boxplot(ax=axs[0], x=cluster_labels_unscaled, y="CHO", data=df_all_data_moltypes,showfliers=False,color='tab:green',whis=whis)
sns.boxplot(ax=axs[1], x=cluster_labels_normdot, y="CHO", data=df_all_data_moltypes,showfliers=False,color='tab:green',whis=whis)
sns.boxplot(ax=axs[2], x=cluster_labels_qt, y="CHO", data=df_all_data_moltypes,showfliers=False,color='tab:green',whis=whis)
axs[0].set_title('CHO')
[ax.set_ylabel("") for ax in axs]
[ax.grid(axis='y',alpha=0.5) for ax in axs]
axs[0].set_ylabel("Naive clustering \n Concentration (µg$\,$m$^{-3}$)")
axs[1].set_ylabel("Normdot clustering \n Concentration (µg$\,$m$^{-3}$)")
axs[2].set_ylabel("QT clustering \n Concentration (µg$\,$m$^{-3}$)")

axs = subfigs[1].subplots(nrows=3, ncols=1, sharey=True)
sns.boxplot(ax=axs[0], x=cluster_labels_unscaled, y="CHON", data=df_all_data_moltypes,showfliers=False,color='tab:blue',whis=whis)
sns.boxplot(ax=axs[1], x=cluster_labels_normdot, y="CHON", data=df_all_data_moltypes,showfliers=False,color='tab:blue',whis=whis)
sns.boxplot(ax=axs[2], x=cluster_labels_qt, y="CHON", data=df_all_data_moltypes,showfliers=False,color='tab:blue',whis=whis)
axs[0].set_title('CHON')
[ax.set_ylabel("") for ax in axs]
[ax.grid(axis='y',alpha=0.5) for ax in axs]


axs = subfigs[2].subplots(nrows=3, ncols=1, sharey=True)
sns.boxplot(ax=axs[0], x=cluster_labels_unscaled, y="CHOS", data=df_all_data_moltypes,showfliers=False,color='tab:red',whis=whis)
sns.boxplot(ax=axs[1], x=cluster_labels_normdot, y="CHOS", data=df_all_data_moltypes,showfliers=False,color='tab:red',whis=whis)
sns.boxplot(ax=axs[2], x=cluster_labels_qt, y="CHOS", data=df_all_data_moltypes,showfliers=False,color='tab:red',whis=whis)
axs[0].set_title('CHOS')
[ax.set_ylabel("") for ax in axs]
[ax.grid(axis='y',alpha=0.5) for ax in axs]

axs = subfigs[3].subplots(nrows=3, ncols=1, sharey=True)
sns.boxplot(ax=axs[0], x=cluster_labels_unscaled, y="CHONS", data=df_all_data_moltypes,showfliers=False,color='tab:gray',whis=whis)
sns.boxplot(ax=axs[1], x=cluster_labels_normdot, y="CHONS", data=df_all_data_moltypes,showfliers=False,color='tab:gray',whis=whis)
sns.boxplot(ax=axs[2], x=cluster_labels_qt, y="CHONS", data=df_all_data_moltypes,showfliers=False,color='tab:gray',whis=whis)
axs[0].set_title('CHONS')
[ax.set_ylabel("") for ax in axs]
[ax.grid(axis='y',alpha=0.5) for ax in axs]


plt.show()

sns.reset_orig()

#%%Load air quality data

df_all_merge, df_merge_beijing_winter,df_merge_beijing_summer,df_merge_delhi_summer,df_merge_delhi_autumn = load_beijingdelhi_merge(newindex=df_all_data.index)

df_all_merge['cluster_labels_unscaled'] = cluster_labels_unscaled
df_all_merge['cluster_labels_qt'] = cluster_labels_qt
df_all_merge['cluster_labels_normdot'] = cluster_labels_normdot

##Load and add HYSPLIT precip data

ds_HYSPLIT_precip = pd.read_csv(r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\HYSPLIT_precip.csv",index_col='date_mid',parse_dates=True)
df_all_merge['HYSPLIT_precip'] = ds_HYSPLIT_precip


df_all_merge_grouped = pd.concat([df_all_merge]*3).groupby(np.concatenate([cluster_labels_unscaled,cluster_labels_qt+10,cluster_labels_normdot+50]))


#%%Diurnal profiles of air quality

df_all_merge_Beijing_win = df_all_merge.loc[ds_dataset_cat == "Beijing_winter"]
df_all_merge_Beijing_sum = df_all_merge.loc[ds_dataset_cat == "Beijing_summer"]
df_all_merge_Delhi_sum = df_all_merge.loc[ds_dataset_cat == "Delhi_summer"]
df_all_merge_Delhi_aut = df_all_merge.loc[ds_dataset_cat == "Delhi_autumn"]

fig,ax = plt.subplots(2,3,figsize=(10,10))
ax = ax.ravel()
ax[0].scatter(df_all_merge_Beijing_win.index.hour,df_all_merge_Beijing_win['co_ppbv'],marker='o',facecolors='none', edgecolors='blue')
ax[0].scatter(df_all_merge_Beijing_sum.index.hour,df_all_merge_Beijing_sum['co_ppbv'],marker='o',facecolors='none', edgecolors='red')
ax[0].scatter(df_all_merge_Delhi_sum.index.hour,df_all_merge_Delhi_sum['co_ppbv'],marker='o',facecolors='none', edgecolors='k')
ax[0].scatter(df_all_merge_Delhi_aut.index.hour,df_all_merge_Delhi_aut['co_ppbv'],marker='o',facecolors='none', edgecolors='gray')

ax[1].scatter(df_all_merge_Beijing_win.index.hour,df_all_merge_Beijing_win['no2_ppbv'],marker='o',facecolors='none', edgecolors='blue')
ax[1].scatter(df_all_merge_Beijing_sum.index.hour,df_all_merge_Beijing_sum['no2_ppbv'],marker='o',facecolors='none', edgecolors='red')
ax[1].scatter(df_all_merge_Delhi_sum.index.hour,df_all_merge_Delhi_sum['no2_ppbv'],marker='o',facecolors='none', edgecolors='k')
ax[1].scatter(df_all_merge_Delhi_aut.index.hour,df_all_merge_Delhi_aut['no2_ppbv'],marker='o',facecolors='none', edgecolors='gray')

ax[2].scatter(df_all_merge_Beijing_win.index.hour,df_all_merge_Beijing_win['o3_ppbv'],marker='o',facecolors='none', edgecolors='blue')
ax[2].scatter(df_all_merge_Beijing_sum.index.hour,df_all_merge_Beijing_sum['o3_ppbv'],marker='o',facecolors='none', edgecolors='red')
ax[2].scatter(df_all_merge_Delhi_sum.index.hour,df_all_merge_Delhi_sum['o3_ppbv'],marker='o',facecolors='none', edgecolors='k')
ax[2].scatter(df_all_merge_Delhi_aut.index.hour,df_all_merge_Delhi_aut['o3_ppbv'],marker='o',facecolors='none', edgecolors='gray')

ax[3].scatter(df_all_merge_Beijing_win.index.hour,df_all_merge_Beijing_win['AMS_NO3'],marker='o',facecolors='none', edgecolors='blue')
ax[3].scatter(df_all_merge_Beijing_sum.index.hour,df_all_merge_Beijing_sum['AMS_NO3'],marker='o',facecolors='none', edgecolors='red')
ax[3].scatter(df_all_merge_Delhi_sum.index.hour,df_all_merge_Delhi_sum['AMS_NO3'],marker='o',facecolors='none', edgecolors='k')
ax[3].scatter(df_all_merge_Delhi_aut.index.hour,df_all_merge_Delhi_aut['AMS_NO3'],marker='o',facecolors='none', edgecolors='gray')



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

precip_scale_max = 1.1 * df_all_merge_grouped['HYSPLIT_precip'].quantile(0.95,interpolation='lower').max()
precip_scale_min = 0.9 * df_all_merge_grouped['HYSPLIT_precip'].quantile(0.05,interpolation='lower').min()

wind_scale_max = 1.1 * df_all_merge_grouped['ws_ms'].quantile(0.95,interpolation='lower').max()
wind_scale_min = 0.9 * df_all_merge_grouped['ws_ms'].quantile(0.05,interpolation='lower').min()


limits = [[0,co_scale_max],[0,no2_scale_max],[precip_scale_min,precip_scale_max],[0,o3_scale_max],[0,so2_scale_max],[rh_scale_min,rh_scale_max]]
#limits = [[0,co_scale_max],[0,no2_scale_max],[precip_scale_min,precip_scale_max],[0,o3_scale_max],[0,so2_scale_max],[wind_scale_min,wind_scale_max]]

whis=[5,95]
sns.set_context("talk", font_scale=1)

#Unscaled data
fig,ax = plt.subplots(2,3,figsize=(12,8))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x='cluster_labels_unscaled', y="co_ppbv", data=df_all_merge,showfliers=False,color='tab:gray',whis=whis)
sns.boxplot(ax=ax[1], x='cluster_labels_unscaled', y="no2_ppbv", data=df_all_merge,showfliers=False,color='tab:blue',whis=whis)
sns.boxplot(ax=ax[3], x='cluster_labels_unscaled', y="o3_ppbv", data=df_all_merge,showfliers=False,color='tab:green',whis=whis)
sns.boxplot(ax=ax[4], x='cluster_labels_unscaled', y="so2_ppbv", data=df_all_merge,showfliers=False,color='tab:red',whis=whis)
sns.boxplot(ax=ax[2], x='cluster_labels_unscaled', y="HYSPLIT_precip", data=df_all_merge,showfliers=False,color='tab:olive',whis=whis)
sns.boxplot(ax=ax[5], x='cluster_labels_unscaled', y="RH", data=df_all_merge,showfliers=False,color='tab:cyan',whis=whis)
[axis.set_ylim(lim) for axis,lim in zip(ax,limits)]
plt.suptitle('Unscaled data, 4 clusters')
plt.tight_layout()
plt.show()


#qt data
fig,ax = plt.subplots(2,3,figsize=(12,8))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x='cluster_labels_qt', y="co_ppbv", data=df_all_merge,showfliers=False,color='tab:gray',whis=whis)
sns.boxplot(ax=ax[1], x='cluster_labels_qt', y="no2_ppbv", data=df_all_merge,showfliers=False,color='tab:blue',whis=whis)
sns.boxplot(ax=ax[3], x='cluster_labels_qt', y="o3_ppbv", data=df_all_merge,showfliers=False,color='tab:green',whis=whis)
sns.boxplot(ax=ax[4], x='cluster_labels_qt', y="so2_ppbv", data=df_all_merge,showfliers=False,color='tab:red',whis=whis)
sns.boxplot(ax=ax[2], x='cluster_labels_qt', y="HYSPLIT_precip", data=df_all_merge,showfliers=False,color='tab:olive',whis=whis)
sns.boxplot(ax=ax[5], x='cluster_labels_qt', y="RH", data=df_all_merge,showfliers=False,color='tab:cyan',whis=whis)
[axis.set_ylim(lim) for axis,lim in zip(ax,limits)]
plt.suptitle('qt data, 7 clusters')
plt.tight_layout()
plt.show()

#normdot data
fig,ax = plt.subplots(2,3,figsize=(12,8))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x='cluster_labels_normdot', y="co_ppbv", data=df_all_merge,showfliers=False,color='tab:gray',whis=whis)
sns.boxplot(ax=ax[1], x='cluster_labels_normdot', y="no2_ppbv", data=df_all_merge,showfliers=False,color='tab:blue',whis=whis)
sns.boxplot(ax=ax[3], x='cluster_labels_normdot', y="o3_ppbv", data=df_all_merge,showfliers=False,color='tab:green',whis=whis)
sns.boxplot(ax=ax[4], x='cluster_labels_normdot', y="so2_ppbv", data=df_all_merge,showfliers=False,color='tab:red',whis=whis)
sns.boxplot(ax=ax[2], x='cluster_labels_normdot', y="HYSPLIT_precip", data=df_all_merge,showfliers=False,color='tab:olive',whis=whis)
sns.boxplot(ax=ax[5], x='cluster_labels_normdot', y="RH", data=df_all_merge,showfliers=False,color='tab:cyan',whis=whis)
[axis.set_ylim(lim) for axis,lim in zip(ax,limits)]
plt.suptitle('normdot data, 8 clusters')
plt.tight_layout()
plt.show()


sns.reset_orig()



#%%Now the same for the AMS data

#Make dataframes of the mean of each pmf factor for each cluster, as a fraction so all add up to 1
df_all_merge_AMS_PMF_frac_unscaled = df_all_merge[['AMS_FFOA','AMS_COA','AMS_BBOA','AMS_OOA']].groupby(df_all_merge['cluster_labels_unscaled']).mean()
df_all_merge_AMS_PMF_frac_unscaled = df_all_merge_AMS_PMF_frac_unscaled.div(df_all_merge_AMS_PMF_frac_unscaled.sum(axis=1,skipna=False),axis=0)
df_all_merge_AMS_PMF_frac_qt = df_all_merge[['AMS_FFOA','AMS_COA','AMS_BBOA','AMS_OOA']].groupby(df_all_merge['cluster_labels_qt']).mean()
df_all_merge_AMS_PMF_frac_qt = df_all_merge_AMS_PMF_frac_qt.div(df_all_merge_AMS_PMF_frac_qt.sum(axis=1,skipna=False),axis=0)
df_all_merge_AMS_PMF_frac_normdot = df_all_merge[['AMS_FFOA','AMS_COA','AMS_BBOA','AMS_OOA']].groupby(df_all_merge['cluster_labels_normdot']).mean()
df_all_merge_AMS_PMF_frac_normdot = df_all_merge_AMS_PMF_frac_normdot.div(df_all_merge_AMS_PMF_frac_normdot.sum(axis=1,skipna=False),axis=0)


#Make all the y scales the same
scale=1.1
limits = [
    [0,scale*df_all_merge_grouped['AMS_NH4'].quantile(0.75).max()],
    [0,scale*df_all_merge_grouped['AMS_NO3'].quantile(0.75).max()],
    [0,scale*df_all_merge_grouped['AMS_Chl'].quantile(0.75).max()],
    [0,scale*df_all_merge_grouped['AMS_Org'].quantile(0.75).max()],
    [0,scale*df_all_merge_grouped['AMS_SO4'].quantile(0.75).max()]]
    

sns.set_context("talk", font_scale=1)

#unscaled data
fig,ax = plt.subplots(2,4,figsize=(12,8))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x='cluster_labels_unscaled', y="AMS_Org", data=df_all_merge,showfliers=False,color='tab:green')
sns.boxplot(ax=ax[1], x='cluster_labels_unscaled', y="AMS_NO3", data=df_all_merge,showfliers=False,color='tab:blue')
sns.boxplot(ax=ax[2], x='cluster_labels_unscaled', y="AMS_SO4", data=df_all_merge,showfliers=False,color='tab:red')
df_all_merge_AMS_PMF_frac_unscaled.plot(ax=ax[3],kind='bar', stacked=True,
                                       color=['dimgray', 'silver', 'tab:brown','mediumpurple'],
                                       legend=False,width=0.9)
ax[3].set_ylabel('PMF fraction')
sns.boxplot(ax=ax[4], x='cluster_labels_unscaled', y="AMS_FFOA", data=df_all_merge,showfliers=False,color='dimgray')
sns.boxplot(ax=ax[5], x='cluster_labels_unscaled', y="AMS_COA", data=df_all_merge,showfliers=False,color='silver')
sns.boxplot(ax=ax[6], x='cluster_labels_unscaled', y="AMS_BBOA", data=df_all_merge,showfliers=False,color='tab:brown')
sns.boxplot(ax=ax[7], x='cluster_labels_unscaled', y="AMS_OOA", data=df_all_merge,showfliers=False,color='mediumpurple')
#[ax[i].set_ylim(limits[i]) for i in range(len(limits))]
plt.suptitle('unscaled data, 7 clusters')
[axis.set_xlabel('')  for axis in ax]
plt.tight_layout()
plt.show()


#qt data
fig,ax = plt.subplots(2,4,figsize=(14,8))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x='cluster_labels_qt', y="AMS_Org", data=df_all_merge,showfliers=False,color='tab:green')
sns.boxplot(ax=ax[1], x='cluster_labels_qt', y="AMS_NO3", data=df_all_merge,showfliers=False,color='tab:blue')
sns.boxplot(ax=ax[2], x='cluster_labels_qt', y="AMS_SO4", data=df_all_merge,showfliers=False,color='tab:red')
df_all_merge_AMS_PMF_frac_qt.plot(ax=ax[3],kind='bar', stacked=True,
                                       color=['dimgray', 'silver', 'tab:brown','mediumpurple'],
                                       legend=False,width=0.9)
ax[3].set_ylabel('PMF fraction')
sns.boxplot(ax=ax[4], x='cluster_labels_qt', y="AMS_FFOA", data=df_all_merge,showfliers=False,color='dimgray')
sns.boxplot(ax=ax[5], x='cluster_labels_qt', y="AMS_COA", data=df_all_merge,showfliers=False,color='silver')
sns.boxplot(ax=ax[6], x='cluster_labels_qt', y="AMS_BBOA", data=df_all_merge,showfliers=False,color='tab:brown')
sns.boxplot(ax=ax[7], x='cluster_labels_qt', y="AMS_OOA", data=df_all_merge,showfliers=False,color='mediumpurple')
#(ax[i].set_ylim(limits[i]) for i in range(len(limits)))
plt.suptitle('qt data, 7 clusters')
[axis.set_xlabel('')  for axis in ax]
plt.tight_layout()
plt.show()

#normdot data
fig,ax = plt.subplots(2,4,figsize=(14,8))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x='cluster_labels_normdot', y="AMS_Org", data=df_all_merge,showfliers=False,color='tab:green')
sns.boxplot(ax=ax[1], x='cluster_labels_normdot', y="AMS_NO3", data=df_all_merge,showfliers=False,color='tab:blue')
sns.boxplot(ax=ax[2], x='cluster_labels_normdot', y="AMS_SO4", data=df_all_merge,showfliers=False,color='tab:red')
df_all_merge_AMS_PMF_frac_normdot.plot(ax=ax[3],kind='bar', stacked=True,
                                       color=['dimgray', 'silver', 'tab:brown','mediumpurple'],
                                       legend=False,width=0.9)
ax[3].set_ylabel('PMF fraction')
sns.boxplot(ax=ax[4], x='cluster_labels_normdot', y="AMS_FFOA", data=df_all_merge,showfliers=False,color='dimgray')
sns.boxplot(ax=ax[5], x='cluster_labels_normdot', y="AMS_COA", data=df_all_merge,showfliers=False,color='silver')
sns.boxplot(ax=ax[6], x='cluster_labels_normdot', y="AMS_BBOA", data=df_all_merge,showfliers=False,color='tab:brown')
sns.boxplot(ax=ax[7], x='cluster_labels_normdot', y="AMS_OOA", data=df_all_merge,showfliers=False,color='mediumpurple')
#(ax[i].set_ylim(limits[i]) for i in range(len(limits)))
plt.suptitle('normdot data, 8 clusters')
[axis.set_xlabel('')  for axis in ax]
plt.tight_layout()
plt.show()

sns.reset_orig()




#%%Check precip vs AMS org vs co
fig,ax = plt.subplots()
df_all_merge.loc[df_all_merge['HYSPLIT_precip']==0].plot.scatter(x='co_ppbv',y='AMS_Org',c='k',ax=ax)
df_all_merge.loc[df_all_merge['HYSPLIT_precip']>0].plot.scatter(x='co_ppbv',y='AMS_Org',c='b',ax=ax)

plt.show()

#%%Plot AQ vs wind/precip

plt.scatter(df_all_merge['HYSPLIT_precip'],df_all_data_moltypes['CHO'])
plt.show()
plt.scatter(df_all_merge['ws_ms'],df_all_data_moltypes['CHO'])
plt.show()

#%%Plot wind rose per city per cluster


plot_windrose_percluster(df_all_merge,cluster_labels_unscaled,ds_dataset_cat,suptitle='Unscaled data')
plot_windrose_percluster(df_all_merge,cluster_labels_qt,ds_dataset_cat,suptitle='qt data')
plot_windrose_percluster(df_all_merge,cluster_labels_normdot,ds_dataset_cat,suptitle='normdot data')

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



fig,ax = plt.subplots(2,2,figsize=(8,8))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x=cluster_labels_unscaled, y="CHO", data=df_all_data_moltypes_pct,showfliers=False,color='tab:green',whis=whis)
sns.boxplot(ax=ax[1], x=cluster_labels_unscaled, y="CHON", data=df_all_data_moltypes_pct,showfliers=False,color='tab:blue',whis=whis)
sns.boxplot(ax=ax[2], x=cluster_labels_unscaled, y="CHOS", data=df_all_data_moltypes_pct,showfliers=False,color='tab:red',whis=whis)
sns.boxplot(ax=ax[3], x=cluster_labels_unscaled, y="CHONS", data=df_all_data_moltypes_pct,showfliers=False,color='tab:gray',whis=whis)
plt.suptitle('Unscaled data percentiles, 4 clusters')
plt.tight_layout()
plt.show()


# #Unscaled data
# fig,ax = plt.subplots(2,2,figsize=(8,8))
# ax = ax.ravel()
# sns.boxplot(ax=ax[0], x=cluster_labels_unscaled, y="CHO", data=df_all_data_moltypes_frac,showfliers=False,color='tab:green',whis=whis)
# sns.boxplot(ax=ax[1], x=cluster_labels_unscaled, y="CHON", data=df_all_data_moltypes_frac,showfliers=False,color='tab:blue',whis=whis)
# sns.boxplot(ax=ax[2], x=cluster_labels_unscaled, y="CHOS", data=df_all_data_moltypes_frac,showfliers=False,color='tab:red',whis=whis)
# sns.boxplot(ax=ax[3], x=cluster_labels_unscaled, y="CHONS", data=df_all_data_moltypes_frac,showfliers=False,color='tab:gray',whis=whis)

# #[axis.set_ylim(lim) for axis,lim in zip(ax,limits)]
# plt.suptitle('Unscaled data fraction, 4 clusters')
# plt.tight_layout()
# plt.show()


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

# #qt data
# fig,ax = plt.subplots(2,2,figsize=(8,8))
# ax = ax.ravel()
# sns.boxplot(ax=ax[0], x=cluster_labels_qt, y="CHO", data=df_all_data_moltypes_frac,showfliers=False,color='tab:green',whis=whis)
# sns.boxplot(ax=ax[1], x=cluster_labels_qt, y="CHON", data=df_all_data_moltypes_frac,showfliers=False,color='tab:blue',whis=whis)
# sns.boxplot(ax=ax[2], x=cluster_labels_qt, y="CHOS", data=df_all_data_moltypes_frac,showfliers=False,color='tab:red',whis=whis)
# sns.boxplot(ax=ax[3], x=cluster_labels_qt, y="CHONS", data=df_all_data_moltypes_frac,showfliers=False,color='tab:gray',whis=whis)

# #[axis.set_ylim(lim) for axis,lim in zip(ax,limits)]
# plt.suptitle('qt data fraction, 7 clusters')
# plt.tight_layout()
# plt.show()


#normdot data
fig,ax = plt.subplots(2,2,figsize=(8,8))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x=cluster_labels_normdot, y="CHO", data=df_all_data_moltypes,showfliers=False,color='tab:green',whis=whis)
sns.boxplot(ax=ax[1], x=cluster_labels_normdot, y="CHON", data=df_all_data_moltypes,showfliers=False,color='tab:blue',whis=whis)
sns.boxplot(ax=ax[2], x=cluster_labels_normdot, y="CHOS", data=df_all_data_moltypes,showfliers=False,color='tab:red',whis=whis)
sns.boxplot(ax=ax[3], x=cluster_labels_normdot, y="CHONS", data=df_all_data_moltypes,showfliers=False,color='tab:gray',whis=whis)

#[axis.set_ylim(lim) for axis,lim in zip(ax,limits)]
plt.suptitle('normdot data, 8 clusters')
plt.tight_layout()
plt.show()

# #normdot data
# fig,ax = plt.subplots(2,2,figsize=(8,8))
# ax = ax.ravel()
# sns.boxplot(ax=ax[0], x=cluster_labels_normdot, y="CHO", data=df_all_data_moltypes_frac,showfliers=False,color='tab:green',whis=whis)
# sns.boxplot(ax=ax[1], x=cluster_labels_normdot, y="CHON", data=df_all_data_moltypes_frac,showfliers=False,color='tab:blue',whis=whis)
# sns.boxplot(ax=ax[2], x=cluster_labels_normdot, y="CHOS", data=df_all_data_moltypes_frac,showfliers=False,color='tab:red',whis=whis)
# sns.boxplot(ax=ax[3], x=cluster_labels_normdot, y="CHONS", data=df_all_data_moltypes_frac,showfliers=False,color='tab:gray',whis=whis)

# #[axis.set_ylim(lim) for axis,lim in zip(ax,limits)]
# plt.suptitle('normdot data fraction, 7 clusters')
# plt.tight_layout()
# plt.show()

sns.reset_orig()

    
#%%Plot data by molecule type2, as percentiles
sns.set_context("talk", font_scale=1)
whis=[5,95]

#UNSCALED DATA
fig,ax = plt.subplots(1,3,figsize=(8,5),sharey=True)
ax = ax.ravel()
sns.boxplot(ax=ax[0], x=cluster_labels_unscaled, y="CHOX", data=df_all_data_moltypes2_pct,showfliers=False,color='tab:green',whis=whis)
sns.boxplot(ax=ax[1], x=cluster_labels_unscaled, y="CHNX", data=df_all_data_moltypes2_pct,showfliers=False,color='tab:blue',whis=whis)
sns.boxplot(ax=ax[2], x=cluster_labels_unscaled, y="CHSX", data=df_all_data_moltypes2_pct,showfliers=False,color='tab:red',whis=whis)
ax[0].set_title('CHOX')
ax[1].set_title('CHNX')
ax[2].set_title('CHSX')
ax[0].set_ylabel('Percentile')
ax[1].set_ylabel('')
ax[2].set_ylabel('')
plt.suptitle('Unscaled data percentiles, 4 clusters')
plt.tight_layout()
plt.show()


#qt DATA
fig,ax = plt.subplots(1,3,figsize=(8,5),sharey=True)
ax = ax.ravel()
sns.boxplot(ax=ax[0], x=cluster_labels_qt, y="CHOX", data=df_all_data_moltypes2_pct,showfliers=False,color='tab:green',whis=whis)
sns.boxplot(ax=ax[1], x=cluster_labels_qt, y="CHNX", data=df_all_data_moltypes2_pct,showfliers=False,color='tab:blue',whis=whis)
sns.boxplot(ax=ax[2], x=cluster_labels_qt, y="CHSX", data=df_all_data_moltypes2_pct,showfliers=False,color='tab:red',whis=whis)
ax[0].set_title('CHOX')
ax[1].set_title('CHNX')
ax[2].set_title('CHSX')
ax[0].set_ylabel('Percentile')
ax[1].set_ylabel('')
ax[2].set_ylabel('')
plt.suptitle('qt data percentiles, 7 clusters')
plt.tight_layout()
plt.show()

#normdot DATA
fig,ax = plt.subplots(1,3,figsize=(8,5),sharey=True)
ax = ax.ravel()
sns.boxplot(ax=ax[0], x=cluster_labels_normdot, y="CHOX", data=df_all_data_moltypes2_pct,showfliers=False,color='tab:green',whis=whis)
sns.boxplot(ax=ax[1], x=cluster_labels_normdot, y="CHNX", data=df_all_data_moltypes2_pct,showfliers=False,color='tab:blue',whis=whis)
sns.boxplot(ax=ax[2], x=cluster_labels_normdot, y="CHSX", data=df_all_data_moltypes2_pct,showfliers=False,color='tab:red',whis=whis)
ax[0].set_title('CHOX')
ax[1].set_title('CHNX')
ax[2].set_title('CHSX')
ax[0].set_ylabel('Percentile')
ax[1].set_ylabel('')
ax[2].set_ylabel('')
plt.suptitle('normdot data percentiles, 8 clusters')
plt.tight_layout()
plt.show()

sns.reset_orig()


#%%Plot data by molecule type2, as percentiles AND means
sns.set_context("talk", font_scale=1)
whis=[5,95]

#UNSCALED DATA
fig,ax = plt.subplots(2,3,figsize=(10,8))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x=cluster_labels_unscaled, y="CHOX", data=df_all_data_moltypes2,showfliers=False,color='tab:green',whis=whis)
sns.boxplot(ax=ax[1], x=cluster_labels_unscaled, y="CHNX", data=df_all_data_moltypes2,showfliers=False,color='tab:blue',whis=whis)
sns.boxplot(ax=ax[2], x=cluster_labels_unscaled, y="CHSX", data=df_all_data_moltypes2,showfliers=False,color='tab:red',whis=whis)
sns.boxplot(ax=ax[3], x=cluster_labels_unscaled, y="CHOX", data=df_all_data_moltypes2_pct,showfliers=False,color='tab:green',whis=whis)
sns.boxplot(ax=ax[4], x=cluster_labels_unscaled, y="CHNX", data=df_all_data_moltypes2_pct,showfliers=False,color='tab:blue',whis=whis)
sns.boxplot(ax=ax[5], x=cluster_labels_unscaled, y="CHSX", data=df_all_data_moltypes2_pct,showfliers=False,color='tab:red',whis=whis)
ax[0].set_title('CHOX')
ax[1].set_title('CHNX')
ax[2].set_title('CHSX')
ax[0].set_ylabel('µg$\,$m$^{-3}$')
ax[1].set_ylabel('')
ax[2].set_ylabel('')
ax[4].set_ylabel('')
ax[5].set_ylabel('')
ax[3].set_ylabel('Percentile')
ax[4].set_ylabel('')
ax[5].set_ylabel('')
plt.suptitle('Unscaled data, 4 clusters')
plt.tight_layout()
plt.show()

#qt DATA
fig,ax = plt.subplots(2,3,figsize=(10,8))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x=cluster_labels_qt, y="CHOX", data=df_all_data_moltypes2,showfliers=False,color='tab:green',whis=whis)
sns.boxplot(ax=ax[1], x=cluster_labels_qt, y="CHNX", data=df_all_data_moltypes2,showfliers=False,color='tab:blue',whis=whis)
sns.boxplot(ax=ax[2], x=cluster_labels_qt, y="CHSX", data=df_all_data_moltypes2,showfliers=False,color='tab:red',whis=whis)
sns.boxplot(ax=ax[3], x=cluster_labels_qt, y="CHOX", data=df_all_data_moltypes2_pct,showfliers=False,color='tab:green',whis=whis)
sns.boxplot(ax=ax[4], x=cluster_labels_qt, y="CHNX", data=df_all_data_moltypes2_pct,showfliers=False,color='tab:blue',whis=whis)
sns.boxplot(ax=ax[5], x=cluster_labels_qt, y="CHSX", data=df_all_data_moltypes2_pct,showfliers=False,color='tab:red',whis=whis)
ax[0].set_title('CHOX')
ax[1].set_title('CHNX')
ax[2].set_title('CHSX')
ax[0].set_ylabel('µg$\,$m$^{-3}$')
ax[1].set_ylabel('')
ax[2].set_ylabel('')
ax[4].set_ylabel('')
ax[5].set_ylabel('')
ax[3].set_ylabel('Percentile')
ax[4].set_ylabel('')
ax[5].set_ylabel('')
plt.suptitle('qt data, 7 clusters')
plt.tight_layout()
plt.show()


#normdot DATA
fig,ax = plt.subplots(2,3,figsize=(10,8))
ax = ax.ravel()
sns.boxplot(ax=ax[0], x=cluster_labels_normdot, y="CHOX", data=df_all_data_moltypes2,showfliers=False,color='tab:green',whis=whis)
sns.boxplot(ax=ax[1], x=cluster_labels_normdot, y="CHNX", data=df_all_data_moltypes2,showfliers=False,color='tab:blue',whis=whis)
sns.boxplot(ax=ax[2], x=cluster_labels_normdot, y="CHSX", data=df_all_data_moltypes2,showfliers=False,color='tab:red',whis=whis)
sns.boxplot(ax=ax[3], x=cluster_labels_normdot, y="CHOX", data=df_all_data_moltypes2_pct,showfliers=False,color='tab:green',whis=whis)
sns.boxplot(ax=ax[4], x=cluster_labels_normdot, y="CHNX", data=df_all_data_moltypes2_pct,showfliers=False,color='tab:blue',whis=whis)
sns.boxplot(ax=ax[5], x=cluster_labels_normdot, y="CHSX", data=df_all_data_moltypes2_pct,showfliers=False,color='tab:red',whis=whis)
ax[0].set_title('CHOX')
ax[1].set_title('CHNX')
ax[2].set_title('CHSX')
ax[0].set_ylabel('µg$\,$m$^{-3}$')
ax[1].set_ylabel('')
ax[2].set_ylabel('')
ax[4].set_ylabel('')
ax[5].set_ylabel('')
ax[3].set_ylabel('Percentile')
ax[4].set_ylabel('')
ax[5].set_ylabel('')
plt.suptitle('normdot data, 8 clusters')
plt.tight_layout()
plt.show()




#%%Plot molecule type, plus air quality and met for one cluster workflow


#%%Plot different molecules by cluster

#Make dataframes of the mean of each pmf factor for each cluster, as a fraction so all add up to 1
df_all_merge_AMS_PMF_frac_unscaled = df_all_merge[['AMS_FFOA','AMS_COA','AMS_BBOA','AMS_OOA']].groupby(df_all_merge['cluster_labels_unscaled']).mean()
df_all_merge_AMS_PMF_frac_unscaled = df_all_merge_AMS_PMF_frac_unscaled.div(df_all_merge_AMS_PMF_frac_unscaled.sum(axis=1,skipna=False),axis=0)
df_all_merge_AMS_PMF_frac_qt = df_all_merge[['AMS_FFOA','AMS_COA','AMS_BBOA','AMS_OOA']].groupby(df_all_merge['cluster_labels_qt']).mean()
df_all_merge_AMS_PMF_frac_qt = df_all_merge_AMS_PMF_frac_qt.div(df_all_merge_AMS_PMF_frac_qt.sum(axis=1,skipna=False),axis=0)
df_all_merge_AMS_PMF_frac_normdot = df_all_merge[['AMS_FFOA','AMS_COA','AMS_BBOA','AMS_OOA']].groupby(df_all_merge['cluster_labels_normdot']).mean()
df_all_merge_AMS_PMF_frac_normdot = df_all_merge_AMS_PMF_frac_normdot.div(df_all_merge_AMS_PMF_frac_normdot.sum(axis=1,skipna=False),axis=0)

  

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
    
    [axis.set_ylabel(ylab)  for axis, ylab in zip(ax,ylabels)]
    
    #Add letters in boxes for each subfigure
    trans = mtransforms.ScaledTranslation(2/72, -5/72, fig.dpi_scale_trans)
    
    for axis, letter in zip(ax,string.ascii_lowercase):
        axis.text(0.0, 1.0, ('(' + letter + ')'), transform=axis.transAxes + trans,
                fontsize='medium', verticalalignment='top', fontfamily='serif',
                bbox=dict(facecolor='1.0', edgecolor='none', pad=1.0))
    
    
    
    
    
    ax[0].set_ylabel(ylabels[0])
    
    if 'suptitle' in kwargs:    
        plt.suptitle(kwargs.get('suptitle'))
    plt.tight_layout()
    plt.show()


#plot_orbitrap_ams_aqmet(cluster_labels_unscaled,df_all_data_moltypes,df_all_merge,suptitle='Naive workflow, 4 clusters')
plot_orbitrap_ams_aqmet(cluster_labels_qt,df_all_data_moltypes,df_all_merge,suptitle='QT workflow, 7 clusters')
#plot_orbitrap_ams_aqmet(cluster_labels_normdot,df_all_data_moltypes,df_all_merge,suptitle='Normdot workflow, 8 clusters')



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

#Calculate the top30 unusually high or low peaks in each cluster
#top_peaks_unscaled = cluster_top_percentiles(df_all_data,cluster_labels_unscaled,30)


top_peaks_unscaled = cluster_top_percentiles(df_all_data,cluster_labels_unscaled,30,mol_labels=ds_mol_aerosolomics)
top_peaks_qt = cluster_top_percentiles(df_all_data,cluster_labels_qt,30,mol_labels=ds_mol_aerosolomics)
top_peaks_normdot= cluster_top_percentiles(df_all_data,cluster_labels_normdot,30,mol_labels=ds_mol_aerosolomics)
bottom_peaks_unscaled= cluster_top_percentiles(df_all_data,cluster_labels_unscaled,30,highest=False,mol_labels=ds_mol_aerosolomics)
bottom_peaks_qt= cluster_top_percentiles(df_all_data,cluster_labels_qt,30,highest=False,mol_labels=ds_mol_aerosolomics)
bottom_peaks_normdot= cluster_top_percentiles(df_all_data,cluster_labels_normdot,30,highest=False,mol_labels=ds_mol_aerosolomics)

#Export to CSV
export_path = r'C:\Users\mbcx5jt5\Dropbox (The University of Manchester)\Complex-SOA\Clustering\Unusual_Peaks'

top_peaks_unscaled.to_csv(export_path + '\pct_top_peaks_unscaled.csv')
top_peaks_qt.to_csv(export_path + '\pct_top_peaks_qt.csv')
top_peaks_normdot.to_csv(export_path + '\pct_top_peaks_normdot.csv')

bottom_peaks_unscaled.to_csv(export_path + '\pct_bottom_peaks_unscaled.csv')
bottom_peaks_qt.to_csv(export_path + '\pct_bottom_peaks_qt.csv')
bottom_peaks_normdot.to_csv(export_path + '\pct_bottom_peaks_normdot.csv')




#%%Extract the top peaks for each cluster

df_JT_peaks = pd.read_csv(r"C:\Users\mbcx5jt5\Dropbox (The University of Manchester)\Complex-SOA\Clustering\Cluster_Top_Peaks\JT_mol_list.csv",index_col='Formula',encoding='ISO-8859-1')

#Extract the top n peaks for each cluster, tag with labels from SAri and aerosolomics, and save as csv
def extract_clusters_top_peaks_csv(df_data,cluster_labels,n_peaks,csvpath,**kwargs):
    
    #Check if peak labels are there
    if "sari_peaks" in kwargs:
        ds_sari_peaks = kwargs.get("sari_peaks")
        sari_peaks = True
    else:
        sari_peaks = False
    if "aerosolomics_peaks" in kwargs:
        ds_aerosolomics_peaks = kwargs.get("aerosolomics_peaks")
        aerosolomics = True
    else:
        aerosolomics = False
    if "JT_peaks" in kwargs:
        df_JT_peaks = kwargs.get("JT_peaks")
        JT_peaks = True
    else:
        JT_peaks = False
    #Percentage threshold rather than top n peaks
    if "pct" in kwargs:
        pct = kwargs.get("pct")
    else:
        pct = 0
    
    
    
    
    #make empty csv
    with open(csvpath, "w") as my_empty_csv:
        pass    

    unique_clusters = np.unique(cluster_labels)
    
    for cluster in unique_clusters:
        df_top_peaks = cluster_extract_peaks(df_all_data.loc[cluster_labels == cluster].mean(axis=0), df_all_data.T,n_peaks,dp=1,dropRT=False)
        df_top_peaks.index = np.arange(0,n_peaks)+1
        df_top_peaks = df_top_peaks.drop('Name',axis=1)
        
        
        #Extract the labels from Sari's list
        if(sari_peaks):
            ds_Sari_list = pd.Series(np.empty(n_peaks,dtype='<U10'),index=df_top_peaks.index)
            for peak in df_top_peaks.index:
                mol_nospace = df_top_peaks['Formula'].loc[peak].replace(" ","")              
                try:
                    ds_Sari_list.loc[peak] = ds_sari_peaks.loc[mol_nospace].values[0]
                except:
                    ds_Sari_list.loc[peak] = ''
            df_top_peaks['Sari'] = ds_Sari_list
        
        #Same for aerosolomics
        if(aerosolomics):
            ds_aero_list = pd.Series(np.empty(n_peaks,dtype='<U10'),index=df_top_peaks.index)
            for peak in df_top_peaks.index:
                try:
                    ds_aero_list.loc[peak] = ds_aerosolomics_peaks.loc[df_top_peaks['Formula'].loc[peak]]
                except:
                    ds_aero_list.loc[peak] = ''
            #pdb.set_trace()
            df_top_peaks['Aerosolomics'] = ds_aero_list
            
        #Same for JT's peaks
        if(JT_peaks):
            df_JT_list = pd.DataFrame(np.empty([n_peaks,3]),dtype='<U10',index=df_top_peaks.index,columns=df_JT_peaks.columns)
            for peak in df_top_peaks.index:
                try:
                    df_JT_list.loc[peak] = df_JT_peaks.loc[df_top_peaks['Formula'].loc[peak]]
                except:
                    df_JT_list.loc[peak] = ['','','']
            df_top_peaks[['Potential source','Reference(s)']] = df_JT_list[['Source','Reference']]
        
        
        #Use percentage threshold, percentage > pct
        if pct>0:
            df_top_peaks = df_top_peaks.loc[df_top_peaks['peak_pct']>=pct]
        
        
        #Edit headers
        df_top_peaks.rename(columns={'RT':'RT (min)'},inplace=True)
        df_top_peaks.rename(columns={'peak_pct':'Fraction (%)'},inplace=True)
        
        #append to csv
        if "prefix" in kwargs:
            header = kwargs.get("prefix")
        else:
            header = ""
        header = header + 'Cluster ' + str(cluster) + '\n'
        footer = '\n\n'
        with open(csvpath, 'a', newline='\n') as file_buffer:
            #Add a header 
            file_buffer.write(header)
            #Append to csv
            df_top_peaks.to_csv(file_buffer,line_terminator='')
            #Add footer
            file_buffer.write(footer)
        
        
    
    

path_unscaled = r"C:\Users\mbcx5jt5\Dropbox (The University of Manchester)\Complex-SOA\Clustering\Cluster_Top_Peaks\top_peaks_unscaled.csv"
path_qt = r"C:\Users\mbcx5jt5\Dropbox (The University of Manchester)\Complex-SOA\Clustering\Cluster_Top_Peaks\top_peaks_qt.csv"
path_normdot = r"C:\Users\mbcx5jt5\Dropbox (The University of Manchester)\Complex-SOA\Clustering\Cluster_Top_Peaks\top_peaks_normdot.csv"
extract_clusters_top_peaks_csv(df_all_data,cluster_labels_unscaled,30,path_unscaled, prefix="Unscaled ",JT_peaks=df_JT_peaks,pct=1.)
extract_clusters_top_peaks_csv(df_all_data,cluster_labels_qt,30,path_qt, prefix="qt ",JT_peaks=df_JT_peaks,pct=1.)
extract_clusters_top_peaks_csv(df_all_data,cluster_labels_normdot,30,path_normdot, prefix="normdot ",JT_peaks=df_JT_peaks,pct=1.)

#extract_clusters_top_peaks_csv(df_all_data,cluster_labels_normdot,30,path_normdot, prefix="normdot ",sari_peaks=Sari_peaks_list,aerosolomics_peaks=ds_mol_aerosolomics_nodup,JT_peaks=df_JT_peaks,pct=1.)







#%%Classify unusually high peaks

#What fraction of CHOX/CHNX/CHSX are unusually high versus unusually low??
#What are each of the molecules doing?


fig,ax= plt.subplots(1,3,figsize=(10,5))
ax = ax.ravel()
sns.boxplot(y=df_all_data_moltypes2['CHOX'],x=cluster_labels_unscaled,ax=ax[0])
sns.boxplot(y=df_all_data_moltypes2['CHSX'],x=cluster_labels_unscaled,ax=ax[1])
sns.boxplot(y=df_all_data_moltypes2['CHNX'],x=cluster_labels_unscaled,ax=ax[2])

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