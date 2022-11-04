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
import time

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback, TerminateOnNaN
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, FunctionTransformer, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score,silhouette_score,adjusted_rand_score
from sklearn.manifold import TSNE

import math


import os
os.chdir('C:/Work/Python/Github/Orbitrap_clustering')
from ae_functions import *
from orbitrap_functions import *





#%%Load data from HDF
filepath = r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\PMF data\ORBITRAP_Data_Pre_PMF.h5"
df_all_data, df_all_err, ds_all_mz = Load_pre_PMF_data(filepath,join='inner')

df_all_sig_noise = (df_all_data / df_all_err).fillna(0)

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


#%%Make distributions plots of the top X peaks
feature_labels = df_all_data.columns.get_level_values(0)
top_features_hist(df_all_data.to_numpy(),25,logx=True,feature_labels=feature_labels,suptitle="Unscaled data log x scale",supxlabel='µgm3',supylabel='counts')
top_features_hist(df_all_data.to_numpy(),25,feature_labels=feature_labels,suptitle="Unscaled data linear x scale",supxlabel='µgm3',supylabel='counts')

top_features_hist(df_all_sig_noise.to_numpy(),25,logx=True,feature_labels=feature_labels,suptitle="Sig/noise log x scale",supylabel='counts')
top_features_hist(df_all_sig_noise.to_numpy(),25,feature_labels=feature_labels,suptitle="Sig/noise linear x scale",supylabel='counts')

top_peaks = cluster_extract_peaks(df_all_data.sum(axis=0), df_all_data.T,25,dropRT=False)[['peak_pct','Formula']]
top_peaks_sig_noise = cluster_extract_peaks(df_all_sig_noise.sum(axis=0), df_all_data.T,25,dropRT=False)[['peak_pct','Formula']]

#Do a pair plot for the top 10 features
import seaborn as sns
df_top10 = df_all_data[top_peaks.index[0:10]]
df_top10.columns = df_top10.columns.get_level_values(0) + ", " +  df_top10.columns.get_level_values(1).astype(str)
sns.pairplot(df_top10).fig.suptitle("Unscaled data", y=1.01)


df_top10_sig_noise = df_all_sig_noise[top_peaks_sig_noise.index[0:10]]
df_top10_sig_noise.columns = df_top10_sig_noise.columns.get_level_values(0) + ", " +  df_top10_sig_noise.columns.get_level_values(1).astype(str)
sns.pairplot(df_top10_sig_noise).fig.suptitle("Sig/noise data", y=1.01)

#%% Now apply standardscaler and see what happens
#We don't really want to do this as the features are all the same scale and so we want the big features to dominate
df_top10_ss = pd.DataFrame(StandardScaler().fit_transform(df_top10.to_numpy()),columns=df_top10.columns)
sns.pairplot(df_top10_ss).fig.suptitle("Unscaled data, StandardScaler", y=1.01)

df_top10_sig_noise_ss = pd.DataFrame(StandardScaler().fit_transform(df_top10_sig_noise.to_numpy()),columns=df_top10_sig_noise.columns)
sns.pairplot(df_top10_sig_noise_ss).fig.suptitle("Sig/noise, StandardScaler", y=1.01)

#%%Try a power transformation of the data
df_top10_yj = pd.DataFrame(PowerTransformer(method="yeo-johnson").fit_transform(df_top10.to_numpy()),columns=df_top10.columns)
sns.pairplot(df_top10_yj).fig.suptitle("Unscaled data, yeo-johnson", y=1.01,fontsize=20)

df_top10_sig_noise_yj = pd.DataFrame(PowerTransformer(method="yeo-johnson").fit_transform(df_top10_sig_noise.to_numpy()),columns=df_top10_sig_noise.columns)
sns.pairplot(df_top10_sig_noise_yj).fig.suptitle("Sig/noise, yeo-johnson", y=1.01,fontsize=20)

#%%Try a quantile transformation of the data
qt = QuantileTransformer(output_distribution="uniform")

df_top10_qt = pd.DataFrame(qt.fit_transform(df_top10.to_numpy()),columns=df_top10.columns)
sns.pairplot(df_top10_qt).fig.suptitle("Unscaled data, quantile transform", y=1.01,fontsize=20)

df_top10_sig_noise_qt = pd.DataFrame(qt.fit_transform(df_top10_sig_noise.to_numpy()),columns=df_top10_sig_noise.columns)
sns.pairplot(df_top10_sig_noise_qt).fig.suptitle("Sig/noise, quantile transform", y=1.01,fontsize=20)
