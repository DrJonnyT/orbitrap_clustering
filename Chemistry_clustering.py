# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:23:51 2022

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

from sklearn.preprocessing import StandardScaler, FunctionTransformer, RobustScaler
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
df_all_data, df_all_err, ds_all_mz = Load_pre_PMF_data(filepath)

#Save data to CSV
df_all_data.to_csv(r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\PMF data\df_all_data.csv")
df_all_err.to_csv(r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\PMF data\df_all_err.csv")
ds_all_mz.to_csv(r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\PMF data\ds_all_mz.csv",index=False,header=False)

pd.DataFrame(df_all_data.columns.get_level_values(0),df_all_data.columns.get_level_values(1)).to_csv(r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\PMF data\RT_formula.csv",header=False)

#Load all time data, ie start/mid/end
df_all_times = pd.read_csv(r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\PMF data\Times_all.csv")
df_all_times['date_start'] = pd.to_datetime(df_all_times['date_start'],dayfirst=True)
df_all_times['date_mid'] = pd.to_datetime(df_all_times['date_mid'],dayfirst=True)
df_all_times['date_end'] = pd.to_datetime(df_all_times['date_end'],dayfirst=True)

df_all_times.set_index(df_all_times['date_mid'],inplace=True)
fuzzy_index = pd.merge_asof(pd.DataFrame(index=df_all_data.index),df_all_times,left_index=True,right_index=True,direction='nearest',tolerance=pd.Timedelta(hours=1.25))
df_all_times = df_all_times.loc[fuzzy_index['date_mid']]

ds_dataset_cat = delhi_beijing_datetime_cat(df_all_data.index)

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

#%%Get a dataframe of the elements in each molecule
formulae = df_all_data.columns.get_level_values(0).to_numpy()

ds_formula_cat = pd.Categorical(np.zeros(len(formulae)),categories=['CHO','CHON','CHOS','CHONS'],ordered=True)

df_element_counts = pd.DataFrame(np.nan, index=formulae, columns=['C','H','N','O','S'])
df_element_xtoC = pd.DataFrame(np.nan, index=formulae, columns=['H','N','O','S'])

for i in range(len(formulae)):
    this_chemform = ChemForm(formulae[i])
    df_element_counts.iloc[i] = [this_chemform.C,this_chemform.H,this_chemform.N,this_chemform.O,this_chemform.S]
    df_element_xtoC.iloc[i] = np.array([this_chemform.H,this_chemform.N,this_chemform.O,this_chemform.S])/this_chemform.C
    
    if(this_chemform.S>0):
        if(this_chemform.N>0):
            ds_formula_cat[i] = 'CHONS'
        else:
            ds_formula_cat[i] = 'CHOS'
    elif(this_chemform.N>0):
        ds_formula_cat[i] = 'CHON'
    else:
        ds_formula_cat[i] = 'CHO'
    

#%%Clustering on this
#df_element_xtoC_SS = pd.DataFrame(StandardScaler().fit_transform(df_element_xtoC),index=df_element_xtoC.index)

#compare_cluster_metrics(df_element_xtoC_SS,2,10,'kmeans','Real space ',' metrics')

# #%%#Go with 5 clusters then
# num_clusters = 5
# clustering = KMeans(num_clusters).fit_predict(df_element_xtoC_SS)
# df_cluster_formulae = []
# for cluster in range(num_clusters):
#     formulae_this_cluster = formulae[clustering == cluster]
#     df_cluster_formulae.append(formulae_this_cluster)
#     pdb.set_trace()

#%%Just manually divide by chemistry
df_all_data_chem = pd.DataFrame(np.nan,index=df_all_data.index,columns=['CHO','CHON','CHOS','CHONS'])
df_all_data_chem['CHO'] = df_all_data[df_all_data.columns[ds_formula_cat=='CHO']].sum(axis=1)
df_all_data_chem['CHON'] = df_all_data[df_all_data.columns[ds_formula_cat=='CHON']].sum(axis=1)
df_all_data_chem['CHOS'] = df_all_data[df_all_data.columns[ds_formula_cat=='CHOS']].sum(axis=1)
df_all_data_chem['CHONS'] = df_all_data[df_all_data.columns[ds_formula_cat=='CHONS']].sum(axis=1)

all_data_chem_SS = StandardScaler().fit_transform(df_all_data_chem)
compare_cluster_metrics(df_all_data_chem,2,10,'agglom','Real space ',' metrics')