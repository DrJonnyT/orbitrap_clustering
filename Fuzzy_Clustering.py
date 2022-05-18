# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 11:35:33 2022

@author: mbcx5jt5
"""

"""
Created on Fri Nov  5 14:37:36 2021

@author: mbcx5jt5
"""
#Set random seed for repeatability
from numpy.random import seed
seed(1337)
import tensorflow as tf
tf.random.set_seed(1338)

import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow.keras as keras
import kerastuner as kt

from sklearn.preprocessing import RobustScaler, StandardScaler,FunctionTransformer,MinMaxScaler
from sklearn.pipeline import Pipeline


from sklearn.metrics.cluster import contingency_matrix

import scipy.cluster.hierarchy as sch
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score,silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, NMF
from sklearn import metrics

from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as plticker

import skfuzzy as fuzz

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pickle

import os
os.chdir('C:/Work/Python/Github/Orbitrap_clustering')
from ae_functions import *


# %%Load data

# %%Load data

path='C:/Users/mbcx5jt5/Google Drive/Shared_York_Man2/'
df_beijing_data, df_beijing_err, df_beijing_metadata, df_beijing_raw = beijing_load(
    path + 'BJ_UnAmbNeg9.1.1_20210505-Times_Fixed.xlsx',path + 'BJ_UnAmbNeg9.1.1_20210505-Times_Fixed.xlsx',
    peaks_sheetname="Compounds",metadata_sheetname="massloading_Beijing")

df_delhi_data, df_delhi_err, df_delhi_metadata, df_delhi_raw = delhi_load2(path + '/Delhi/Orbitrap/')

df_all_data = pd.concat([df_beijing_data, df_delhi_data], axis=0, join="inner")
df_all_err = pd.concat([df_beijing_err, df_delhi_err], axis=0, join="inner")
df_all_raw = pd.concat([df_beijing_raw, df_delhi_raw], axis=1, join="inner")
df_all_raw = df_all_raw.loc[:,~df_all_raw.columns.duplicated()] #Remove duplicate columns: m/z, RT, molecular weight, formula

dataset_cat = delhi_beijing_datetime_cat(df_all_data)
df_dataset_cat = pd.DataFrame(delhi_beijing_datetime_cat(df_all_data),columns=['dataset_cat'],index=df_all_data.index)
ds_dataset_cat = df_dataset_cat['dataset_cat']

time_cat = delhi_calc_time_cat(df_all_data)
df_time_cat = pd.DataFrame(delhi_calc_time_cat(df_all_data),columns=['time_cat'],index=df_all_data.index)
ds_time_cat = df_time_cat['time_cat']

mz_columns = pd.DataFrame(df_all_raw['Molecular Weight'].loc[df_all_data.columns])


#Sort columns by m/z
mz_columns_sorted = mz_columns.sort_values("Molecular Weight",axis=0)
df_all_data.columns= mz_columns['Molecular Weight']
df_all_data.sort_index(axis=1,inplace=True)
df_all_data.columns = mz_columns_sorted.index
mz_columns = mz_columns_sorted

# %%Check largest peaks
chemform_namelist_beijing = load_chemform_namelist(path + 'Beijing_Amb3.1_MZ.xlsx')
chemform_namelist_delhi = load_chemform_namelist(path + 'Delhi_Amb3.1_MZ.xlsx')
chemform_namelist_all = combine_chemform_namelists(chemform_namelist_beijing,chemform_namelist_delhi)




#%%

#Augment the data
df_aug = augment_data_noise(df_beijing_filters,50,1,0)


#%%Scale data for input into AE
scalefactor = 1e6
pipe_1e6 = FunctionTransformer(lambda x: np.divide(x,scalefactor),inverse_func = lambda x: np.multiply(x,scalefactor))
pipe_1e6.fit(df_aug)
scaled_df = pd.DataFrame(pipe_1e6.transform(df_aug),columns=df_beijing_filters.columns)
ae_input=scaled_df.to_numpy()
scaled_df_val = pd.DataFrame(pipe_1e6.transform(df_beijing_filters), columns=df_beijing_filters.columns,index=df_beijing_filters.index)
ae_input_val = scaled_df_val.to_numpy()

df_beijing_summer_1e6 = pd.DataFrame(pipe_1e6.transform(df_beijing_summer),columns=df_beijing_summer.columns)

minmax_Beijing = MinMaxScaler()
beijing_filters_minmax = minmax_Beijing.fit_transform(df_beijing_filters.to_numpy())

#df scaled so it is normalised by the total from each filter
df_beijing_summer_norm = df_beijing_summer_1e6.div(df_beijing_summer_1e6.sum(axis=1), axis=0)

#%%Make dataframe with top 70% of data signal
df_scaled_top70 = extract_top_npercent(scaled_df_val,70,plot=True)
scaled_top70_np = df_scaled_top70.to_numpy()

df_beijing_summer_1e6_top70 = extract_top_npercent(df_beijing_summer_1e6,70)


#%%PCA transform the native dataset
pca7 = PCA(n_components = 7)
beijing_filters_PCA7_space = pca7.fit_transform(pipe_1e6.transform(df_beijing_summer_1e6))

#%%Fuzzy clustering of PCA7 space
num_clusters = 5
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        beijing_filters_PCA7_space.transpose(), num_clusters, 2, error=0.005, maxiter=1000, init=None)
cluster_membership = np.argmax(u, axis=0)

#%%t-SNE of data and fuzzy clustering
#Make an array with the data and cluster centers in
tsne = TSNE(n_components=2, random_state=0)
tsne_input = np.concatenate((cntr,beijing_filters_PCA7_space),axis=0)
tsne_output = tsne.fit_transform(tsne_input)
tsne_centers, tsne_data = np.array_split(tsne_output,[num_clusters],axis=0)


colormap = ['k','blue','red','yellow','gray','purple','aqua','gold','orange']

plt.scatter(tsne_data[:, 0], tsne_data[:, 1],
            c=cluster_membership,
            cmap=ListedColormap(colormap[0:num_clusters]))
plt.scatter(tsne_centers[:,0], tsne_centers[:,1],c='k',marker='x',s=250)
plt.show()



plt.scatter(tsne_data[:, 0], tsne_data[:, 1],
            c=u[2,:], cmap='tab20'            )

#%%

pca2 = PCA(n_components = 2)
pca2_data = pca2.fit_transform(beijing_filters_PCA7_space)
pca2_centers = pca2.transform(cntr)

plt.scatter(pca2_data[:, 0], pca2_data[:, 1],
            c=cluster_membership,
            cmap=ListedColormap(colormap[0:num_clusters]))
plt.scatter(pca2_centers[:,0], pca2_centers[:,1],c='k',marker='x',s=250)
plt.show()





#%%Fuzzy clustering of top70% of dataset
num_clusters = 5
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        df_beijing_summer_1e6_top70.to_numpy().transpose(), num_clusters, 3, error=0.005, maxiter=1000, init=None)
cluster_membership = np.argmax(u, axis=0)

df_beijing_summer_scaled_top70_fclusters_mtx = pd.DataFrame((df_beijing_summer_1e6_top70.to_numpy().sum(axis=1) * u).transpose())
df_beijing_summer_scaled_top70_fclusters_mtx.columns = [("fclust"+str(num)) for num in range(num_clusters)]
df_beijing_summer_scaled_top70_fclusters_mtx.index = df_beijing_summer_1e6_top70.index


#%%t-SNE of data and fuzzy clustering
#Make an array with the data and cluster centers in
tsne = TSNE(n_components=2)
tsne_input = np.concatenate((cntr,df_beijing_summer_1e6_top70.to_numpy()),axis=0)
tsne_output = tsne.fit_transform(tsne_input)
tsne_centers, tsne_data = np.array_split(tsne_output,[num_clusters],axis=0)


colormap = ['k','blue','red','yellow','gray','purple','aqua','gold','orange']

plt.scatter(tsne_data[:, 0], tsne_data[:, 1],
            c=cluster_membership,
            cmap=ListedColormap(colormap[0:num_clusters]))
plt.scatter(tsne_centers[:,0], tsne_centers[:,1],c='k',marker='x',s=250)
plt.show()



# plt.scatter(tsne_data[:, 0], tsne_data[:, 1],
#             c=u[2,:], cmap='tab20'            )

#%%

pca2 = PCA(n_components = 2)
pca2_data = pca2.fit_transform(df_beijing_summer_1e6_top70.to_numpy())
pca2_centers = pca2.transform(cntr)

plt.scatter(pca2_data[:, 0], pca2_data[:, 1],
            c=cluster_membership,
            cmap=ListedColormap(colormap[0:num_clusters]))
plt.scatter(pca2_centers[:,0], pca2_centers[:,1],c='k',marker='x',s=250)
plt.show()



#%%Load AQ data
#Load the met data
df_merge_beijing_summer = pd.read_csv(path+'aphh_summer_filter_aggregate_merge.csv')
df_merge_beijing_summer["DateTime"] =pd.to_datetime(df_merge_beijing_summer["date_mid"])
df_merge_beijing_summer.set_index('DateTime',inplace=True)
df_merge_beijing_summer['date_start'] = pd.to_datetime(df_merge_beijing_summer['date_start'])
df_merge_beijing_summer['date_end'] = pd.to_datetime(df_merge_beijing_summer['date_end'])
df_merge_beijing_summer['time_cat'] = pd.Categorical(delhi_calc_time_cat(df_merge_beijing_summer),['Morning','Midday' ,'Afternoon','Night'], ordered=True)


#Photochemical age. Original calculation from Parrish (1992)
k_toluene = 5.63e-12
k_benzene = 1.22e-12
OH_conc = 1.5e6
df_merge_beijing_summer["toluene_over_benzene_syft"]=  df_merge_beijing_summer["toluene_ppb_syft"] / df_merge_beijing_summer["benzene._ppb_syft"]
df_merge_beijing_summer["toluene_over_benzene_syft"].values[df_merge_beijing_summer["toluene_over_benzene_syft"] > 5] = np.nan#Remove one big spike
#benzene/tolluene emission ratio
benzene_tolluene_ER = df_merge_beijing_summer["toluene_over_benzene_syft"].quantile(0.99)

df_merge_beijing_summer["photochem_age_h"] = 1/(3600*OH_conc*(k_toluene-k_benzene))  * (np.log(benzene_tolluene_ER) - np.log(df_merge_beijing_summer["toluene_over_benzene_syft"]))

df_merge_beijing_summer["nox_over_noy"] = df_merge_beijing_summer["nox_ppbv"] / df_merge_beijing_summer["noy_ppbv"]
df_merge_beijing_summer["-log10_nox/noy"] = - np.log10(df_merge_beijing_summer["nox_over_noy"])

#AMS fractions
df_merge_beijing_summer["Total_ams"] = df_merge_beijing_summer["Org_ams"] + df_merge_beijing_summer["NO3_ams"] + df_merge_beijing_summer["SO4_ams"] + df_merge_beijing_summer["NH4_ams"] + df_merge_beijing_summer["Chl_ams"]
df_merge_beijing_summer["Org_ams_frac"] = df_merge_beijing_summer["Org_ams"] / df_merge_beijing_summer["Total_ams"]
df_merge_beijing_summer["NO3_ams_frac"] = df_merge_beijing_summer["NO3_ams"] / df_merge_beijing_summer["Total_ams"]
df_merge_beijing_summer["SO4_ams_frac"] = df_merge_beijing_summer["SO4_ams"] / df_merge_beijing_summer["Total_ams"]
df_merge_beijing_summer["NH4_ams_frac"] = df_merge_beijing_summer["NH4_ams"] / df_merge_beijing_summer["Total_ams"]
df_merge_beijing_summer["Chl_ams_frac"] = df_merge_beijing_summer["Chl_ams"] / df_merge_beijing_summer["Total_ams"]

df_merge_beijing_summer["OOA1_ams_frac"] = df_merge_beijing_summer["OOA1_ams"] / df_merge_beijing_summer["Org_ams"]
df_merge_beijing_summer["OOA2_ams_frac"] = df_merge_beijing_summer["OOA2_ams"] / df_merge_beijing_summer["Org_ams"]
df_merge_beijing_summer["OOA3_ams_frac"] = df_merge_beijing_summer["OOA3_ams"] / df_merge_beijing_summer["Org_ams"]
df_merge_beijing_summer["HOA_ams_frac"] = df_merge_beijing_summer["HOA_ams"] / df_merge_beijing_summer["Org_ams"]
df_merge_beijing_summer["COA_ams_frac"] = df_merge_beijing_summer["COA_ams"] / df_merge_beijing_summer["Org_ams"]

#%%Add in filters total and fuzzy clusters
df_merge_beijing_summer = pd.concat([df_merge_beijing_summer, df_beijing_summer_1e6.sum(axis=1)], axis=1).reindex(df_beijing_summer_1e6.index)
df_merge_beijing_summer['filters_total'] = df_merge_beijing_summer[0]
df_merge_beijing_summer.drop(columns=0,inplace=True)

#df_merge_beijing_summer = pd.concat([df_merge_beijing_summer,df_beijing_summer_scaled_top70_fclusters_mtx],axis=1)


#%%Testing fclust correlations
beijing_summer_scaled_top70_fclust_corr = corr_2df(df_merge_beijing_summer,df_beijing_summer_scaled_top70_fclusters_mtx)
fig, ax = plt.subplots(figsize=(20,25)) 
sns.heatmap(beijing_summer_scaled_top70_fclust_corr,ax=ax)
plt.show()

plt.scatter(df_beijing_summer_scaled_top70_fclusters_mtx['fclust4'],df_merge_beijing_summer['photochem_age_h'])
plt.show()

#%%Testing NMF on top70 dataframe


def get_score(model, data, scorer=metrics.explained_variance_score):
    """ Estimate performance of the model on the data """
    prediction = model.inverse_transform(model.transform(data))
    return scorer(data, prediction)


#Work out how many factors
ks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
perfs_train = []
for k in ks:
    nmf = NMF(n_components=k).fit(df_beijing_summer_1e6_top70.clip(lower=0))
    perfs_train.append(get_score(nmf, df_beijing_summer_1e6_top70.clip(lower=0)))
print(perfs_train)

##Go with 5 for now? But need 13 to explain 90% of variance

#%%5-component Orbitrap PMF
n_components = 5
model = NMF(n_components=n_components)
#a = model.fit(df_beijing_summer_1e6_top70.clip(lower=0))
W = model.fit_transform(df_beijing_summer_1e6_top70.clip(lower=0))
H = model.components_

#1 What is the time series of the 2 factors? Need each factor as a t series
# Factor0 = H[0]
# Factor1 = H[1]
# Factor2 = H[2]
# Factor2 = H[3]
# Factor2 = H[4]

# Factor0_mtx = np.outer(W.T[0], H[0])
# Factor1_mtx = np.outer(W.T[1], H[1])
# Factor2_mtx = np.outer(W.T[2], H[2])
# Factor2_mtx = np.outer(W.T[3], H[3])
# Factor2_mtx = np.outer(W.T[4], H[4])

Factors_totals = np.ndarray(W.shape)

for x in np.arange(n_components):
    Factors_totals[:,x] =  (np.outer(W.T[x], H[x]).sum(axis=1))
    
df_factor_totals = pd.DataFrame(Factors_totals)
df_factor_totals.columns = [("factor"+str(num)) for num in range(n_components)]
df_factor_totals.index = df_merge_beijing_summer.index

df_factor_fractions = df_factor_totals.div(df_factor_totals.sum(axis=1),axis=0)
df_factor_fractions.columns = [("factor"+str(num)+"_frac") for num in range(n_components)]

nmf_aq_corr = corr_2df(df_merge_beijing_summer,df_factor_totals)

fig, ax = plt.subplots(figsize=(20,25)) 
sns.heatmap(nmf_aq_corr,ax=ax)
plt.show()


#plt.scatter(df_factor_totals['factor3'],df_merge_beijing_summer['o3_ppbv'])

# fig,ax1 = plt.subplots()
# ax1.plot(df_merge_beijing_summer['o3_ppbv'])
# ax2 = ax1.twinx()
# ax2.plot(df_merge_beijing_summer['filters_total'],c='k')
# plt.show()

#%%Generate best correlations for some AMS PMF factors with varying numbers of orbitrap nmf factors

n_components = [2,3,4,5,6,7,8,9,10]
AMS_PMF_max_R = pd.DataFrame(index=n_components,columns=['HOA_ams','COA_ams','OOA1_ams','OOA2_ams','OOA3_ams'])
AMS_PMF_min_R = pd.DataFrame(index=n_components,columns=['HOA_ams','COA_ams','OOA1_ams','OOA2_ams','OOA3_ams'])
AMS_PMF_frac_max_R = pd.DataFrame(index=n_components,columns=['HOA_ams_frac','COA_ams_frac','OOA1_ams_frac','OOA2_ams_frac','OOA3_ams_frac'])
AMS_PMF_frac_min_R = pd.DataFrame(index=n_components,columns=['HOA_ams_frac','COA_ams_frac','OOA1_ams_frac','OOA2_ams_frac','OOA3_ams_frac'])

for nfact in n_components:
    model = NMF(n_components=nfact)
    W = model.fit_transform(df_beijing_summer_1e6_top70.clip(lower=0))
    H = model.components_
    Factors_totals = np.ndarray(W.shape)

    for x in np.arange(nfact):
        Factors_totals[:,x] =  (np.outer(W.T[x], H[x]).sum(axis=1))
    
    df_factor_totals = pd.DataFrame(Factors_totals)
    df_factor_totals.columns = [("factor"+str(num)) for num in range(nfact)]
    df_factor_totals.index = df_merge_beijing_summer.index

    df_factor_fractions = df_factor_totals.div(df_factor_totals.sum(axis=1),axis=0)
    df_factor_fractions.columns = [("factor"+str(num)+"_frac") for num in range(nfact)]

    nmf_aq_corr = corr_2df(df_merge_beijing_summer,df_factor_totals)
    
    AMS_PMF_max_R.loc[nfact]['HOA_ams'] = corr_2df(pd.DataFrame(df_merge_beijing_summer['HOA_ams']),df_factor_totals).max().max()
    AMS_PMF_max_R.loc[nfact]['COA_ams'] = corr_2df(pd.DataFrame(df_merge_beijing_summer['COA_ams']),df_factor_totals).max().max()
    AMS_PMF_max_R.loc[nfact]['OOA1_ams'] = corr_2df(pd.DataFrame(df_merge_beijing_summer['OOA1_ams']),df_factor_totals).max().max()
    AMS_PMF_max_R.loc[nfact]['OOA2_ams'] = corr_2df(pd.DataFrame(df_merge_beijing_summer['OOA2_ams']),df_factor_totals).max().max()
    AMS_PMF_max_R.loc[nfact]['OOA3_ams'] = corr_2df(pd.DataFrame(df_merge_beijing_summer['OOA3_ams']),df_factor_totals).max().max()
    
    AMS_PMF_min_R.loc[nfact]['HOA_ams'] = corr_2df(pd.DataFrame(df_merge_beijing_summer['HOA_ams']),df_factor_totals).min().min()
    AMS_PMF_min_R.loc[nfact]['COA_ams'] = corr_2df(pd.DataFrame(df_merge_beijing_summer['COA_ams']),df_factor_totals).min().min()
    AMS_PMF_min_R.loc[nfact]['OOA1_ams'] = corr_2df(pd.DataFrame(df_merge_beijing_summer['OOA1_ams']),df_factor_totals).min().min()
    AMS_PMF_min_R.loc[nfact]['OOA2_ams'] = corr_2df(pd.DataFrame(df_merge_beijing_summer['OOA2_ams']),df_factor_totals).min().min()
    AMS_PMF_min_R.loc[nfact]['OOA3_ams'] = corr_2df(pd.DataFrame(df_merge_beijing_summer['OOA3_ams']),df_factor_totals).min().min()
    
    AMS_PMF_frac_max_R.loc[nfact]['HOA_ams_frac'] = corr_2df(pd.DataFrame(df_merge_beijing_summer['HOA_ams_frac']),df_factor_totals).max().max()
    AMS_PMF_frac_max_R.loc[nfact]['COA_ams_frac'] = corr_2df(pd.DataFrame(df_merge_beijing_summer['COA_ams_frac']),df_factor_totals).max().max()
    AMS_PMF_frac_max_R.loc[nfact]['OOA1_ams_frac'] = corr_2df(pd.DataFrame(df_merge_beijing_summer['OOA1_ams_frac']),df_factor_totals).max().max()
    AMS_PMF_frac_max_R.loc[nfact]['OOA2_ams_frac'] = corr_2df(pd.DataFrame(df_merge_beijing_summer['OOA2_ams_frac']),df_factor_totals).max().max()
    AMS_PMF_frac_max_R.loc[nfact]['OOA3_ams_frac'] = corr_2df(pd.DataFrame(df_merge_beijing_summer['OOA3_ams_frac']),df_factor_totals).max().max()

    AMS_PMF_frac_min_R.loc[nfact]['HOA_ams_frac'] = corr_2df(pd.DataFrame(df_merge_beijing_summer['HOA_ams_frac']),df_factor_totals).min().min()
    AMS_PMF_frac_min_R.loc[nfact]['COA_ams_frac'] = corr_2df(pd.DataFrame(df_merge_beijing_summer['COA_ams_frac']),df_factor_totals).min().min()
    AMS_PMF_frac_min_R.loc[nfact]['OOA1_ams_frac'] = corr_2df(pd.DataFrame(df_merge_beijing_summer['OOA1_ams_frac']),df_factor_totals).min().min()
    AMS_PMF_frac_min_R.loc[nfact]['OOA2_ams_frac'] = corr_2df(pd.DataFrame(df_merge_beijing_summer['OOA2_ams_frac']),df_factor_totals).min().min()
    AMS_PMF_frac_min_R.loc[nfact]['OOA3_ams_frac'] = corr_2df(pd.DataFrame(df_merge_beijing_summer['OOA3_ams_frac']),df_factor_totals).min().min()

#    AMS_PMF_max_R2 = AMS_PMF_max_R**2
#    AMS_PMF_frac_max_R2 = AMS_PMF_frac_max_R**2
    

#%%Plot correlation between orbitrap and AMS PMF
AMS_PMF_max_R.plot(figsize=(10,6),ylabel='Max R', xlabel='Num orbitrap components',title='Best correlation R between AMS and Orbitrap (top 70%) PMF components')
AMS_PMF_frac_max_R.plot(figsize=(10,6),ylabel='Max R', xlabel='Num orbitrap components',title='Best correlation R between AMS and Orbitrap (top 70%) fractional PMF components')


AMS_PMF_min_R.plot(figsize=(10,6),ylabel='Min R', xlabel='Num orbitrap components',title='Best anticorrelation R between AMS and Orbitrap (top 70%) PMF components')
AMS_PMF_frac_min_R.plot(figsize=(10,6),ylabel='Min R', xlabel='Num orbitrap components',title='Best anticorrelation R between AMS and Orbitrap (top 70%) fractional PMF components')



#%%Basic correlations of all mz/RT in the dataset with AQ data
big_corr_top70 = corr_2df(df_merge_beijing_summer,df_beijing_summer_1e6_top70)
fig, ax = plt.subplots(figsize=(50,50)) 
sns.heatmap(big_corr_top70,ax=ax)

big_corr_norm = corr_2df(df_merge_beijing_summer,df_beijing_summer_norm)
fig, ax = plt.subplots(figsize=(50,50)) 
sns.heatmap(big_corr_norm,ax=ax)

frac_corr_norm = corr_2df(df_merge_beijing_summer.filter(like='frac'),df_beijing_summer_norm)
fig, ax = plt.subplots(figsize=(150,8)) 
sns.heatmap(frac_corr_norm,ax=ax)

#Find the molecules that best correlate with the fractions of the PMF factors
HOA_peaks_frac = frac_corr_norm.loc['HOA_ams_frac']
HOA_peaks_frac = top_corr_peaks(HOA_peaks_frac,chemform_namelist_all,50,dp=2)

COA_peaks_frac = frac_corr_norm.loc['COA_ams_frac']
COA_peaks_frac = top_corr_peaks(COA_peaks_frac,chemform_namelist_all,50,dp=2)

OOA1_peaks_frac = frac_corr_norm.loc['OOA1_ams_frac']
OOA1_peaks_frac = top_corr_peaks(OOA1_peaks_frac,chemform_namelist_all,50,dp=2)
OOA2_peaks_frac = frac_corr_norm.loc['OOA2_ams_frac']
OOA2_peaks_frac = top_corr_peaks(OOA2_peaks_frac,chemform_namelist_all,50,dp=2)
OOA3_peaks_frac = frac_corr_norm.loc['OOA3_ams_frac']
OOA3_peaks_frac = top_corr_peaks(OOA3_peaks_frac,chemform_namelist_all,50,dp=2)


pmf_corr = corr_2df(df_merge_beijing_summer.filter(regex="_ams$"),df_factor_totals)
pmf_frac_corr = corr_2df(df_merge_beijing_summer.filter(like='frac'),df_factor_fractions)





#%%Calculate PMF and fclust fractions by time of day classification
# %%Munge Orbitrap nmf data for time_cat analysis
#df_beijing_summer_scaled_top70_fclusters_cat_stats = df_beijing_summer_scaled_top70_fclusters_mtx.groupby(df_merge_beijing_summer['time_cat']).describe()
#df_beijing_summer_scaled_top70_nmf_cat_stats = df_factor_fractions.groupby(df_merge_beijing_summer['time_cat']).describe()

df_fclust_cat_mean = df_beijing_summer_scaled_top70_fclusters_mtx[['fclust0','fclust1','fclust2','fclust3','fclust4']].groupby(df_merge_beijing_summer['time_cat']).mean()
df_fclust_cat_std = df_beijing_summer_scaled_top70_fclusters_mtx[['fclust0','fclust1','fclust2','fclust3','fclust4']].groupby(df_merge_beijing_summer['time_cat']).std()
df_fclust_cat_mean_norm = df_fclust_cat_mean.div(df_fclust_cat_mean.sum(axis=1),axis=0)

df_nmf_cat_mean = df_factor_totals[['factor0','factor1','factor2','factor3','factor4']].groupby(df_merge_beijing_summer['time_cat']).mean()
df_nmf_cat_std = df_factor_totals[['factor0','factor1','factor2','factor3','factor4']].groupby(df_merge_beijing_summer['time_cat']).std()
df_nmf_cat_mean_norm = df_nmf_cat_mean.div(df_nmf_cat_mean.sum(axis=1),axis=0)





#%%nmf Line plot
fig,ax = plt.subplots(2,1,figsize=(7,10))
ax1=ax[0]
ax2=ax[1]
ax1.set_title('Orbitrap PMF, 5 factors')

ax1.plot(df_nmf_cat_mean.index, df_nmf_cat_mean['factor0'], linewidth=5,c='b',label='factor0')
ax1.plot(df_nmf_cat_mean.index, df_nmf_cat_mean['factor1'], linewidth=5,c='lime',label='factor1')
ax1.plot(df_nmf_cat_mean.index, df_nmf_cat_mean['factor2'], linewidth=5,c='r',label='factor2')
ax1.plot(df_nmf_cat_mean.index, df_nmf_cat_mean['factor3'], linewidth=5,c='orange',label='factor3')
ax1.plot(df_nmf_cat_mean.index, df_nmf_cat_mean['factor4'], linewidth=5,c='pink',label='factor4')
ax1.set_ylabel('µg m$^{-3}$')
ax1.set_ylim(0,)
ax1.legend(bbox_to_anchor=(1.22, 0.7))

ax2.stackplot(df_nmf_cat_mean.index,df_nmf_cat_mean_norm['factor0'], df_nmf_cat_mean_norm['factor1'],
              df_nmf_cat_mean_norm['factor2'],df_nmf_cat_mean_norm['factor3'],
              df_nmf_cat_mean_norm['factor4'], labels=['factor0','factor1','factor2','factor3','factor4'],
              colors=['b','lime','r','orange','pink'])
ax2.set_ylabel('Fraction')
ax2.set_ylim(0,)
ax2.legend(bbox_to_anchor=(1.22, 0.7))



#%%fclust Line plot
fig,ax = plt.subplots(2,1,figsize=(7,10))
ax1=ax[0]
ax2=ax[1]
ax1.set_title('Orbitrap PMF, 5 fclusts')

ax1.plot(df_fclust_cat_mean.index, df_fclust_cat_mean['fclust0'], linewidth=5,c='b',label='fclust0')
ax1.plot(df_fclust_cat_mean.index, df_fclust_cat_mean['fclust1'], linewidth=5,c='lime',label='fclust1')
ax1.plot(df_fclust_cat_mean.index, df_fclust_cat_mean['fclust2'], linewidth=5,c='r',label='fclust2')
ax1.plot(df_fclust_cat_mean.index, df_fclust_cat_mean['fclust3'], linewidth=5,c='orange',label='fclust3')
ax1.plot(df_fclust_cat_mean.index, df_fclust_cat_mean['fclust4'], linewidth=5,c='pink',label='fclust4')
ax1.set_ylabel('µg m$^{-3}$')
ax1.set_ylim(0,)
ax1.legend(bbox_to_anchor=(1.22, 0.7))

ax2.stackplot(df_fclust_cat_mean.index,df_fclust_cat_mean_norm['fclust0'], df_fclust_cat_mean_norm['fclust1'],
              df_fclust_cat_mean_norm['fclust2'],df_fclust_cat_mean_norm['fclust3'],
              df_fclust_cat_mean_norm['fclust4'], labels=['fclust0','fclust1','fclust2','fclust3','fclust4'],
              colors=['b','lime','r','orange','pink'])
ax2.set_ylabel('Fraction')
ax2.set_ylim(0,)
ax2.legend(bbox_to_anchor=(1.22, 0.7))



#%%Trying to recreate Sari's graph from her IAC2022 abstract
#7-component Orbitrap PMF
n_components = 7
model = NMF(n_components=n_components)
#a = model.fit(df_beijing_summer_1e6_top70.clip(lower=0))
W = model.fit_transform(df_beijing_summer_1e6.clip(lower=0))
H = model.components_

#1 What is the time series of the 2 factors? Need each factor as a t series
# Factor0 = H[0]
# Factor1 = H[1]
# Factor2 = H[2]
# Factor2 = H[3]
# Factor2 = H[4]

# Factor0_mtx = np.outer(W.T[0], H[0])
# Factor1_mtx = np.outer(W.T[1], H[1])
# Factor2_mtx = np.outer(W.T[2], H[2])
# Factor2_mtx = np.outer(W.T[3], H[3])
# Factor2_mtx = np.outer(W.T[4], H[4])
df_nmf_factors = pd.DataFrame(H,columns=df_beijing_summer_1e6.columns)
Factors_totals = np.ndarray(W.shape)

for x in np.arange(n_components):
    Factors_totals[:,x] =  (np.outer(W.T[x], H[x]).sum(axis=1))
    
df_factor_totals = pd.DataFrame(Factors_totals)
df_factor_totals.columns = [("factor"+str(num)) for num in range(n_components)]
df_factor_totals.index = df_beijing_summer.index

df_factor_fractions = df_factor_totals.div(df_factor_totals.sum(axis=1),axis=0)
df_factor_fractions.columns = [("factor"+str(num)+"_frac") for num in range(n_components)]


#%%Plot Sari style PMF
fig,ax = plt.subplots(7,1,figsize=(10,10))
df_factor_totals['factor0'].plot(ax=ax[0])
df_factor_totals['factor1'].plot(ax=ax[1])
df_factor_totals['factor2'].plot(ax=ax[2])
df_factor_totals['factor3'].plot(ax=ax[3])
df_factor_totals['factor4'].plot(ax=ax[4])
df_factor_totals['factor5'].plot(ax=ax[5])
df_factor_totals['factor6'].plot(ax=ax[6])
plt.show()


#%%
mz_columns = pd.DataFrame(df_beijing_raw['Molecular Weight'].loc[df_nmf_factors.columns])
fig,ax = plt.subplots(7,1,figsize=(10,10))
ax[0].stem(mz_columns.to_numpy(),df_nmf_factors.loc[0],markerfmt=' ')
ax[1].stem(mz_columns.to_numpy(),df_nmf_factors.loc[1],markerfmt=' ')
ax[2].stem(mz_columns.to_numpy(),df_nmf_factors.loc[2],markerfmt=' ')
ax[3].stem(mz_columns.to_numpy(),df_nmf_factors.loc[3],markerfmt=' ')
ax[4].stem(mz_columns.to_numpy(),df_nmf_factors.loc[4],markerfmt=' ')
ax[5].stem(mz_columns.to_numpy(),df_nmf_factors.loc[5],markerfmt=' ')
ax[6].stem(mz_columns.to_numpy(),df_nmf_factors.loc[6],markerfmt=' ')
ax[0].set_xlim(right=500)
ax[1].set_xlim(right=500)
ax[2].set_xlim(right=500)
ax[3].set_xlim(right=500)
ax[4].set_xlim(right=500)
ax[5].set_xlim(right=500)
ax[6].set_xlim(right=500)
plt.show()


#%%
Can you not just do nmf all the time on everyting? Make sure that the autoencoder is all positive
Make sure that the PCA is all positive
Normalise or whatever
Run NMF
Then normalise that output, and use that to scale the total? And then you can correlate. Can you pick out anything useful though? From the factor profile?
No? Like if you want to work out the useful molecules, can you reasonably do that?
You could use a bad autoencoder? Like just a linear one? And limit the dimensions
