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
from sklearn.decomposition import PCA

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

path='C:/Users/mbcx5jt5/Google Drive/Shared_York_Man2/'
df_beijing_raw, df_beijing_filters, df_beijing_metadata = beijing_load(
    path + 'BJ_UnAmbNeg9.1.1_20210505-Times_Fixed.xlsx',path + 'BJ_UnAmbNeg9.1.1_20210505-Times_Fixed.xlsx',
    peaks_sheetname="Compounds",metadata_sheetname="massloading_Beijing")

df_delhi_raw, df_delhi_filters, df_delhi_metadata = delhi_load(
    path + 'Delhi_Amb3.1_MZ.xlsx',path + 'Delhi/Delhi_massloading_autumn_summer.xlsx')

df_beijing_winter = df_beijing_filters.iloc[0:124].copy()
df_beijing_summer = df_beijing_filters.iloc[124:].copy()

df_all_filters = df_beijing_filters.append(df_delhi_filters,sort=True).fillna(0)
df_all_raw = df_beijing_raw.transpose().append(df_delhi_raw.transpose(),sort=True).transpose()

# %%Check largest peaks

#IM STILLL WORKING OUT HOW TO JOIN THEM ALL TOGETHER BEARING IN MIND THAT NOT ALL MOLECULES ARE THE SAME IN THE NAMELISTS
#YOU END UP WITH SOME AS A LIST, SOME AS [], SOME AS THE RIGHT MOLECULE, ITS QUITE ANNOYING AND I WANT TO JUST USE A FOR LOOP
#LIKE A NORMAL PERSON

chemform_namelist_beijing = load_chemform_namelist(path + 'Beijing_Amb3.1_MZ.xlsx')
chemform_namelist_delhi = load_chemform_namelist(path + 'Delhi_Amb3.1_MZ.xlsx')



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


#Photochemical age. Original calculation from Parrish (1992)
k_toluene = 5.63e-12
k_benzene = 1.22e-12
OH_conc = 1.5e6
df_merge_beijing_summer["toluene_over_benzene_syft"]=  df_merge_beijing_summer["toluene_ppb_syft"] / df_merge_beijing_summer["benzene._ppb_syft"]
df_merge_beijing_summer["toluene_over_benzene_syft"].values[df_merge_beijing_summer["toluene_over_benzene_syft"] > 5] = np.nan#Remove one big spike
#benzene/tolluene emission ratio
benzene_tolluene_ER = df_merge_beijing_summer["toluene_over_benzene_syft"].quantile(0.99)

df_merge_beijing_summer["Photochem_age_h"] = 1/(3600*OH_conc*(k_toluene-k_benzene))  * (np.log(benzene_tolluene_ER) - np.log(df_merge_beijing_summer["toluene_over_benzene_syft"]))

df_merge_beijing_summer["nox_over_noy"] = df_merge_beijing_summer["nox_ppbv"] / df_merge_beijing_summer["noy_ppbv"]
df_merge_beijing_summer["-log10_nox/noy"] = - np.log10(df_merge_beijing_summer["nox_over_noy"])

#%%Add in filters total and fuzzy clusters
df_merge_beijing_summer = pd.concat([df_merge_beijing_summer, df_beijing_summer_1e6.sum(axis=1)], axis=1).reindex(df_beijing_summer_1e6.index)
df_merge_beijing_summer['filters_total'] = df_merge_beijing_summer[0]
df_merge_beijing_summer.drop(columns=0,inplace=True)

df_merge_beijing_summer = pd.concat([df_merge_beijing_summer,df_beijing_summer_scaled_top70_fclusters_mtx],axis=1)


#%%Testing fclust correlations
beijing_summer_scaled_top70_fclust_corr = corr_2df(df_merge_beijing_summer,df_beijing_summer_scaled_top70_fclusters_mtx)
fig, ax = plt.subplots(figsize=(20,20)) 
sns.heatmap(beijing_summer_scaled_top70_fclust_corr,ax=ax)

plt.scatter(df_beijing_summer_scaled_top70_fclusters_mtx['fclust4'],df_merge_beijing_summer['o3_ppbv'])


#%%Testing NMF on top70 dataframe
from sklearn.decomposition import NMF
from sklearn import metrics

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

#%%
n_components = 10
model = NMF(n_components=n_components)
a = model.fit(df_beijing_summer_1e6_top70.clip(lower=0))
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
    
df_Factor_totals = pd.DataFrame(Factors_totals)
df_Factor_totals.columns = [("factor"+str(num)) for num in range(n_components)]
df_Factor_totals.index = df_merge_beijing_summer.index

nmf_aq_corr = corr_2df(df_merge_beijing_summer,df_Factor_totals)

plt.scatter(df_Factor_totals['factor3'],df_merge_beijing_summer['o3_ppbv'])

fig,ax1 = plt.subplots()
ax1.plot(df_merge_beijing_summer['o3_ppbv'])
ax2 = ax1.twinx()
ax2.plot(df_merge_beijing_summer['filters_total'],c='k')
plt.show()


Can you not just do nmf all the time on everyting? Make sure that the autoencoder is all positive
Make sure that the PCA is all positive
Normalise or whatever
Run NMF
Then normalise that output, and use that to scale the total? And then you can correlate. Can you pick out anything useful though? From the factor profile?
No? Like if you want to work out the useful molecules, can you reasonably do that?