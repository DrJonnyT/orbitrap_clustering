# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:34:32 2022

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

from sklearn.preprocessing import RobustScaler, StandardScaler,FunctionTransformer,MinMaxScaler,Normalizer
from sklearn.pipeline import Pipeline


from sklearn.metrics.cluster import contingency_matrix

from scipy.stats import pearsonr

import scipy.cluster.hierarchy as sch
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score,silhouette_score, adjusted_rand_score, explained_variance_score, mean_squared_error
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
import math

import time

import os
os.chdir('C:/Work/Python/Github/Orbitrap_clustering')
from ae_functions import *

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

#%%Load chemform namelists
chemform_namelist_beijing = load_chemform_namelist(path + 'Beijing_Amb3.1_MZ.xlsx')
chemform_namelist_delhi = load_chemform_namelist(path + 'Delhi_Amb3.1_MZ.xlsx')
chemform_namelist_all = combine_chemform_namelists(chemform_namelist_beijing,chemform_namelist_delhi)


#%%Prescale datasets
#Divide whole thing by 1e6
scalefactor = 1e6
pipe_1e6 = FunctionTransformer(lambda x: np.divide(x,scalefactor),inverse_func = lambda x: np.multiply(x,scalefactor))
pipe_1e6.fit(df_all_filters)

df_all_filters_1e6 = pd.DataFrame(pipe_1e6.transform(df_all_filters),columns=df_all_filters.columns)
ds_all_filters_total_1e6 = df_all_filters_1e6.sum(axis=1)

#Normalise so the mean of the whole matrix is 1
orig_mean = df_all_filters.mean().mean()
pipe_norm1_mtx = FunctionTransformer(lambda x: np.divide(x,orig_mean),inverse_func = lambda x: np.multiply(x,orig_mean))
pipe_norm1_mtx.fit(df_all_filters)
df_all_filters_norm1 = pd.DataFrame(pipe_norm1_mtx.transform(df_all_filters),columns=df_all_filters.columns)

#Minmax scaling
minmaxscaler_all = MinMaxScaler()
df_all_filters_minmax = pd.DataFrame(minmaxscaler_all.fit_transform(df_all_filters.to_numpy()),columns=df_all_filters.columns)

#Standard scaling
standardscaler_all = StandardScaler()
df_all_filters_standard = pd.DataFrame(standardscaler_all.fit_transform(df_all_filters.to_numpy()),columns=df_all_filters.columns)

#Robust scaling
robustscaler_all = RobustScaler()
df_all_filters_robust = pd.DataFrame(robustscaler_all.fit_transform(df_all_filters.to_numpy()),columns=df_all_filters.columns)

#df scaled so it is normalised by the total from each filter
df_all_filters_norm = df_all_filters.div(df_all_filters.sum(axis=1), axis=0)

#Log data and add one
offset_min = df_all_filters.min().min() * (-1)
pipe_log1p = FunctionTransformer(lambda x: np.log1p(x+offset_min),inverse_func = lambda x: (np.expm1(x) - offset_min) )
df_all_filters_log1p = pd.DataFrame(pipe_log1p.fit_transform(df_all_filters.to_numpy()),columns=df_all_filters.columns)

#%%Export data for R tests
np.savetxt(path + "/processed/all_filters_norm1.csv", df_all_filters_norm1.to_numpy(), delimiter=",")
np.savetxt(path + "/processed/all_filters_log1p.csv", df_all_filters_log1p.to_numpy(), delimiter=",")

df_all_filters_norm1.to_csv(path+ "/processed/all_filters_log1p.csv")


#%%Try nmf on whole dataset of 4 experiments
def get_score(model, data, scorer=explained_variance_score):
    """ Estimate performance of the model on the data """
    prediction = model.inverse_transform(model.transform(data))
    return scorer(data, prediction)


#Work out how many factors
nmf_input = df_all_filters_1e6.clip(lower=0).values
ks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
perfs_train = []
nmf_recon_err = []
for k in ks:
    nmf = NMF(n_components=k).fit(nmf_input)
    perfs_train.append(get_score(nmf, nmf_input))
    nmf_recon_err.append(nmf.reconstruction_err_)
print(perfs_train)

fig,ax = plt.subplots(1)
ax.plot(ks,perfs_train,marker='x')
ax.set_ylim(0,)
ax.set_ylabel('Explained variance score (x)')
ax.set_xlabel('Num factors')
ax.xaxis.set_major_locator(plticker.MultipleLocator(base=2.0) )
ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0) )

ax2=ax.twinx()
color='k'
ax2.plot(ks,nmf_recon_err,marker='o',color=color)
ax2.set_ylabel('NMF reconstruction error (o)')
ax2.set_ylim(0,)



#%%4-factor nmf
nmf4 = NMF(n_components=4).fit(nmf_input)
W = nmf4.transform(nmf_input)
H = nmf4.components_

df_nmf4_factors = pd.DataFrame(nmf4.components_,columns=df_all_filters.columns)

#Collate the factor totals
factor_totals = np.ndarray(W.shape)
for x in np.arange(4):
    factor_totals[:,x] =  (np.outer(W.T[x], H[x]).sum(axis=1))
df_factor_totals = pd.DataFrame(factor_totals)
df_factor_totals.columns = [("factor"+str(num)+"") for num in range(4)]

plt.scatter(ds_all_filters_total_1e6,factor_totals.sum(axis=1))

#Bar chart of the avg factors for each dataset
df_nmf4_datetimecat_mean = df_factor_totals.groupby(dataset_cat).mean()

fig,ax = plt.subplots(1,figsize=(8,6))
df_nmf4_datetimecat_mean.plot.bar(ax=ax)
ax.set_ylabel('µg m$^{-3}$')
ax.set_ylim(0,)
ax.legend(bbox_to_anchor=(1.32, 0.7))
plt.show()

df_nmf4_factor_frac = df_factor_totals.div(df_factor_totals.sum(axis=1),axis=0)
df_nmf4_datetimecat_mean_frac = df_nmf4_datetimecat_mean.div(df_nmf4_datetimecat_mean.sum(axis=1),axis=0)
df_nmf4_datetimecat_mean_frac.columns = df_nmf4_factor_frac.columns

fig,ax = plt.subplots(1,figsize=(8,6))
ax.set_ylabel('Fraction')
ax.set_ylim(0,)
df_nmf4_datetimecat_mean_frac.plot.bar(stacked=True,ax=ax)
ax.legend(bbox_to_anchor=(1.32, 0.7))
plt.show()

#%%
#Calculate nmf4 loss per sample

#Calculate the loss per sample for an autoencoder
#x and y must be numpy arrays
def nmf_score_per_sample(nmf_model,data,scorer=mean_squared_error):
    score_per_sample = []
    for i in range(data.shape[0]):
        score_i = get_score(nmf_model,data=data[i:i+1],scorer=scorer)
        score_per_sample.append(score_i)
    return score_per_sample


ds_nmf4_score_per_sample = pd.Series(nmf_score_per_sample(nmf4,nmf_input), index=df_all_filters.index)

index_top_nmf4_mse= ds_nmf4_score_per_sample.nlargest(1).index

print(ds_nmf4_score_per_sample[index_top_nmf4_mse])
ds_nmf4_score_per_sample.plot(title='4-factor PMF MSE')

fig,ax = plt.subplots(1,figsize=(8,5))
plt.plot(ds_nmf4_score_per_sample.to_numpy())
plt.title('4-factor PMF MSE error')

#%%Plot PMF4 factors and high loss sample
mz_columns = pd.DataFrame(df_all_raw['Molecular Weight'].loc[df_nmf4_factors.columns])
fig,ax = plt.subplots(5,1,figsize=(10,10))
ax[0].stem(mz_columns.to_numpy(),df_nmf4_factors.loc[0],markerfmt=' ')
ax[1].stem(mz_columns.to_numpy(),df_nmf4_factors.loc[1],markerfmt=' ')
ax[2].stem(mz_columns.to_numpy(),df_nmf4_factors.loc[2],markerfmt=' ')
ax[3].stem(mz_columns.to_numpy(),df_nmf4_factors.loc[3],markerfmt=' ')
ax[4].stem(mz_columns.to_numpy(),df_all_filters_1e6.loc[index_top_nmf4_mse].transpose().values,markerfmt=' ',label='Sample with highest loss')
ax[0].set_xlim(right=500)
ax[1].set_xlim(right=500)
ax[2].set_xlim(right=500)
ax[3].set_xlim(right=500)
ax[4].set_xlim(right=500)
ax[4].legend()
plt.show()

#%%12-factor nmf
nmf12 = NMF(n_components=12).fit(nmf_input)
W = nmf12.transform(nmf_input)
H = nmf12.components_

#Collate the factor totals
factor_totals = np.ndarray(W.shape)
for x in np.arange(12):
    factor_totals[:,x] =  (np.outer(W.T[x], H[x]).sum(axis=1))
df_factor_totals = pd.DataFrame(factor_totals)
df_factor_totals.columns = [("factor"+str(num)+"") for num in range(12)]

plt.scatter(ds_all_filters_total_1e6,factor_totals.sum(axis=1))

#Bar chart of the avg factors for each dataset
df_nmf12_datetimecat_mean = df_factor_totals.groupby(dataset_cat).mean()

fig,ax = plt.subplots(1)
df_nmf12_datetimecat_mean.plot.bar(ax=ax)
ax.set_ylabel('µg m$^{-3}$')
ax.set_ylim(0,)
ax.legend(bbox_to_anchor=(1.32, 0.7))
plt.show()

df_nmf12_factor_frac = df_factor_totals.div(df_factor_totals.sum(axis=1),axis=0)
df_nmf12_datetimecat_mean_frac = df_nmf12_datetimecat_mean.div(df_nmf12_datetimecat_mean.sum(axis=1),axis=0)
df_nmf12_datetimecat_mean_frac.columns = df_nmf12_factor_frac.columns

fig,ax = plt.subplots(1)
ax.set_ylabel('Fraction')
ax.set_ylim(0,)
df_nmf12_datetimecat_mean_frac.plot.bar(stacked=True,ax=ax,cmap=plt.cm.get_cmap('tab20b', 12))
ax.legend(bbox_to_anchor=(1.32, 0.7))
plt.show()


#[['factor0','factor1','factor2','factor3','factor4']]
df_nmf12_cat_mean = df_factor_totals.groupby(dataset_cat).mean()
#df_nmf12_cat_std = df_factor_totals[['factor0','factor1','factor2','factor3','factor4']].groupby(df_merge_beijing_summer['time_cat']).std()
df_nmf12_cat_mean_norm = df_nmf12_cat_mean.div(df_nmf12_cat_mean.sum(axis=1),axis=0)

#%%Calculate nmf12 highest loss sample
ds_nmf12_score_per_sample = pd.Series(nmf_score_per_sample(nmf12,nmf_input), index=df_all_filters.index)

index_top_nmf12_mse= ds_nmf12_score_per_sample.nlargest(1).index

print(ds_nmf12_score_per_sample[index_top_nmf12_mse])
ds_nmf12_score_per_sample.plot(title='12-factor PMF MSE')

#%%nmf Line plot
fig,ax = plt.subplots(2,1,figsize=(7,10))
ax1=ax[0]
ax2=ax[1]
ax1.set_title('Orbitrap PMF, 12 factors')

#ax1.plot(df_nmf_cat_mean.index, df_nmf_cat_mean['factor0'], linewidth=5,c='b',label='factor0')
#ax1.plot(df_nmf_cat_mean.index, df_nmf_cat_mean['factor1'], linewidth=5,c='lime',label='factor1')
#ax1.plot(df_nmf_cat_mean.index, df_nmf_cat_mean['factor2'], linewidth=5,c='r',label='factor2')
#ax1.plot(df_nmf_cat_mean.index, df_nmf_cat_mean['factor3'], linewidth=5,c='orange',label='factor3')
#ax1.plot(df_nmf_cat_mean.index, df_nmf_cat_mean['factor4'], linewidth=5,c='pink',label='factor4')
df_nmf12_cat_mean.plot.bar(ax=ax1)
ax1.set_ylabel('µg m$^{-3}$')
ax1.set_ylim(0,)
ax1.legend(bbox_to_anchor=(1.22, 0.7))

# ax2.stackplot(df_nmf_cat_mean.index,df_nmf12_cat_mean_norm['factor0'], df_nmf_cat_mean_norm['factor1'],
#               df_nmf_cat_mean_norm['factor2'],df_nmf_cat_mean_norm['factor3'],
#               df_nmf_cat_mean_norm['factor4'], labels=['factor0','factor1','factor2','factor3','factor4'],
#              colors=['b','lime','r','orange','pink'])
df_nmf12_cat_mean_norm.plot.bar(ax=ax2,stacked=True,cmap=plt.cm.get_cmap('tab20b', 12))
ax2.set_ylabel('Fraction')
ax2.set_ylim(0,)
ax2.legend(bbox_to_anchor=(1.22, 0.7))




#%%Diurnal plots of real space cluster labels

#Try 5 clusters initially
agglom = AgglomerativeClustering(n_clusters = 5, linkage = 'ward')
clustering = agglom.fit(df_all_filters_1e6.values)
c = relabel_clusters_most_freq(clustering.labels_)

a = pd.DataFrame(c,columns=['clust'],index=df_dataset_cat.index)
b = df_dataset_cat

df_clust_cat_counts = a.groupby(b['dataset_cat'])['clust'].value_counts(normalize=True).unstack()
df_cat_clust_counts = b.groupby(a['clust'])['dataset_cat'].value_counts(normalize=True).unstack()


fig,ax = plt.subplots(2,1,figsize=(7,10))
ax1=ax[0]
ax2=ax[1]
df_clust_cat_counts.plot.area(ax=ax1)
df_cat_clust_counts.plot.bar(ax=ax2,stacked=True,colormap='RdBu',width=0.8)
ax1.set_title('Real space data, 5 clusters')
ax1.set_ylabel('Fraction')
ax2.set_ylabel('Fraction')
ax1.set_xlabel('')
ax2.set_xlabel('Cluster number')
ax1.legend(title='Cluster number',bbox_to_anchor=(1.25, 0.7))
ax2.legend(bbox_to_anchor=(1.25, 0.7))
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(reversed(handles), reversed(labels),title='Cluster number', bbox_to_anchor=(1.25, 0.7))
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(reversed(handles), reversed(labels), bbox_to_anchor=(1.01, 0.65))

plt.show()

#%%plot the time series
fig,ax = plt.subplots(1,figsize=(9,3))
plt.plot(c)
plt.title('Cluster labels')
plt.xlabel('~Time (arb units)')



#%%Plot the 5 clusters

cluster_totals = np.ndarray([df_all_filters_1e6.shape[1],5])
for x in np.arange(5):
    cluster_totals[:,x] =  (df_all_filters_1e6.iloc[c==x]).sum(axis=0)
df_cluster_totals = pd.DataFrame(cluster_totals)
df_cluster_totals.columns = [("cluster"+str(num)+"") for num in range(5)]
df_cluster_totals = df_cluster_totals


mz_columns = pd.DataFrame(df_all_raw['Molecular Weight'].loc[df_nmf4_factors.columns])
fig,ax = plt.subplots(5,1,figsize=(10,10))
ax[0].stem(mz_columns.to_numpy(),df_cluster_totals['cluster0'],markerfmt=' ')
ax[1].stem(mz_columns.to_numpy(),df_cluster_totals['cluster1'],markerfmt=' ')
ax[2].stem(mz_columns.to_numpy(),df_cluster_totals['cluster2'],markerfmt=' ')
ax[3].stem(mz_columns.to_numpy(),df_cluster_totals['cluster3'],markerfmt=' ')
ax[4].stem(mz_columns.to_numpy(),df_cluster_totals['cluster4'],markerfmt=' ')
ax[0].set_xlim(right=400)
ax[1].set_xlim(right=400)
ax[2].set_xlim(right=400)
ax[3].set_xlim(right=400)
ax[4].set_xlim(right=400)
ax[4].legend()
ax[0].set_title('Real-space, 5 cluster profiles')
ax[4].set_xlabel('m/z')
plt.show()




#%%Try autoencoder
df_aug = augment_data_noise(df_all_filters_norm1,50,1,0)
ae_input = df_aug.values
ae_input_val = df_all_filters_norm1.values
input_dim = ae_input.shape[1]
#%%Now compare loss for different latent dimensions
#This is NOT using kerastuner, and is using log-spaced intermediate layers
#WARNING THIS TAKES ABOUT HALF AN HOUR


latent_dims = []
AE1_MSE_best50epoch =[]
AE2_MSE_best50epoch =[]
AE3_MSE_best50epoch =[]
AE4_MSE_best50epoch =[]
best_hps_array = []


verbose = 0

start_time = time.time()

for latent_dim in range(1,25):
    print(latent_dim)
    K.clear_session()
    latent_dims.append(latent_dim)
    
    #Test for 1 intermediate layer
    ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=1)
    history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=50,verbose=verbose)
    val_acc_per_epoch = history.history['val_loss']
    AE1_MSE_best50epoch.append(min(history.history['val_loss']))
    
    #Test for 2 intermediate layers
    ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=2)
    history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=50,verbose=verbose)
    val_acc_per_epoch = history.history['val_loss']
    AE2_MSE_best50epoch.append(min(history.history['val_loss']))
    
    #Test for 3 intermediate layers
    ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=3)
    history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=50,verbose=verbose)
    val_acc_per_epoch = history.history['val_loss']
    AE3_MSE_best50epoch.append(min(history.history['val_loss']))
    
    #Test for 4 intermediate layers
    ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=4)
    history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=50,verbose=verbose)
    val_acc_per_epoch = history.history['val_loss']
    AE4_MSE_best50epoch.append(min(history.history['val_loss']))
    
print("--- %s seconds ---" % (time.time() - start_time))
    
#%%Plot the data for the different number of layers above
fig,ax=plt.subplots(2,1,figsize=(12,8))
ax[0].set_title('Finding optimum latent dims- simple relu AE')
ax[0].plot(latent_dims,AE1_MSE_best50epoch,c='black')
ax[0].plot(latent_dims,AE2_MSE_best50epoch,c='red')
ax[0].plot(latent_dims,AE3_MSE_best50epoch,c='gray')
ax[0].plot(latent_dims,AE4_MSE_best50epoch)
ax[0].set_xlabel('Number of latent dims')
ax[0].set_ylabel('Best MSE in first 50 epochs')
ax[1].plot(latent_dims,AE1_MSE_best50epoch,c='black')
ax[1].plot(latent_dims,AE2_MSE_best50epoch,c='red')
ax[1].plot(latent_dims,AE3_MSE_best50epoch,c='gray')
ax[1].plot(latent_dims,AE4_MSE_best50epoch)
ax[1].set_xlabel('Number of latent dims')
ax[1].set_ylabel('Best MSE in first 50 epochs')
ax[1].set_yscale('log')
ax[1].legend([1,2,3,4],title='Intermediate layers')


loc = plticker.MultipleLocator(base=10.0) # this locator puts ticks at regular intervals
ax[0].xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
ax[0].xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
ax[1].xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
ax[1].xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))

plt.show()

#%%Work out how many epochs to train for
#Based on the above, use an AE with 2 intermediate layers and latent dim of 20
ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=21,int_layers=2)
history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=300)
val_acc_per_epoch = history.history['val_loss']
fig,ax = plt.subplots(1,figsize=(8,6))
ax.plot(val_acc_per_epoch)
plt.show()
#Now retrain model based on best epoch
best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=21,int_layers=2)
ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=best_epoch)
print('Best epoch: %d' % (best_epoch,))


fig,ax=plt.subplots(2,1,figsize=(12,8))
ax[0].set_title('Finding optimum epochs- simple relu AE')
ax[0].plot(history.epoch,val_acc_per_epoch)
ax[0].set_xlabel('Number of epochs')
ax[0].set_ylabel('MSE')
ax[1].plot(history.epoch,val_acc_per_epoch)
ax[1].set_xlabel('Number of latent dims')
ax[1].set_ylabel('Best MSE in first 50 epochs')
ax[1].set_yscale('log')
loc = plticker.MultipleLocator(base=10.0) # this locator puts ticks at regular intervals
ax[0].xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
ax[0].xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
ax[1].xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
ax[1].xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
plt.show()

#%%Plot input vs output
latent_space = ae_obj.encoder(ae_input_val).numpy()
decoded_latent_space = ae_obj.decoder(latent_space)

fig,ax = plt.subplots(1)
plt.scatter(ae_input_val,ae_obj.ae(ae_input_val))
plt.title("AE input vs output")
plt.xlabel('AE input')
plt.ylabel('AE output')
plt.show()


#%%Evaluate loss per sample for AE
# loss_per_sample = []
# for i in range(ae_input_val.shape[0]):
#     loss_i = ae_obj.ae.evaluate(x=ae_input_val[i:i+1],
#                              y=ae_input_val[i:i+1],
#                              batch_size=None,
#                              verbose=0,
#                              steps=1
#                              )
#     loss_per_sample.append(loss_i)




#%%See that one filter has higher loss in the AE, but not sure why
ds_AE_loss_per_sample = pd.Series(AE_calc_loss_per_sample(ae_obj.ae,ae_input_val,ae_input_val), index=df_all_filters.index)
#%%
index_top_loss= ds_AE_loss_per_sample.nlargest(1).index
print(ds_AE_loss_per_sample[index_top_loss])


cluster_extract_peaks(df_all_filters.loc[index_top_loss].mean(), df_all_raw,10,chemform_namelist_all,dp=1,printdf=False)

cluster_extract_peaks(df_all_filters.mean(), df_all_raw,10,chemform_namelist_all,dp=1,printdf=False)

cluster_extract_peaks(df_all_filters_log1p.mean(), df_all_raw,10,chemform_namelist_all,dp=1,printdf=False)

#%%Make some plots of the above


#%%Top feature explorer
def top_features_hist(input_data,num_features,figsize='DEFAULT',num_bins=25,logx=False):
    if str(figsize) == 'DEFAULT':
        figsize=(12,10)
    
    if(logx==True):
        df_input = pd.DataFrame(input_data).clip(lower=0.01)
    else:
        df_input = pd.DataFrame(input_data)
    
    #Catch if more features requested than there are features in the data
    if(num_features > input_data.shape[1]):
         num_features = input_data.shape[1]
    
    peaks_sum = df_input.sum()
    index_top_features = peaks_sum.nlargest(num_features).index
    
    
    cols = round(math.sqrt(num_features))
    rows = cols
    while rows * cols < num_features:
        rows += 1
    fig, ax_arr = plt.subplots(rows, cols,figsize=figsize)
    
    if(logx==True):
        fig.suptitle('Logscale histograms of top ' + str(num_features) + ' features',size=14)
    else:
        fig.suptitle('Histograms of top ' + str(num_features) + ' features',size=14)
    ax_arr = ax_arr.reshape(-1)
    
    for i in range(len(ax_arr)):
        if i >= num_features:
            ax_arr[i].axis('off')
        else:
            data = df_input.iloc[:,index_top_features[i]]
            if(logx==True):
                logbins = np.logspace(np.log10(data.min()),np.log10(data.max()),num_bins)
                ax_arr[i].hist(data,bins=logbins)
                ax_arr[i].set_xscale('log')
            else:
                ax_arr[i].hist(data,bins=num_bins)
            
            
    plt.tight_layout()
    plt.show()
    
#%%Make some plots of histograms of top features
top_features_hist(ae_input_val,25,logx=True)
top_features_hist(ae_input_val,25)

top_features_hist(latent_space,25,logx=True)
top_features_hist(latent_space,25)

top_features_hist(df_all_filters_minmax.values,25,logx=True)
top_features_hist(df_all_filters_minmax.values,25)

top_features_hist(df_all_filters_standard.values,25,logx=True)
top_features_hist(df_all_filters_standard.values,25)

top_features_hist(df_all_filters_robust.values,25,logx=True)
top_features_hist(df_all_filters_robust.values,25)

top_features_hist(df_all_filters_norm.values,25,logx=True)
top_features_hist(df_all_filters_norm.values,25)

top_features_hist(df_all_filters_log1p.values,25,logx=True)
top_features_hist(df_all_filters_log1p.values,25)

#%%Do some clustering of the various spaces
#%%Now do some clustering on the real-space data
############################################################################################
#CLUSTERING AND FACTOR ANALYSIS OF real-space data DATA
############################################################################################
#Do it with the linearly scaled data?
#%%Real space dendrogram
#Lets try a dendrogram to work out the optimal number of clusters
fig,axes = plt.subplots(1,1,figsize=(20,10))
plt.title('Real space dendrogram')
dendrogram = sch.dendrogram(sch.linkage(ae_input_val, method='ward'))
plt.show()


# %%#How many clusters should we have? Log-space
min_clusters = 2
max_clusters = 30

num_clusters_index = range(min_clusters,(max_clusters+1),1)
ch_score = np.empty(len(num_clusters_index))
db_score = np.empty(len(num_clusters_index))
silhouette_scores = np.empty(len(num_clusters_index))

for num_clusters in num_clusters_index:
    agglom_native = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    #agglom_native = KMeans(n_clusters = num_clusters)
    clustering = agglom_native.fit(ae_input_val)
    ch_score[num_clusters-min_clusters] = calinski_harabasz_score(ae_input_val, clustering.labels_)
    db_score[num_clusters-min_clusters] = davies_bouldin_score(ae_input_val, clustering.labels_)
    silhouette_scores[num_clusters-min_clusters] = silhouette_score(ae_input_val, clustering.labels_)

fig,ax1 = plt.subplots(figsize=(10,6))
ax2=ax1.twinx()
ax1.plot(num_clusters_index,ch_score,label="CH score")
ax2.plot(num_clusters_index,db_score,c='red',label="DB score")
ax2.plot(num_clusters_index,silhouette_scores,c='black',label="Silhouette score")
ax1.set_xlabel("Num clusters")
ax1.set_ylabel("CH score")
ax2.set_ylabel("DB score")
plt.title("How many clusters? Log-space data")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)
ax1.xaxis.set_major_locator(plticker.MultipleLocator(base=5.0))
ax1.xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
ax1.xaxis.grid(which='both',c='lightgrey',linestyle='--')
plt.show()




#%%tSNE plots for latent space clusters
colormap = ['k','blue','red','yellow','gray','purple','aqua','gold','orange']

for num_clusters in range(2,10):
    agglom = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    clustering = agglom.fit(ae_input_val)
    # plt.plot(clustering.labels_)
    # plt.scatter(df_beijing_filters.index,clustering.labels_,marker='.')
    
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(ae_input_val)
    plt.figure(figsize=(9,5))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                c=clustering.labels_,
                cmap=ListedColormap(colormap[0:num_clusters]))

    bounds = np.arange(0,num_clusters+1) -0.5
    ticks = np.arange(0,num_clusters)
    plt.colorbar(boundaries=bounds,ticks=ticks,label='Cluster')
    plt.title('tSNE real space, ' + str(num_clusters) + ' clusters')
    plt.show()






#%%
############################################################################################
#CLUSTERING AND FACTOR ANALYSIS OF log-space DATA
############################################################################################
#%%Log-space space dendrogram
#Lets try a dendrogram to work out the optimal number of clusters
fig,axes = plt.subplots(1,1,figsize=(20,10))
plt.title('Real space dendrogram')
dendrogram = sch.dendrogram(sch.linkage(df_all_filters_log1p.to_numpy(), method='ward'))
plt.show()


# %%#How many clusters should we have? Log-space
min_clusters = 2
max_clusters = 30

num_clusters_index = range(min_clusters,(max_clusters+1),1)
ch_score = np.empty(len(num_clusters_index))
db_score = np.empty(len(num_clusters_index))
silhouette_scores = np.empty(len(num_clusters_index))

for num_clusters in num_clusters_index:
    agglom_native = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    #agglom_native = KMeans(n_clusters = num_clusters)
    clustering = agglom_native.fit(df_all_filters_log1p.to_numpy())
    ch_score[num_clusters-min_clusters] = calinski_harabasz_score(df_all_filters_log1p.to_numpy(), clustering.labels_)
    db_score[num_clusters-min_clusters] = davies_bouldin_score(df_all_filters_log1p.to_numpy(), clustering.labels_)
    silhouette_scores[num_clusters-min_clusters] = silhouette_score(df_all_filters_log1p.to_numpy(), clustering.labels_)
fig,ax1 = plt.subplots(figsize=(10,6))
ax2=ax1.twinx()
ax1.plot(num_clusters_index,ch_score,label="CH score")
ax2.plot(num_clusters_index,db_score,c='red',label="DB score")
ax2.plot(num_clusters_index,silhouette_scores,c='black',label="Silhouette score")
ax1.set_xlabel("Num clusters")
ax1.set_ylabel("CH score")
ax2.set_ylabel("DB score")
plt.title("How many clusters? real-space data")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)
ax1.xaxis.set_major_locator(plticker.MultipleLocator(base=5.0))
ax1.xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
plt.show()




#%%tSNE plots for latent space clusters
colormap = ['k','blue','red','yellow','gray','purple','aqua','gold','orange']

for num_clusters in range(2,10):
    agglom = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    clustering = agglom.fit(df_all_filters_log1p.to_numpy())
    # plt.plot(clustering.labels_)
    # plt.scatter(df_beijing_filters.index,clustering.labels_,marker='.')
    
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(df_all_filters_log1p.to_numpy())
    plt.figure(figsize=(9,5))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                c=clustering.labels_,
                cmap=ListedColormap(colormap[0:num_clusters]))

    bounds = np.arange(0,num_clusters+1) -0.5
    ticks = np.arange(0,num_clusters)
    plt.colorbar(boundaries=bounds,ticks=ticks,label='Cluster')
    plt.title('tSNE real space, ' + str(num_clusters) + ' clusters')
    plt.show()



# %%Principal component analysis 
#How many components to use?

min_components = 2
max_components = 30

num_components_index = range(min_components,(max_components+1),1)
pca_variance_explained = np.empty(len(num_components_index))
pca_scaled_variance_explained = np.empty(len(num_components_index))
pca_ae_variance_explained = np.empty(len(num_components_index))

for num_components in num_components_index:
    pca = PCA(n_components = num_components)
    #pca_ae = PCA(n_components = num_components)
    prin_comp = pca.fit_transform(np.nan_to_num(ae_input_val))
    pca_variance_explained[num_components-min_components] = pca.explained_variance_ratio_.sum()
    #prin_comp = pca.fit_transform(ae_input_val)
    #pca_scaled_variance_explained[num_components-min_components] = pca.explained_variance_ratio_.sum()
    #prin_comp_ae = pca_ae.fit_transform(latent_space)
    #pca_ae_variance_explained[num_components-min_components] = pca_ae.explained_variance_ratio_.sum()
    
fig,ax = plt.subplots(figsize=(12,8))
ax.plot(num_components_index,pca_variance_explained,label="PCA on linearly scaled data",marker='x')
#ax.plot(num_components_index,pca_scaled_variance_explained,label="PCA on scaled AE input")
#ax.plot(num_components_index,pca_ae_variance_explained,c='red',label="PCA on latent space")
ax.xaxis.set_major_locator(plticker.MultipleLocator(base=5.0))
ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
ax.set_xlabel("Num PCA components")
ax.set_ylabel("Fraction of variance explained")
plt.legend(loc="lower right")
plt.title("How many PCA components?")
plt.grid(which='major')
plt.grid(which='minor',c='lightgrey')
plt.show()

# %%Principal component analysis  on log1p dataset
#How many components to use?

min_components = 2
max_components = 30

num_components_index = range(min_components,(max_components+1),1)
pca_variance_explained = np.empty(len(num_components_index))
pca_scaled_variance_explained = np.empty(len(num_components_index))
pca_ae_variance_explained = np.empty(len(num_components_index))

for num_components in num_components_index:
    pca = PCA(n_components = num_components)
    #pca_ae = PCA(n_components = num_components)
    prin_comp = pca.fit_transform(np.nan_to_num(df_all_filters_log1p.to_numpy()))
    pca_variance_explained[num_components-min_components] = pca.explained_variance_ratio_.sum()
    #prin_comp = pca.fit_transform(ae_input_val)
    #pca_scaled_variance_explained[num_components-min_components] = pca.explained_variance_ratio_.sum()
    #prin_comp_ae = pca_ae.fit_transform(latent_space)
    #pca_ae_variance_explained[num_components-min_components] = pca_ae.explained_variance_ratio_.sum()
    
fig,ax = plt.subplots(figsize=(12,8))
ax.plot(num_components_index,pca_variance_explained,label="PCA on log scaled data",marker='x')
#ax.plot(num_components_index,pca_scaled_variance_explained,label="PCA on scaled AE input")
#ax.plot(num_components_index,pca_ae_variance_explained,c='red',label="PCA on latent space")
ax.xaxis.set_major_locator(plticker.MultipleLocator(base=5.0))
ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
ax.set_xlabel("Num PCA components")
ax.set_ylabel("Fraction of variance explained")
plt.legend(loc="lower right")
plt.title("How many PCA components?")
plt.grid(which='major')
plt.grid(which='minor',c='lightgrey')
plt.show()

#%%PCA7 transform the native dataset
#Go with 7 because it covers 95% of the variance of the original dataset
pca7 = PCA(n_components = 7)
all_filters_PCA7_space = pca7.fit_transform(pipe_1e6.transform(ae_input_val))

# %%Comparing clustering labels
################################################
####COMPARING DIFFERENT CLUSTER LABELS######
################################################
clusters = []
#arand_real = []
arand_real_minmax = []
arand_real_pca7 = []
arand_real_ae = []
arand_real_log1p = []
arand_real_norm = []


for num_clusters in range(2,10):
    agglom = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    clustering_real = agglom.fit(df_all_filters_1e6.to_numpy())
    # agglom = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    # clustering_top70 = agglom.fit(scaled_top70_np)
    agglom = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    clustering_minmax = agglom.fit(df_all_filters_minmax.to_numpy())
    agglom = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    clustering_norm = agglom.fit(df_all_filters_norm.to_numpy())
    # agglom = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    # clustering_norm1 = agglom.fit(df_all_filters_norm1.to_numpy())
    agglom = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    clustering_pca7 = agglom.fit(all_filters_PCA7_space)
    agglom = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    clustering_ae = agglom.fit(latent_space)
    agglom = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    clustering_log1p = agglom.fit(df_all_filters_log1p.to_numpy())
    
    clusters.append(num_clusters)
    #arand_real_top70.append(adjusted_rand_score(clustering_real.labels_, clustering_top70.labels_))
    arand_real_minmax.append(adjusted_rand_score(clustering_real.labels_, clustering_minmax.labels_))
    arand_real_pca7.append(adjusted_rand_score(clustering_real.labels_, clustering_pca7.labels_))
    arand_real_log1p.append(adjusted_rand_score(clustering_real.labels_, clustering_log1p.labels_))
    arand_real_ae.append(adjusted_rand_score(clustering_real.labels_, clustering_ae.labels_))
    arand_real_norm.append(adjusted_rand_score(clustering_real.labels_, clustering_norm.labels_))
    #arand_real_norm1.append(adjusted_rand_score(clustering_real.labels_, clustering_norm1.labels_))
    
#%%
fig,ax = plt.subplots(1,figsize=(7,5))
#ax.plot(clusters,arand_real_top70,label='Real-space vs top 70%',c='b')
ax.plot(clusters,arand_real_minmax,label='Real-space vs MinMax data',c='k')
ax.plot(clusters,arand_real_pca7,label='Real-space vs PCA-space',c='r')
ax.plot(clusters,arand_real_ae,label='Real-space vs AE latent space',c='gray')
ax.plot(clusters,arand_real_log1p,label='Real-space vs logged real space',c='y')
ax.plot(clusters,arand_real_norm,label='Real-space vs normalised so all samples sum to 1',c='b')
#ax.plot(clusters,arand_real_norm1,label='Real-space vs normalised so whole matrix mean is 1',c='r')
ax.set_title('Adjusted Rand score (how similar are the cluster labels)')

# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

ax.legend(loc='upper center',bbox_to_anchor=(0.5, -0.15))
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Similarity')
#plt.tight_layout()
plt.show()


#%%Compare cluster profiles
#First gather cluster profiles
#Start with a few clusters BUT I think it's very ambiguous how many to use going forward. Nbclust and n_clust in R were no help
#Maybe try 8 eventually??? I guess you don't have to use the same number for each dataset

#This will have a thing where you get the uncentred R for each cluster compared to each other cluster

#labels1 & labels 2 should be numpy arrays
#input data should be numpy
def correlate_clusters(labels1,labels2,input_data,norm=True,sub_from_mean=False):
    if(sub_from_mean==True):
        norm=True
        datamean = input_data.mean(axis=0)
        datamean = datamean / (datamean.sum(axis=0,keepdims=1))
        
    labels1 = relabel_clusters_most_freq(labels1)
    labels2 = relabel_clusters_most_freq(labels2)
    
    num_clusters_1 = len(np.unique(labels1))
    num_clusters_2 = len(np.unique(labels2))
    
    correlation_matrix = np.ndarray([num_clusters_1,num_clusters_2])
    for i in np.arange(num_clusters_1):
        for j in np.arange(num_clusters_2):
            cluster1_i = input_data[labels1==i].mean(axis=0)
            cluster2_j = input_data[labels2==j].mean(axis=0)
            
            if(norm==True):
                cluster1_i = cluster1_i/(cluster1_i.sum(axis=0,keepdims=1))
                cluster2_j = cluster2_j/(cluster2_j.sum(axis=0,keepdims=1))
            if(sub_from_mean==True):
                cluster1_i = cluster1_i - datamean
                cluster2_j = cluster2_j - datamean
                #pdb.set_trace()
                #Standard centred Pearson's R, because there could be negative data, it does not sum to zero. Or does it?
                correlation_matrix[i][j] = pearsonr(cluster1_i,cluster2_j)[0]
                
            else:
                #Uncentred R aka normalised dot product
                correlation_matrix[i][j] = np.dot(cluster1_i,cluster2_j) / np.sqrt( np.dot(cluster1_i,cluster1_i) * np.dot(cluster2_j,cluster2_j)   )
    
    return correlation_matrix


def summarise_cluster_comparison(input_data1,input_data2,nclust1,nclust2_min,nclust2_max,norm=True,sub_from_mean=False):
    #Clustering for input_data1
    agglom1 = AgglomerativeClustering(n_clusters = nclust1, linkage = 'ward')
    clustering1 = agglom1.fit(input_data1)
    #pdb.set_trace()
    max_R_mtx = np.empty([(nclust2_max),(nclust2_max-nclust2_min+1)])
    max_R_mtx.fill(np.nan)
    df_max_R_mtx = pd.DataFrame(max_R_mtx, index=np.arange(nclust2_max).tolist(), columns=np.arange(nclust2_min,nclust2_max+1).tolist())
    df_max_R_mtx.rename_axis(index='cluster_index', columns="num_clusters",inplace=True)
    df_clust_freq_mtx = df_max_R_mtx.copy()
    
    
    for nclust2 in range(nclust2_min,nclust2_max+1):
        agglom2 = AgglomerativeClustering(n_clusters = nclust2, linkage = 'ward')
        clustering2 = agglom2.fit(input_data2)    
        correlation_matrix = correlate_clusters(clustering1.labels_,clustering2.labels_,input_data1,norm=norm,sub_from_mean=sub_from_mean)
        #pdb.set_trace()
        df_max_R_mtx[nclust2].iloc[0:nclust2] = correlation_matrix.max(axis=0)
        
        #Number of data points for each cluster
        df_clust_freq_mtx[nclust2].iloc[0:nclust2] = np.unique(relabel_clusters_most_freq(clustering2.labels_), return_counts=True)[1]
    
    
    return df_max_R_mtx, df_clust_freq_mtx
    
    

#%%Try plot this badboy
a,b = summarise_cluster_comparison(ae_input_val,df_all_filters_norm.to_numpy(),5,2,8,sub_from_mean=True)

table_vals = np.char.replace(np.around(a.values,2).astype(str),'nan','')
vals = np.nan_to_num(np.around(b.values,2))
norm = plt.Normalize(vals.min()-1, vals.max()+1)
colours = plt.cm.spring(norm(vals))

fontsize=14

fig = plt.figure()
ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])


the_table=ax.table(cellText=table_vals, rowLabels=a.index, colLabels=a.columns, 
                     colWidths = [0.12]*vals.shape[1],loc='center', 
                    cellColours=colours)
the_table.auto_set_font_size(False)
the_table.set_fontsize(fontsize)
ax.set_xlabel('Number of clusters',fontsize=fontsize,labelpad=-20)
ax.xaxis.set_label_position('top') 
ax.set_ylabel('Cluster index',fontsize=fontsize,labelpad=25)
the_table.scale(1.5,1.5)

img = plt.imshow(colours, cmap="hot")
plt.colorbar()
img.set_visible(False)

plt.tight_layout()
plt.show()

#%%

fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111, frameon=True, xticks = [], yticks = [])
cells = np.random.randint(0, 100, (10, 10))
img = plt.imshow(cells, cmap="hot")
plt.colorbar()
img.set_visible(False)
tb = plt.table(cellText = cells, 
    rowLabels = range(10), 
    colLabels = range(10), 
    loc = 'center',
    cellColours = img.to_rgba(cells))
ax.add_table(tb)
plt.show()

