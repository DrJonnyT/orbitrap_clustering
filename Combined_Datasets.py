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

import scipy.cluster.hierarchy as sch
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score,silhouette_score, adjusted_rand_score, explained_variance_score
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
df_beijing_raw, df_beijing_filters, df_beijing_metadata = beijing_load(
    path + 'BJ_UnAmbNeg9.1.1_20210505-Times_Fixed.xlsx',path + 'BJ_UnAmbNeg9.1.1_20210505-Times_Fixed.xlsx',
    peaks_sheetname="Compounds",metadata_sheetname="massloading_Beijing")

df_delhi_raw, df_delhi_filters, df_delhi_metadata = delhi_load2(path + '/Delhi/Orbitrap/')

df_all_filters = pd.concat([df_beijing_filters, df_delhi_filters], axis=0, join="inner")
dataset_cat = delhi_beijing_datetime_cat(df_all_filters)
df_dataset_cat = pd.DataFrame(pd.Categorical(delhi_beijing_datetime_cat(df_all_filters),['Beijing_winter','Beijing_summer' ,'Delhi_summer','Delhi_autumn'], ordered=True),columns=['dataset_cat'],index=df_all_filters.index)
ds_dataset_cat = pd.Series(pd.Categorical(delhi_beijing_datetime_cat(df_all_filters),['Beijing_winter','Beijing_summer' ,'Delhi_summer','Delhi_autumn'], ordered=True),index=df_all_filters.index)
#df_beijing_winter = df_beijing_filters.iloc[0:124].copy()
#df_beijing_summer = df_beijing_filters.iloc[124:].copy()

#df_all_filters = df_beijing_filters.append(df_delhi_filters,sort=True).fillna(0)
df_all_raw = df_beijing_raw.transpose().append(df_delhi_raw.transpose(),sort=True).transpose()

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
pipe_norm_mtx = FunctionTransformer(lambda x: np.divide(x,orig_mean),inverse_func = lambda x: np.multiply(x,orig_mean))
pipe_norm_mtx.fit(df_all_filters)
df_all_filters_norm1 = pd.DataFrame(pipe_norm_mtx.transform(df_all_filters),columns=df_all_filters.columns)

#Minmax scaling
minmaxscaler_all = MinMaxScaler()
df_all_filters_minmax = minmaxscaler_all.fit_transform(df_all_filters.to_numpy())

#Standard scaling
standardscaler_all = StandardScaler()
df_all_filters_standard = standardscaler_all.fit_transform(df_all_filters.to_numpy())

#Standard scaling
robustscaler_all = RobustScaler()
df_all_filters_robust = robustscaler_all.fit_transform(df_all_filters.to_numpy())

#df scaled so it is normalised by the total from each filter
df_all_filters_norm = df_all_filters.div(df_all_filters.sum(axis=1), axis=0)


#%%Try nmf on whole dataset of 4 experiments
def get_score(model, data, scorer=explained_variance_score):
    """ Estimate performance of the model on the data """
    prediction = model.inverse_transform(model.transform(data))
    return scorer(data, prediction)


#Work out how many factors
nmf_input = df_all_filters_1e6.clip(lower=0).values
ks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
perfs_train = []
for k in ks:
    nmf = NMF(n_components=k).fit(nmf_input)
    perfs_train.append(get_score(nmf, nmf_input))
print(perfs_train)

fig,ax = plt.subplots(1)
ax.plot(ks,perfs_train,marker='x')
ax.set_ylim(0,)
ax.set_ylabel('Explained variance score')
ax.set_xlabel('Num PMF factors')
ax.xaxis.set_major_locator(plticker.MultipleLocator(base=2.0) )
ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0) )


#%%4-factor nmf
nmf4 = NMF(n_components=4).fit(nmf_input)
W = nmf4.transform(nmf_input)
H = nmf4.components_

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
df_nmf12_datetimecat_mean_frac.plot.bar(stacked=True,ax=ax)
ax.legend(bbox_to_anchor=(1.32, 0.7))
plt.show()


#[['factor0','factor1','factor2','factor3','factor4']]
df_nmf12_cat_mean = df_factor_totals.groupby(dataset_cat).mean()
#df_nmf12_cat_std = df_factor_totals[['factor0','factor1','factor2','factor3','factor4']].groupby(df_merge_beijing_summer['time_cat']).std()
df_nmf12_cat_mean_norm = df_nmf12_cat_mean.div(df_nmf12_cat_mean.sum(axis=1),axis=0)

#%%nmf Line plot
fig,ax = plt.subplots(2,1,figsize=(7,10))
ax1=ax[0]
ax2=ax[1]
ax1.set_title('Orbitrap PMF, 5 factors')

#ax1.plot(df_nmf_cat_mean.index, df_nmf_cat_mean['factor0'], linewidth=5,c='b',label='factor0')
#ax1.plot(df_nmf_cat_mean.index, df_nmf_cat_mean['factor1'], linewidth=5,c='lime',label='factor1')
#ax1.plot(df_nmf_cat_mean.index, df_nmf_cat_mean['factor2'], linewidth=5,c='r',label='factor2')
#ax1.plot(df_nmf_cat_mean.index, df_nmf_cat_mean['factor3'], linewidth=5,c='orange',label='factor3')
#ax1.plot(df_nmf_cat_mean.index, df_nmf_cat_mean['factor4'], linewidth=5,c='pink',label='factor4')
df_nmf12_cat_mean.plot(ax=ax1)
ax1.set_ylabel('µg m$^{-3}$')
ax1.set_ylim(0,)
ax1.legend(bbox_to_anchor=(1.22, 0.7))

# ax2.stackplot(df_nmf_cat_mean.index,df_nmf12_cat_mean_norm['factor0'], df_nmf_cat_mean_norm['factor1'],
#               df_nmf_cat_mean_norm['factor2'],df_nmf_cat_mean_norm['factor3'],
#               df_nmf_cat_mean_norm['factor4'], labels=['factor0','factor1','factor2','factor3','factor4'],
#              colors=['b','lime','r','orange','pink'])
df_nmf12_cat_mean_norm.plot.bar(ax=ax2,stacked=True)
ax2.set_ylabel('Fraction')
ax2.set_ylim(0,)
ax2.legend(bbox_to_anchor=(1.22, 0.7))




#%%Diurnal plots of real space cluster labels

#Try 5 clusters initially
agglom = AgglomerativeClustering(n_clusters = 10, linkage = 'ward')
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
ax2.legend(reversed(handles), reversed(labels), bbox_to_anchor=(1.25, 0.65))

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




#%%
ds_AE_loss_per_sample = pd.Series(AE_calc_loss_per_sample(ae_obj.ae,ae_input_val,ae_input_val), index=df_all_filters.index)
#%%
index_top_loss= ds_AE_loss_per_sample.nlargest(2).index
print(ds_AE_loss_per_sample[index_top_loss])




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
top_features_hist(ae_input_val,25,logscale=True)
top_features_hist(ae_input_val,25)

top_features_hist(latent_space,25,logscale=True)
top_features_hist(latent_space,25)
