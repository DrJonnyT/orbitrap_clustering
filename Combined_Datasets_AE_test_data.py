# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:15:45 2022

@author: mbcx5jt5
"""

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
df_beijing_raw, df_beijing_filters, df_beijing_metadata = beijing_load(
    path + 'BJ_UnAmbNeg9.1.1_20210505-Times_Fixed.xlsx',path + 'BJ_UnAmbNeg9.1.1_20210505-Times_Fixed.xlsx',
    peaks_sheetname="Compounds",metadata_sheetname="massloading_Beijing")

df_delhi_raw, df_delhi_filters, df_delhi_metadata = delhi_load2(path + '/Delhi/Orbitrap/')

df_all_filters = pd.concat([df_beijing_filters, df_delhi_filters], axis=0, join="inner")
df_all_raw = pd.concat([df_beijing_raw, df_delhi_raw], axis=1, join="inner")
df_all_raw = df_all_raw.loc[:,~df_all_raw.columns.duplicated()] #Remove duplicate columns: m/z, RT, molecular weight, formula

dataset_cat = delhi_beijing_datetime_cat(df_all_filters)
df_dataset_cat = pd.DataFrame(delhi_beijing_datetime_cat(df_all_filters),columns=['dataset_cat'],index=df_all_filters.index)
ds_dataset_cat = df_dataset_cat['dataset_cat']
#pd.Series(delhi_beijing_datetime_cat(df_all_filters),['Beijing_winter','Beijing_summer' ,'Delhi_summer','Delhi_autumn'], ordered=True),index=df_all_filters.index)



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




#%%Try autoencoder
df_aug = augment_data_noise(df_all_filters_norm1,50,1,0)
ae_input = df_aug.values
ae_input_val = df_all_filters_norm1.values
input_dim = ae_input.shape[1]


#%%Test with relu for latent space activation function

latent_dims = []
AE1_MSE_best50epoch =[]
AE2_MSE_best50epoch =[]
AE3_MSE_best50epoch =[]
AE4_MSE_best50epoch =[]
best_hps_array = []


verbose = 0

start_time = time.time()

for latent_dim in range(1,10):
    print(latent_dim)
    K.clear_session()
    latent_dims.append(latent_dim)
    
    #Test for 1 intermediate layer
    ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=1,latent_activation='relu')
    history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=50,verbose=verbose)
    val_acc_per_epoch = history.history['val_loss']
    AE1_MSE_best50epoch.append(min(history.history['val_loss']))
    
    #Test for 2 intermediate layers
    ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=2,latent_activation='relu')
    history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=50,verbose=verbose)
    val_acc_per_epoch = history.history['val_loss']
    AE2_MSE_best50epoch.append(min(history.history['val_loss']))
    
    #Test for 3 intermediate layers
    ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=3,latent_activation='relu')
    history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=50,verbose=verbose)
    val_acc_per_epoch = history.history['val_loss']
    AE3_MSE_best50epoch.append(min(history.history['val_loss']))
    
    #Test for 4 intermediate layers
    ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=4,latent_activation='relu')
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
ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=21,int_layers=2,latent_activation='relu')
history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=300)
val_acc_per_epoch = history.history['val_loss']
fig,ax = plt.subplots(1,figsize=(8,6))
ax.plot(val_acc_per_epoch)
plt.show()
#Now retrain model based on best epoch
best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=21,int_layers=2,latent_activation='relu')
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