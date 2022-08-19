# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 17:00:03 2022

@author: mbcx5jt5
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
#import seaborn as sns
#import math
import time

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import AgglomerativeClustering

import os
os.chdir('C:/Work/Python/Github/Orbitrap_clustering')
from ae_functions import *
from orbitrap_functions import *





#%%Load data from HDF
filepath = r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\PMF data\ORBITRAP_Data_Pre_PMF.h5"
df_all_data, df_all_err, ds_all_mz = Load_pre_PMF_data(filepath)


#Load all time data, ie start/mid/end
df_all_times = pd.read_csv(r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\PMF data\Times_all.csv")
df_all_times['date_start'] = pd.to_datetime(df_all_times['date_start'],dayfirst=True)
df_all_times['date_mid'] = pd.to_datetime(df_all_times['date_mid'],dayfirst=True)
df_all_times['date_end'] = pd.to_datetime(df_all_times['date_end'],dayfirst=True)
df_all_times.set_index(df_all_times['date_mid'],inplace=True)
df_all_times = df_all_times.loc[df_all_data.index]

dataset_cat = delhi_beijing_datetime_cat(df_all_data)
df_dataset_cat = pd.DataFrame(delhi_beijing_datetime_cat(df_all_data),columns=['dataset_cat'],index=df_all_data.index)
ds_dataset_cat = df_dataset_cat['dataset_cat']

time_cat = delhi_calc_time_cat(df_all_times)
df_time_cat = pd.DataFrame(delhi_calc_time_cat(df_all_times),columns=['time_cat'],index=df_all_times.index)
ds_time_cat = df_time_cat['time_cat']


#%%Test data scaling

#Standard scaling
standardscaler_all = StandardScaler()
df_all_data_standard = pd.DataFrame(standardscaler_all.fit_transform(df_all_data.to_numpy()),columns=df_all_data.columns)

standardscaler_all2 = StandardScaler(with_std=False)
df_all_data_standard2 = pd.DataFrame(standardscaler_all2.fit_transform(df_all_data.to_numpy()),columns=df_all_data.columns)

max_col = df_all_data.mean().idxmax()
min_col = df_all_data.mean().idxmin()

df_all_data[max_col].plot(kind='hist',title='No scaling, max column')
df_all_data[min_col].plot(kind='hist',title='No scaling, min column')
df_all_data_standard[max_col].plot(kind='hist',title='Standard scaling, max column')
df_all_data_standard[min_col].plot(kind='hist',title='Standard scaling, min column')
df_all_data_standard2[max_col].plot(kind='hist',title='Standard scaling (not normalised), max column')
df_all_data_standard2[min_col].plot(kind='hist',title='Standard scaling (not normalised), min column')

#%%Normalise so the mean of the whole matrix is 1
orig_mean = df_all_data.mean().mean()
pipe_norm1_mtx = FunctionTransformer(lambda x: np.divide(x,orig_mean),inverse_func = lambda x: np.multiply(x,orig_mean))
pipe_norm1_mtx.fit(df_all_data)
df_all_data_norm1 = pd.DataFrame(pipe_norm1_mtx.transform(df_all_data),columns=df_all_data.columns)


#%%Setup autoencoder
df_aug = augment_data_noise(df_all_data_norm1,50,1,0)
ae_input = df_aug.values
ae_input_val = df_all_data_norm1.values
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

for latent_dim in range(1,30):
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
ax[0].xaxis.set_major_locator(plticker.MultipleLocator(base=5.0))
ax[0].xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
ax[1].xaxis.set_major_locator(plticker.MultipleLocator(base=5.0))
ax[1].xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))

plt.show()


#%%Work out how many epochs to train for
#Based on the above, use an AE with 2 intermediate layers and latent dim of 20
ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=2,int_layers=3)
history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=100)
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
df_latent_space = pd.DataFrame(latent_space,index=df_all_data.index)

fig,ax = plt.subplots(1)
plt.scatter(ae_input_val,ae_obj.ae(ae_input_val))
plt.title("AE input vs output")
plt.xlabel('AE input')
plt.ylabel('AE output')
plt.show()


#%%Plot latent space
plt.scatter(latent_space[:,0],latent_space[:,1],c=)



#%%See what biggest and smallest stdev features of latent space look like
df_latent_space[df_latent_space.std().idxmax()].plot(kind='hist',title='Latent space, max stdev column')
df_latent_space[df_latent_space.std().idxmin()].plot(kind='hist',title='Latent space, min stdev column')




#%%AE top loss
ds_AE_loss_per_sample = pd.Series(AE_calc_loss_per_sample(ae_obj.ae,ae_input_val,ae_input_val), index=df_all_data.index)
plot_top_ae_loss(df_all_data,ds_AE_loss_per_sample)


#%%Clustering







    