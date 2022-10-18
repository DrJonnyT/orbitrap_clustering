# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 16:05:27 2022

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
df_all_data, df_all_err, ds_all_mz = Load_pre_PMF_data(filepath,join='outer')

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


#%%Transform data to regular mz grid
mz_spacing = 1
mz_grid = np.arange(round(ds_all_mz.min()),round(ds_all_mz.max())+mz_spacing,mz_spacing)
df_all_data_grid = df_all_data.groupby(np.round(ds_all_mz), axis=1).sum().reindex(mz_grid,axis=1,fill_value=0)












# #%%#VAE exploration
# input_dim = df_all_data_grid.shape[1]

# encoder_input_layer = tf.keras.layers.Input(shape=(1,input_dim), name="encoder_input_layer")
# conv1d_layer = tf.keras.layers.Conv1D(filters=10, kernel_size=15,data_format = 'channels_first')(encoder_input_layer)


# encoder_input_layer = tf.keras.Input(shape=(1,input_dim), name="encoder_input_layer")
# encoder_input_layer2 = tf.keras.layers.Conv1D(filters=1, kernel_size=15,data_format = 'channels_first',name='input_conv1D')(encoder_input_layer)


# model=tf.keras.Model(inputs=encoder_input_layer,outputs=encoder_input_layer2)

# w[0]=np.asarray(np.concatenate([[1],np.zeros(13),[1]]))





# w = model.layers[1].get_weights()
# #w[0] = np.asarray([[[1]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[1]]])
# w[0]=np.asarray([[np.concatenate([[1],np.zeros(13),[1]])]]).T
# model.layers[1].set_weights(w)
# print("Weights after change:")
# print(model.layers[1].get_weights())


#%%
#Normalise so the mean of the whole matrix is 1
orig_mean = df_all_data_grid.mean().mean()
pipe_norm1_mtx = FunctionTransformer(lambda x: np.divide(x,orig_mean),inverse_func = lambda x: np.multiply(x,orig_mean))
df_all_data_grid_norm1 = pd.DataFrame(pipe_norm1_mtx.fit_transform(df_all_data_grid.to_numpy()),index=df_all_data_grid.index,columns=df_all_data_grid.columns)




#%%Generate synthetic dataset
# %%
# #Let's make some factors that vary with time
#First find the least correlated filters from the various datasets

#Correlate each row with every other row
#My version that is slow but actually works for matrices of different number of rows
def corr_coeff_rowwise_loops(A,B):
    corr_mtx = np.empty([A.shape[0],B.shape[0]])
    for i in range(A.shape[0]):
       #pdb.set_trace()
        Arow = A[i,:]
        for j in range(B.shape[0]):
            Brow = B[j,:]
            #pdb.set_trace()
            corr_mtx[i,j] = pearsonr(Arow,Brow)[0]
    return corr_mtx
            

Beijing_rows_corr = corr_coeff_rowwise_loops(df_all_data_grid_norm1[ds_dataset_cat=='Beijing_winter'].values,df_all_data_grid_norm1[ds_dataset_cat=='Beijing_summer'].values)
Beijing_rows_corr_min_index = np.unravel_index(Beijing_rows_corr.argmin(), Beijing_rows_corr.shape)
Delhi_rows_corr = corr_coeff_rowwise_loops(df_all_data_grid_norm1[ds_dataset_cat=='Delhi_summer'].values,df_all_data_grid_norm1[ds_dataset_cat=='Delhi_autumn'].values)
Delhi_rows_corr_min_index = np.unravel_index(Delhi_rows_corr.argmin(), Delhi_rows_corr.shape)


#This then is 4 factors that are very poorly correlated with each other in terms of their mass spec
factor_A = df_all_data_grid_norm1[ds_dataset_cat=='Beijing_winter'].iloc[Beijing_rows_corr_min_index[0]].values
factor_B = df_all_data_grid_norm1[ds_dataset_cat=='Beijing_summer'].iloc[Beijing_rows_corr_min_index[1]].values
factor_C = df_all_data_grid_norm1[ds_dataset_cat=='Delhi_summer'].iloc[Delhi_rows_corr_min_index[0]].values
factor_D = df_all_data_grid_norm1[ds_dataset_cat=='Delhi_autumn'].iloc[Delhi_rows_corr_min_index[1]].values

#Normalise all to 1
factor_A = 1 * factor_A / factor_A.sum()
factor_B = 1 * factor_B / factor_B.sum()
factor_C = 1 * factor_C / factor_C.sum()
factor_D = 1 * factor_D / factor_D.sum()



# #Try completely separate factors
# factor_A = np.tile([1,0,0,0],160)/160
# factor_B = np.tile([0,1,0,0],160)/160
# factor_C = np.tile([0,0,1,0],160)/160
# factor_D = np.tile([0,0,0,1],160)/160






# #Make factor E so all columns between mz 350 -- 400 are...dataful...
# factor_E  = np.ones(df_all_data.shape[1])
# factor_E[np.ravel(np.logical_and(ds_all_mz.to_numpy() > 350, ds_all_mz.to_numpy() < 400))] = 50
# factor_E = factor_E * np.random.normal(1, 0.3, [df_all_data.shape[1]]).clip(min=0)
# factor_E = 1 * factor_E / factor_E.sum()

#%% VERY EASY AMPLITUDES
#Amplitudes to make it very easy for the model to pick out each factor
amp_A = np.concatenate((np.sin(np.arange(0,math.pi,math.pi/50))*1.5,np.zeros(150))) * 2.5
amp_B = np.concatenate((np.zeros(50),np.sin(np.arange(0,math.pi,math.pi/50))*1.5,np.zeros(100))) * 2.5
amp_C = np.concatenate((np.zeros(100),np.sin(np.arange(0,math.pi,math.pi/50))*1.5,np.zeros(50))) * 2.5
amp_D = np.concatenate((np.zeros(150),np.sin(np.arange(0,math.pi,math.pi/50))*1.5)) * 2.5


#num_cols = df_all_data_grid_norm1.shape[1]
num_cols = len(factor_D)
#No noise
df_factorA = pd.DataFrame((np.random.normal(1, 0, [200,num_cols])) * factor_A).multiply(amp_A,axis=0)
df_factorB = pd.DataFrame((np.random.normal(1, 0, [200,num_cols])) * factor_B).multiply(amp_B,axis=0)
df_factorC = pd.DataFrame((np.random.normal(1, 0, [200,num_cols])) * factor_C).multiply(amp_C,axis=0)
df_factorD = pd.DataFrame((np.random.normal(1, 0, [200,num_cols])) * factor_D).multiply(amp_D,axis=0)


df_factorA.columns = df_all_data_grid_norm1.columns
df_factorB.columns = df_all_data_grid_norm1.columns
df_factorC.columns = df_all_data_grid_norm1.columns
df_factorD.columns = df_all_data_grid_norm1.columns

factorA_total = df_factorA.sum(axis=1)
factorB_total = df_factorB.sum(axis=1)
factorC_total = df_factorC.sum(axis=1)
factorD_total = df_factorD.sum(axis=1)


df_4factor = df_factorA + df_factorB + df_factorC + df_factorD
ds_4factor_total = factorA_total + factorB_total + factorC_total + factorD_total
df_4factor_factors = pd.DataFrame([factor_A,factor_B,factor_C,factor_D],columns=df_all_data_grid.columns)
dataset_index_4factor = np.concatenate([np.zeros(50),np.ones(50),np.ones(50)*2,np.ones(50)*3])
dataset_index_4factor_mtx = pd.DataFrame(np.ones(df_4factor.shape)).multiply(dataset_index_4factor,axis=0)


#%%Trying my FVAE, factorisation VAE
df_4factor_aug = augment_data_noise(df_4factor,50,1,0)
df_4factor_01 = df_4factor.clip(lower=0) / df_4factor.max().max()
df_4factor_aug_01 = df_4factor_aug.clip(lower=0) / df_4factor_aug.max().max()


#%%VAE testing
#%%TRAIN 4-FACTOR VAE
ae_input = df_4factor_aug.to_numpy()
ae_input_val = df_4factor.to_numpy()


#%%Build and train CVAE
#beta_schedule = np.array([1e-5,1e-4,1e-3,1e-2,1e-1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
beta_schedule = np.array([1e-5,1e-4,1e-3,1e-2,1e-1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0,1.25])*1e-4

# beta_schedule = np.concatenate([np.zeros(5),np.arange(0.1,1.1,0.1),np.ones(10,),np.arange(0.9,0.45,-0.05)])

# beta_schedule = np.abs(np.sin(np.arange(0,201)*0.45/math.pi))

#beta_schedule = 1 - np.cos(np.arange(0,50)*0.45/math.pi)
input_dim = ae_input.shape[1]
cvae_obj = CVAE_n_layer(input_dim=input_dim,latent_dim=2,int_layers=3,conv_spacing=14,latent_activation='linear',beta_schedule=beta_schedule)

history_cvae = cvae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=20,verbose=1,callbacks=[TerminateOnNaN()])

latent_space = cvae_obj.encoder(ae_input_val).numpy()
#df_latent_space = pd.DataFrame(latent_space,index=df_all_data.index)

fig,ax = plt.subplots(1)
scatter = ax.scatter(latent_space[:,0],latent_space[:,1],c=dataset_index_4factor)
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper left", title="Factors")
ax.add_artist(legend1)
ax.set_title('CVAE latent space')
ax.set_xlabel('Dim 0')
ax.set_ylabel('Dim 1')
#ax.legend([['A','B','C','D'])
plt.show()


fig,ax = plt.subplots(1)
scatter = ax.scatter(ae_input_val,vae_obj.ae(ae_input_val),c=dataset_index_4factor_mtx)
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper left", title="Factors",ncol=4)
ax.add_artist(legend1)
ax.set_title('AE input vs output')
ax.set_xlabel('AE input')
ax.set_ylabel('AE output')
#ax.legend([['A','B','C','D'])
plt.show()

# norm = tfp.distributions.Normal(0, 1)

# grid_x = norm.quantile(np.linspace(0.05, 0.95, 5))
# grid_y = norm.quantile(np.linspace(0.05, 0.95, 5))




#%%Check Pearson's R correlation between input and output factors
latent0_min = latent_space[:,0].min()
latent0_max = latent_space[:,0].max()
latent1_min = latent_space[:,1].min()
latent1_max = latent_space[:,1].max()

latent_grid = np.mgrid[latent0_min:latent0_max:3j, latent1_min:latent1_max:3j].reshape(2, -1).T

df_latent_grid_decoded = pd.DataFrame(cvae_obj.decoder.predict(latent_grid),columns=df_all_data_grid_norm1.columns)
latent_grid_decoded_corr = corr_coeff_rowwise_loops(df_latent_grid_decoded.to_numpy(),df_4factor_factors.to_numpy())
latent_grid_decoded_corr = np.around(latent_grid_decoded_corr,3)

#ds_all_mz = pd.DataFrame(df_all_raw['Molecular Weight'].loc[df_all_data.columns])
fig,ax = plt.subplots(3,3,figsize=(20,10))
axs = ax.ravel()
    
for i in range(9):   
    axs[i].stem(mz_grid,df_latent_grid_decoded.iloc[i],markerfmt=' ')
    best_R = latent_grid_decoded_corr[i]
    axs[i].text(0.95,0.5,latent_grid_decoded_corr[i].max(),horizontalalignment='right', verticalalignment='top',transform=axs[i].transAxes)
    axs[i].text(0.95,0.4,latent_grid_decoded_corr[i].argmax(),horizontalalignment='right', verticalalignment='top',transform=axs[i].transAxes)
    

plt.setp(ax, xlim=(100,400))
plt.tight_layout()
plt.show()
