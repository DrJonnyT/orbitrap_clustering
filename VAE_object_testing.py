# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 14:46:39 2021

@author: mbcx5jt5
"""

""
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


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

#df_beijing_winter = df_beijing_filters.iloc[0:124].copy()
#df_beijing_summer = df_beijing_filters.iloc[124:].copy()


#Make adf_3clust with one from beijing, one from delhi, and one just some bullshit or something
#Check what I did before


df_all_filters = df_beijing_filters.append(df_delhi_filters,sort=True).fillna(0)
# df_all_raw = df_beijing_raw.transpose().append(df_delhi_raw.transpose(),sort=True).transpose()

#%%

#Augment the data
df_aug = augment_data_noise(df_all_filters,50,1,0)

#%%Scale data for input into AE
scalefactor = 1e6
pipe = FunctionTransformer(lambda x: np.divide(x,scalefactor),inverse_func = lambda x: np.multiply(x,scalefactor))
pipe.fit(df_aug)
scaled_df = pd.DataFrame(pipe.transform(df_beijing_filters),columns=df_beijing_filters.columns)
ae_input=scaled_df.to_numpy()
scaled_df_val = pd.DataFrame(pipe.transform(df_beijing_filters), columns=df_beijing_filters.columns,index=df_beijing_filters.index)
ae_input_val = scaled_df_val.to_numpy()

#%%
input_dim = df_all_filters.shape[1]


#%%Testing VAE
vae_obj = VAE_n_layer(input_dim=input_dim,latent_dim=2,int_layers=3,learning_rate=1e-3)
#history = vae_obj.fit_model(ae_input,x_test=ae_input_val, epochs=300)
history = vae_obj.fit_model((df_all_filters/1e6).to_numpy(), epochs=300)
val_acc_per_epoch = history.history['val_loss']
plt.plot(val_acc_per_epoch)

#%%
vae_latent_space = vae_obj.encode((df_all_filters/1e6).to_numpy())
plt.pcolormesh(vae_latent_space)

#%%
plt.plot(vae_latent_space[:,0])
plt.plot(vae_latent_space[:,1])

#%%Plot input vs output
latent_space = vae_obj.encoder((df_all_filters/1e6).to_numpy()).numpy()
decoded_latent_space = vae_obj.decoder(latent_space)

fig,ax = plt.subplots(1)
plt.scatter((df_all_filters/1e6).to_numpy(),vae_obj.vae((df_all_filters/1e6).to_numpy()))
plt.title("VAE input vs output")
plt.xlabel('VAE input')
plt.ylabel('VAE output')
plt.show()




# %%Make 3cluster data

#Let's make some fake data with 3 well defined clusters to see if we can pick them apart
#max_filter = df_beijing_filters.loc[beijing_rows_sum.idxmax()].ravel()
#min_filter = df_beijing_filters.loc[beijing_rows_sum.idxmin()].ravel()

beijing_filter = df_all_filters.iloc[0].ravel()
delhi_filter = df_all_filters.iloc[400].ravel()

num_filters = df_all_filters.shape[0]
num_cols = df_all_filters.shape[1]

#Now make a pandas frame cointaining just these two, but with added 5% noise
clust0 = pd.DataFrame((np.random.normal(0, 0.05, [int(np.ceil(num_filters/2)),num_cols])+1) * beijing_filter)
clust1 = pd.DataFrame((np.random.normal(0, 0.05, [int(num_filters/2),num_cols])+1) * delhi_filter)
df_2clust = clust0.append(clust1)
df_2clust.columns = df_all_filters.columns
df_2clust.head()

thousand_filter = np.random.normal(1000, 50, num_cols)
clust2 = pd.DataFrame((np.random.normal(0, 0.05, [int(np.ceil(num_filters/2)),num_cols])+1) * thousand_filter )
df_3clust = clust0.append(clust1.append(clust2))
df_3clust.columns = df_all_filters.columns

fig,ax = plt.subplots(1)
ax.plot(clust0.sum(axis=1))
ax.plot(clust1.sum(axis=1),c='k')
ax.plot(clust2.sum(axis=1),c='r')
plt.show()


# %%Let's make 3 factors that vary with time

num_filters = df_all_filters.shape[0]
num_cols = df_all_filters.shape[1]

#Visually these all look a bit different on the time series matrix plot
#Cluster_A = df_beijing_filters.iloc[89].ravel() #Q5 photochemical age, Timestamp('2017-06-11 09:02:30')
#Cluster_B = df_beijing_filters.iloc[158].ravel() #Q50 photochemical age, Timestamp('2017-06-20 16:43:00')
#Cluster_C = df_beijing_filters.iloc[106].ravel()    #Q95 photochemical age, Timestamp('2017-06-12 15:56:30')

Cluster_A = df_all_filters.iloc[0].ravel() #A random filter from Beijing winter
Cluster_B = df_all_filters.iloc[130].ravel() #A random filter from Beijing summer
Cluster_C = df_all_filters.iloc[400].ravel() #A random filter from Delhi




#Normalise all to 1
Cluster_A = 1 * Cluster_A / Cluster_A.sum()
Cluster_B = np.ones(num_cols)/num_cols #1 * Cluster_B / Cluster_B.sum()
Cluster_C = 1 * Cluster_C / Cluster_C.sum()


#For just the winter data, when it's normalised
amp_A = np.append(np.zeros(83),np.arange(0,1,0.015)) * 5 + 1
amp_B = np.ones(150)*2.75
amp_C = np.append(np.arange(1,0,-0.015),np.zeros(83)) * 5 + 0.5


df_clustA = pd.DataFrame((np.random.normal(1, 0.01, [150,num_cols])) * Cluster_A).multiply(amp_A,axis=0)
df_clustB = pd.DataFrame((np.random.normal(1, 0.01, [150,num_cols])) * Cluster_B).multiply(amp_B,axis=0)
df_clustC = pd.DataFrame((np.random.normal(1, 0.01, [150,num_cols])) * Cluster_C).multiply(amp_C,axis=0)
df_clustA.columns = df_all_filters.columns
df_clustB.columns = df_all_filters.columns
df_clustC.columns = df_all_filters.columns

clustA_total = df_clustA.sum(axis=1)
clustB_total = df_clustB.sum(axis=1)
clustC_total = df_clustC.sum(axis=1)


df_3clust_sum = pd.DataFrame([clustA_total,clustB_total,clustC_total]).T
df_3clust_sum.idxmax(axis="columns").plot()
plt.title("Input test cluster labels")
plt.show()

df_3clust = df_clustA + df_clustB + df_clustC

plt.plot(clustA_total,label='Factor A')
plt.plot(clustB_total,label='Factor B')
plt.plot(clustC_total,label='Factor C')
plt.legend()
plt.xlabel('Sample number')
plt.ylabel('Sum of total magnitude')
plt.title('3-factor synthetic data')
plt.show()

df_2clust = df_clustA + df_clustC
plt.plot(clustA_total,label='Factor A')
plt.plot(clustC_total,label='Factor C')
plt.xlabel('Sample number')
plt.ylabel('Sum of total magnitude')
plt.title('2-factor synthetic data')
plt.legend()
plt.show()

#%%Testing VAE with 2clust data
vae_obj = VAE_n_layer(input_dim=input_dim,latent_dim=2,int_layers=3,learning_rate=1e-3)
#history = vae_obj.fit_model(ae_input,x_test=ae_input_val, epochs=300)
history = vae_obj.fit_model((df_2clust).to_numpy(), epochs=500)
val_acc_per_epoch = history.history['val_loss']
plt.plot(val_acc_per_epoch)

#%%
vae_latent_space = vae_obj.encode((df_2clust).to_numpy())
plt.pcolormesh(vae_latent_space)

#%%
plt.plot(vae_latent_space[:,0])
plt.plot(vae_latent_space[:,1])
#plt.plot(vae_latent_space[:,2])

#%%Plot input vs output
latent_space = vae_obj.encoder(df_2clust.to_numpy()).numpy()
decoded_latent_space = vae_obj.decoder(latent_space)

fig,ax = plt.subplots(1)
plt.scatter(df_3clust.to_numpy(),vae_obj.vae(df_2clust.to_numpy()))
plt.title("VAE input vs output")
plt.xlabel('VAE input')
plt.ylabel('VAE output')
plt.show()

#%%Get time series of each column invididually - 2 factor
latent_space_col0 = vae_latent_space.copy()
latent_space_col1 = vae_latent_space.copy()
latent_space_col0[:,1] = 0
latent_space_col1[:,0] = 0

latent_col0_decoded = vae_obj.decoder(latent_space_col0).numpy()
latent_col0_decoded_sum = latent_col0_decoded.sum(axis=1)
latent_col1_decoded = vae_obj.decoder(latent_space_col1).numpy()
latent_col1_decoded_sum = latent_col1_decoded.sum(axis=1)
plt.plot(latent_col0_decoded_sum,label='VAE latent space column 0')
plt.plot(latent_col1_decoded_sum,label='VAE latent space column 1')
plt.xlabel('Sample number')
plt.ylabel('Sum of total magnitude')
plt.title('VAE attempt to pick out 2 factors')
plt.ylim(bottom=0)
plt.legend()
plt.show()






#%%Testing VAE with 3clust data
vae_obj = VAE_n_layer(input_dim=input_dim,latent_dim=3,int_layers=3,learning_rate=1e-3)
#history = vae_obj.fit_model(ae_input,x_test=ae_input_val, epochs=300)
history = vae_obj.fit_model((df_3clust).to_numpy(), epochs=500)
val_acc_per_epoch = history.history['val_loss']
plt.plot(val_acc_per_epoch)

#%%
vae_latent_space = vae_obj.encode((df_3clust).to_numpy())
plt.pcolormesh(vae_latent_space)

#%%
plt.plot(vae_latent_space[:,0])
plt.plot(vae_latent_space[:,1])
plt.plot(vae_latent_space[:,2])

#%%Plot input vs output
latent_space = vae_obj.encoder(df_3clust.to_numpy()).numpy()
decoded_latent_space = vae_obj.decoder(latent_space)

fig,ax = plt.subplots(1)
plt.scatter(df_3clust.to_numpy(),vae_obj.vae(df_3clust.to_numpy()))
plt.title("VAE input vs output")
plt.xlabel('VAE input')
plt.ylabel('VAE output')
plt.show()

#%%Get time series of each column invididually - 3 factor
latent_space_col0 = vae_latent_space.copy()
latent_space_col1 = vae_latent_space.copy()
latent_space_col2 = vae_latent_space.copy()
latent_space_col0[:,1:3] = 0
latent_space_col1[:,[0,2]] = 0
latent_space_col2[:,0:2] = 0

latent_col0_decoded = vae_obj.decoder(latent_space_col0).numpy()
latent_col0_decoded_sum = latent_col0_decoded.sum(axis=1)
latent_col1_decoded = vae_obj.decoder(latent_space_col1).numpy()
latent_col1_decoded_sum = latent_col1_decoded.sum(axis=1)
latent_col2_decoded = vae_obj.decoder(latent_space_col2).numpy()
latent_col2_decoded_sum = latent_col2_decoded.sum(axis=1)
plt.plot(latent_col0_decoded_sum,label='VAE latent space column 0')
plt.plot(latent_col1_decoded_sum,label='VAE latent space column 1')
plt.plot(latent_col2_decoded_sum,label='VAE latent space column 2')
plt.xlabel('Sample number')
plt.ylabel('Sum of total magnitude')
plt.title('VAE attempt to pick out 3 factors')
plt.ylim(bottom=0)
plt.legend()
plt.show()






#%%See if you can use PMF to pick out the factors
# %%PMF on 3 factor test data
#Now do some NMF clustering
from sklearn.decomposition import NMF
num_nmf_factors = 3
nmf_model = NMF(n_components=num_nmf_factors)
#model.fit(latent_space)

#Get the different components
#nmf_features = model.transform(latent_space)
#print(model.components_)

df_2clust_noneg = df_2clust.clip(lower=0)
df_3clust_noneg = df_3clust.clip(lower=0)

W = nmf_model.fit_transform(df_3clust_noneg.to_numpy())
#W = model.fit_transform(df_3clust.values)
H = nmf_model.components_

Factor0 = H[0]
Factor1 = H[1]
Factor2 = H[2]

Factor0_mtx = np.outer(W.T[0], H[0])
Factor1_mtx = np.outer(W.T[1], H[1])
Factor2_mtx = np.outer(W.T[2], H[2])

Factor0_sum = Factor0_mtx.sum(axis=1)
Factor1_sum = Factor1_mtx.sum(axis=1)
Factor2_sum = Factor2_mtx.sum(axis=1)

plt.plot(Factor0_sum,label='PMF factor 0')
plt.plot(Factor1_sum,label='PMF factor 1')
plt.plot(Factor2_sum,label='PMF factor 2')
plt.ylim(bottom=0)
plt.legend()
plt.ylabel('Sum of total magnitude')
plt.xlabel('Filter number')
plt.title('PMF attempt to pick out 3 factors')
plt.show()

# %%PMF on 2 factor test data
#Now do some NMF clustering
num_nmf_factors = 2
nmf_model = NMF(n_components=num_nmf_factors)
#model.fit(latent_space)

#Get the different components
#nmf_features = model.transform(latent_space)
#print(model.components_)

df_2clust_noneg = df_2clust.clip(lower=0)

W = nmf_model.fit_transform(df_2clust_noneg.to_numpy())
H = nmf_model.components_

Factor0 = H[0]
Factor1 = H[1]

Factor0_mtx = np.outer(W.T[0], H[0])
Factor1_mtx = np.outer(W.T[1], H[1])

Factor0_sum = Factor0_mtx.sum(axis=1)
Factor1_sum = Factor1_mtx.sum(axis=1)

plt.plot(Factor0_sum,label='PMF factor 0')
plt.plot(Factor1_sum,label='PMF factor 1')
plt.ylim(bottom=0)
plt.legend()
plt.ylabel('Sum of total magnitude')
plt.xlabel('Filter number')
plt.title('PMF attempt to pick out 2 factors')
plt.show()


# %%PMF on 2 factor latent space test data
#First make the VAE and latent space
vae_obj = VAE_n_layer(input_dim=df_2clust.shape[1],latent_dim=3,int_layers=3,learning_rate=1e-3,latent_activation='relu')
#history = vae_obj.fit_model(ae_input,x_test=ae_input_val, epochs=300)
history = vae_obj.fit_model((df_2clust).to_numpy(), epochs=500)
vae_latent_space_2col = vae_obj.encode((df_2clust).to_numpy())
#%%
#Now do some NMF clustering
num_nmf_factors = 2
nmf_model = NMF(n_components=num_nmf_factors)
#model.fit(latent_space)

#Get the different components
#nmf_features = model.transform(latent_space)
#print(model.components_)

W = nmf_model.fit_transform(vae_latent_space_2col)
H = nmf_model.components_

Factor0 = H[0]
Factor1 = H[1]

Factor0_mtx = np.outer(W.T[0], H[0])
Factor1_mtx = np.outer(W.T[1], H[1])

Factor0_decoded_mtx = vae_obj.decode(Factor0_mtx)
Factor1_decoded_mtx = vae_obj.decode(Factor1_mtx)

Factor0_decoded_sum = Factor0_decoded_mtx.sum(axis=1)
Factor1_decoded_sum = Factor1_decoded_mtx.sum(axis=1)

plt.plot(Factor0_decoded_sum,label='PMF factor 0')
plt.plot(Factor1_decoded_sum,label='PMF factor 1')
plt.ylim(bottom=0)
plt.legend()
plt.ylabel('Sum of total magnitude')
plt.xlabel('Filter number')
plt.title('PMF attempt to pick out 2 factors from latent space')
plt.show()


# %%PMF on 3 factor latent space test data
#First make the VAE and latent space
vae_obj = VAE_n_layer(input_dim=df_3clust.shape[1],latent_dim=3,int_layers=3,learning_rate=1e-3,latent_activation='relu')
#history = vae_obj.fit_model(ae_input,x_test=ae_input_val, epochs=300)
history = vae_obj.fit_model((df_3clust).to_numpy(), epochs=500)
vae_latent_space_3col = vae_obj.encode((df_3clust).to_numpy())
#%%
#Now do some NMF clustering
num_nmf_factors = 3
nmf_model = NMF(n_components=num_nmf_factors)
#model.fit(latent_space)

#Get the different components
#nmf_features = model.transform(latent_space)
#print(model.components_)

W = nmf_model.fit_transform(vae_latent_space_3col)
H = nmf_model.components_

Factor0 = H[0]
Factor1 = H[1]
Factor2 = H[2]

Factor0_mtx = np.outer(W.T[0], H[0])
Factor1_mtx = np.outer(W.T[1], H[1])
Factor2_mtx = np.outer(W.T[2], H[2])

Factor0_decoded_mtx = vae_obj.decode(Factor0_mtx)
Factor1_decoded_mtx = vae_obj.decode(Factor1_mtx)
Factor2_decoded_mtx = vae_obj.decode(Factor2_mtx)

Factor0_decoded_sum = Factor0_decoded_mtx.sum(axis=1)
Factor1_decoded_sum = Factor1_decoded_mtx.sum(axis=1)
Factor2_decoded_sum = Factor2_decoded_mtx.sum(axis=1)

plt.plot(Factor0_decoded_sum,label='PMF factor 0')
plt.plot(Factor1_decoded_sum,label='PMF factor 1')
plt.plot(Factor2_decoded_sum,label='PMF factor 2')
plt.ylim(bottom=0)
plt.legend()
plt.ylabel('Sum of total magnitude')
plt.xlabel('Filter number')
plt.title('PMF attempt to pick out 3 factors from latent space')
plt.show()



# %%
prefix = "Beijing"
factor_sum_filename = prefix + str(range)

df_latent_factors_mtx = pd.DataFrame()
df_factor_profiles = pd.DataFrame(columns=df_beijing_filters.columns)
df_factorsum_tseries = pd.DataFrame()

#Extract and save all the components
#Factor names like for 5 factors, it would go factor5_0, factor5_1...factor5_4
for factor in range(num_nmf_factors):
    factor_name = ("factor_"+str(num_nmf_factors))+"_"+str(factor)
    Factor_lat = H[factor]
    Factor_lat_mtx = np.outer(W.T[factor], H[factor])
    
    #Factor profile
    Factor_decod = pipe.inverse_transform(decoder_ae.predict(np.expand_dims(Factor_lat, axis=0)))

    #Time series of factor as a matrix
    Factor_mtx_decod = pipe.inverse_transform(decoder_ae.predict(Factor_lat_mtx))
    Factor_decod_sum = Factor_mtx_decod.sum(axis=1)
    
    df_latent_factors_mtx = df_latent_factors_mtx.append(pd.DataFrame(Factor_lat_mtx),index=[factor_name])
    df_factor_profiles = df_factor_profiles.append(pd.DataFrame(Factor_decod,columns=df_beijing_filters.columns,index=[factor_name]))
    #factor_sums_tseries.append(Factor_decod_sum,axis=1)
    df_factorsum_tseries[factor_name] = Factor_decod_sum
    
df_factorsum_tseries.index = df_beijing_filters.index


#What is the total residual error in the latent data?

#What is the residual error in the real data?

nmf_total_sum =  df_factorsum_tseries.sum(axis=1)
nmf_residual = beijing_rows_sum - nmf_total_sum
nmf_residual_pct = (nmf_residual / beijing_rows_sum)*100

plt.plot(df_factorsum_tseries)
    
# %%
#1 What is the time series of the 2 factors? Need each factor as a t series
Factor0_lat = H[0]
Factor1_lat = H[1]
#Factor2_lat = H[2]


Factor0_lat_mtx = np.outer(W.T[0], H[0])
Factor1_lat_mtx = np.outer(W.T[1], H[1])
#Factor2_lat_mtx = np.outer(W.T[2], H[2])

#Now need to decode these matrices to get the time series matrix of each factor

Factor0_decod = pipe.inverse_transform(decoder_ae.predict(np.expand_dims(Factor0_lat, axis=0)))
Factor1_decod = pipe.inverse_transform(decoder_ae.predict(np.expand_dims(Factor1_lat, axis=0)))
#Factor2_decod = pipe.inverse_transform(decoder_ae.predict(np.expand_dims(Factor2_lat, axis=0)))


Factor0_mtx_decod = pipe.inverse_transform(decoder_ae.predict(Factor0_lat_mtx))
Factor1_mtx_decod = pipe.inverse_transform(decoder_ae.predict(Factor1_lat_mtx))
#Factor2_mtx_decod = pipe.inverse_transform(decoder_ae.predict(Factor2_lat_mtx))

Factor0_decod_sum = Factor0_mtx_decod.sum(axis=1)
Factor1_decod_sum = Factor1_mtx_decod.sum(axis=1)
#Factor2_decod_sum = Factor2_mtx_decod.sum(axis=1)


plt.plot(Factor0_decod_sum)
plt.plot(Factor1_decod_sum)
#plt.plot(Factor2_decod_sum)

plt.ylim(bottom=0)
plt.show()