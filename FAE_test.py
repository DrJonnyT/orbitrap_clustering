
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

#Need this because otherwise the custom losses don't work
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


from sklearn.preprocessing import RobustScaler, StandardScaler,FunctionTransformer,MinMaxScaler,Normalizer,QuantileTransformer
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
df_all_err.columns= mz_columns['Molecular Weight']
df_all_err.sort_index(axis=1,inplace=True)
df_all_err.columns = mz_columns_sorted.index
mz_columns = mz_columns_sorted

#%%Compare inner vs outer join
df_all_data_outer = pd.concat([df_beijing_data, df_delhi_data], axis=0, join="outer")
df_all_data_outer_sum = df_all_data_outer.sum(axis=1)
df_all_data_inner_sum = df_all_data.sum(axis=1)
m, b = np.polyfit(x=df_all_data_outer_sum, y=df_all_data_inner_sum,deg=1)

sns.scatterplot(df_all_data_outer_sum,df_all_data_inner_sum,marker='o',hue=dataset_cat)
plt.plot(df_all_data_outer_sum, m*df_all_data_outer_sum + b,c='k')
plt.xlabel('Outer join total concentration (µg/m3)')
plt.ylabel('Inner join total concentration (µg/m3)')
plt.text(700,50,"y = " + str(round(m,3)) + " + " + str(round(b,2)),horizontalalignment='right', verticalalignment='bottom')


#%%Work out O:C, H:C, S:C, N:C ratios for all peaks
df_element_ratios = mz_columns.index.get_level_values(0).to_series().apply(lambda x: chemform_ratios(x)[0])
df_element_ratios = pd.DataFrame()
df_element_ratios['H/C'] = mz_columns.index.get_level_values(0).to_series().apply(lambda x: chemform_ratios(x)[0])
df_element_ratios['O/C'] = mz_columns.index.get_level_values(0).to_series().apply(lambda x: chemform_ratios(x)[1])
df_element_ratios['N/C'] = mz_columns.index.get_level_values(0).to_series().apply(lambda x: chemform_ratios(x)[2])
df_element_ratios['S/C'] = mz_columns.index.get_level_values(0).to_series().apply(lambda x: chemform_ratios(x)[3])



#%%Load chemform namelists
chemform_namelist_beijing = load_chemform_namelist(path + 'Beijing_Amb3.1_MZ.xlsx')
chemform_namelist_delhi = load_chemform_namelist(path + 'Delhi_Amb3.1_MZ.xlsx')
chemform_namelist_all = combine_chemform_namelists(chemform_namelist_beijing,chemform_namelist_delhi)


#%%Prescale datasets
#Divide whole thing by 1e6
scalefactor = 1e6
pipe_1e6 = FunctionTransformer(lambda x: np.divide(x,scalefactor),inverse_func = lambda x: np.multiply(x,scalefactor))
pipe_1e6.fit(df_all_data)

df_all_data_1e6 = pd.DataFrame(pipe_1e6.transform(df_all_data),columns=df_all_data.columns)
ds_all_filters_total_1e6 = df_all_data_1e6.sum(axis=1)

#Normalise so the mean of the whole matrix is 1
orig_mean = df_all_data.mean().mean()
pipe_norm1_mtx = FunctionTransformer(lambda x: np.divide(x,orig_mean),inverse_func = lambda x: np.multiply(x,orig_mean))
pipe_norm1_mtx.fit(df_all_data)
df_all_data_norm1 = pd.DataFrame(pipe_norm1_mtx.transform(df_all_data),columns=df_all_data.columns)

#Minmax scaling
minmaxscaler_all = MinMaxScaler()
df_all_data_minmax = pd.DataFrame(minmaxscaler_all.fit_transform(df_all_data.to_numpy()),columns=df_all_data.columns)

#Standard scaling
standardscaler_all = StandardScaler()
df_all_data_standard = pd.DataFrame(standardscaler_all.fit_transform(df_all_data.to_numpy()),columns=df_all_data.columns)

#Robust scaling
robustscaler_all = RobustScaler()
df_all_data_robust = pd.DataFrame(robustscaler_all.fit_transform(df_all_data.to_numpy()),columns=df_all_data.columns)

#df scaled so it is normalised by the total from each filter
df_all_data_norm = df_all_data.div(df_all_data.sum(axis=1), axis=0)

#Log data and add one
offset_min = df_all_data.min().min() * (-1)
pipe_log1p = FunctionTransformer(lambda x: np.log1p(x+offset_min),inverse_func = lambda x: (np.expm1(x) - offset_min) )
df_all_data_log1p = pd.DataFrame(pipe_log1p.fit_transform(df_all_data.to_numpy()),columns=df_all_data.columns)




#%%Try autoencoder
df_all_data_aug = augment_data_noise(df_all_data_norm1,25,1,0)
ae_input = df_aug.values
ae_input_val = df_all_data_norm1.values
input_dim = ae_input.shape[1]

ae_input = ae_input.clip(min=0)
ae_input_val = ae_input_val.clip(min=0)





#%%Test with relu for latent space activation function

latent_dims = []
AE1_MSE_best50epoch =[]
AE2_MSE_best50epoch =[]
AE3_MSE_best50epoch =[]
AE4_MSE_best50epoch =[]
best_hps_array = []


verbose = 0

start_time = time.time()


for latent_dim in range(5,30):
    print(latent_dim)
    K.clear_session()
    latent_dims.append(latent_dim)
    
    #Test for 1 intermediate layer
    ae_obj = FAE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=1,latent_activation='relu',decoder_output_activation='relu')
    history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=100,verbose=verbose)
    val_acc_per_epoch = history.history['val_loss']
    AE1_MSE_best50epoch.append(min(history.history['val_loss']))
    
    #Test for 2 intermediate layers
    ae_obj = FAE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=2,latent_activation='relu',decoder_output_activation='relu')
    history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=100,verbose=verbose)
    val_acc_per_epoch = history.history['val_loss']
    AE2_MSE_best50epoch.append(min(history.history['val_loss']))
    
    #Test for 3 intermediate layers
    ae_obj = FAE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=3,latent_activation='relu',decoder_output_activation='relu')
    history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=100,verbose=verbose)
    val_acc_per_epoch = history.history['val_loss']
    AE3_MSE_best50epoch.append(min(history.history['val_loss']))
    
    #Test for 4 intermediate layers
    ae_obj = FAE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=4,latent_activation='relu',decoder_output_activation='relu')
    history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=100,verbose=verbose)
    val_acc_per_epoch = history.history['val_loss']
    AE4_MSE_best50epoch.append(min(history.history['val_loss']))
    
print("--- %s seconds ---" % (time.time() - start_time))


#%%
# ae_obj = FAE_n_layer(input_dim=input_dim,latent_dim=5,int_layers=1,latent_activation='relu',decoder_output_activation='relu')
# history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=100,verbose=verbose)
# val_acc_per_epoch = history.history['val_loss']
# AE1_MSE_best50epoch.append(min(history.history['val_loss']))
    
#%%Plot the data for the different number of layers above
fig,ax=plt.subplots(2,1,figsize=(12,8))
ax[0].set_title('Finding optimum latent dims- simple relu AE with relu latent space')
ax[0].plot(latent_dims,AE1_MSE_best50epoch,c='black')
ax[0].plot(latent_dims,AE2_MSE_best50epoch,c='red')
ax[0].plot(latent_dims,AE3_MSE_best50epoch,c='gray')
#ax[0].plot(latent_dims,AE4_MSE_best50epoch)
ax[0].set_xlabel('Number of latent dims')
ax[0].set_ylabel('Best MSE in first 50 epochs')
ax[1].plot(latent_dims,AE1_MSE_best50epoch,c='black')
ax[1].plot(latent_dims,AE2_MSE_best50epoch,c='red')
ax[1].plot(latent_dims,AE3_MSE_best50epoch,c='gray')
#ax[1].plot(latent_dims,AE4_MSE_best50epoch)
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
#Based on the above, use an AE with 1 intermediate layers and latent dim of 4
ae_obj = FAE_n_layer(input_dim=input_dim,latent_dim=6,int_layers=1,latent_activation='relu',decoder_output_activation='relu')
history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=100)
val_acc_per_epoch = history.history['val_loss']
fig,ax = plt.subplots(1,figsize=(8,6))
ax.plot(val_acc_per_epoch)
plt.show()
#Now retrain model based on best epoch
best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
ae_obj = FAE_n_layer(input_dim=input_dim,latent_dim=6,int_layers=1,latent_activation='relu',decoder_output_activation='relu')
ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=best_epoch)
print('Best epoch: %d' % (best_epoch,))


fig,ax=plt.subplots(2,1,figsize=(12,8))
ax[0].set_title('Finding optimum epochs- simple relu AE, relu latent space')
ax[0].plot(history.epoch,val_acc_per_epoch)
ax[0].set_xlabel('Number of epochs')
ax[0].set_ylabel('MSE')
ax[1].plot(history.epoch,val_acc_per_epoch)
ax[1].set_xlabel('Number of latent dims')
ax[1].set_ylabel('MSE')
ax[1].set_yscale('log')
#loc = plticker.MultipleLocator(base=10.0) # this locator puts ticks at regular intervals
#ax[0].xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
#ax[0].xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
#ax[1].xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
#ax[1].xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
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

#%%Try decoding each latent space column
lat0 = np.zeros(latent_space.shape)
lat1 = np.zeros(latent_space.shape)
lat2 = np.zeros(latent_space.shape)
lat3 = np.zeros(latent_space.shape)

lat0[:,0] = latent_space[:,0]
lat1[:,1] = latent_space[:,1]
lat2[:,2] = latent_space[:,2]
lat3[:,3] = latent_space[:,3]

lat0_decoded = ae_obj.decoder.predict(lat0)
lat1_decoded = ae_obj.decoder.predict(lat1)
lat2_decoded = ae_obj.decoder.predict(lat2)
lat3_decoded = ae_obj.decoder.predict(lat3)
decoded_total_mtx = lat0_decoded + lat1_decoded + lat2_decoded + lat3_decoded







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
            corr_mtx[i,j] = pearsonr(Arow,Brow)[0]
    return corr_mtx
            

Beijing_rows_corr = corr_coeff_rowwise_loops(df_all_data_norm1[ds_dataset_cat=='Beijing_winter'].values,df_all_data_norm1[ds_dataset_cat=='Beijing_summer'].values)
Beijing_rows_corr_min_index = np.unravel_index(Beijing_rows_corr.argmin(), Beijing_rows_corr.shape)
Delhi_rows_corr = corr_coeff_rowwise_loops(df_all_data_norm1[ds_dataset_cat=='Delhi_summer'].values,df_all_data_norm1[ds_dataset_cat=='Delhi_autumn'].values)
Delhi_rows_corr_min_index = np.unravel_index(Delhi_rows_corr.argmin(), Delhi_rows_corr.shape)


#This then is 4 factors that are very poorly correlated with each other in terms of their mass spec
factor_A = df_all_data_norm1[ds_dataset_cat=='Beijing_winter'].iloc[Beijing_rows_corr_min_index[0]].values
factor_B = df_all_data_norm1[ds_dataset_cat=='Beijing_summer'].iloc[Beijing_rows_corr_min_index[1]].values
factor_C = df_all_data_norm1[ds_dataset_cat=='Delhi_summer'].iloc[Delhi_rows_corr_min_index[0]].values
factor_D = df_all_data_norm1[ds_dataset_cat=='Delhi_autumn'].iloc[Delhi_rows_corr_min_index[1]].values

#Normalise all to 1
factor_A = 1 * factor_A / factor_A.sum()
factor_B = 1 * factor_B / factor_B.sum()
factor_C = 1 * factor_C / factor_C.sum()
factor_D = 1 * factor_D / factor_D.sum()
#Make factor E so all columns between mz 350 -- 400 are...dataful...
factor_E  = np.ones(df_all_data.shape[1])
factor_E[np.ravel(np.logical_and(mz_columns.to_numpy() > 350, mz_columns.to_numpy() < 400))] = 50
factor_E = factor_E * np.random.normal(1, 0.3, [df_all_data.shape[1]]).clip(min=0)
factor_E = 1 * factor_E / factor_E.sum()

#%%Hard mode factor amplitudes

amp_A = np.append(np.arange(1.5,0.5,-0.015),np.arange(0.5,0,-0.5/83)) * 2.5
amp_B = np.abs(np.sin(np.arange(0,2*math.pi,math.pi/75))*2.5) + np.abs(-np.sin(np.arange(0,2*math.pi,math.pi/75))*2.5)
amp_C = np.abs(-np.sin(np.arange(0,3*math.pi,math.pi/50))*2 + 1)
amp_D = np.append(np.arange(0.1,0.6,0.5/83),np.arange(0.6,1.6,0.015)) * 2.5
amp_E = np.sin(np.arange(0,math.pi,math.pi/150))*1.5 * np.random.normal(1, 0.3, 150)


num_cols = df_all_data.shape[1]
#With noise
#df_factorA = pd.DataFrame((np.random.normal(1, 0.3, [150,num_cols])) * factor_A).multiply(amp_A,axis=0)
#df_factorB = pd.DataFrame((np.random.normal(1, 0.3, [150,num_cols])) * factor_B).multiply(amp_B,axis=0)
#df_factorC = pd.DataFrame((np.random.normal(1, 0.3, [150,num_cols])) * factor_C).multiply(amp_C,axis=0)
#No noise
df_factorA = pd.DataFrame((np.random.normal(1, 0, [150,num_cols])) * factor_A).multiply(amp_A,axis=0)
df_factorB = pd.DataFrame((np.random.normal(1, 0, [150,num_cols])) * factor_B).multiply(amp_B,axis=0)
df_factorC = pd.DataFrame((np.random.normal(1, 0, [150,num_cols])) * factor_C).multiply(amp_C,axis=0)
df_factorD = pd.DataFrame((np.random.normal(1, 0, [150,num_cols])) * factor_D).multiply(amp_D,axis=0)
df_factorE = pd.DataFrame((np.random.normal(1, 0, [150,num_cols])) * factor_E).multiply(amp_E,axis=0)

df_factorA.columns = df_all_data_norm1.columns
df_factorB.columns = df_all_data_norm1.columns
df_factorC.columns = df_all_data_norm1.columns
df_factorD.columns = df_all_data_norm1.columns
df_factorE.columns = df_all_data_norm1.columns

factorA_total = df_factorA.sum(axis=1)
factorB_total = df_factorB.sum(axis=1)
factorC_total = df_factorC.sum(axis=1)
factorD_total = df_factorD.sum(axis=1)
factorE_total = df_factorE.sum(axis=1)


df_3factor_sum = pd.DataFrame([factorA_total,factorB_total,factorC_total]).T
df_3factor_sum.idxmax(axis="columns").plot()
plt.title("Input test factor labels")
plt.show()

df_3factor = df_factorA + df_factorB + df_factorC

plt.plot(factorA_total,label='0')
plt.plot(factorB_total,label='1')
plt.plot(factorC_total,label='2')
plt.plot(factorD_total,c='k',label='3')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          ncol=4, fancybox=True,title="Factor number")
plt.title("Test data factor time series (hard mode)")
plt.show()

ds_3factor_total = factorA_total + factorB_total + factorC_total



df_2factor = df_factorA + df_factorC
ds_2factor_total = factorA_total + factorC_total

df_4factor = df_factorA + df_factorB + df_factorC + df_factorD
ds_4factor_total = factorA_total + factorB_total + factorC_total + factorD_total
df_4factor_factors = pd.DataFrame([factor_A,factor_B,factor_C,factor_D],columns=df_all_data.columns)
df_4factor_sum = pd.DataFrame([factorA_total,factorB_total,factorC_total,factorD_total]).T
df_4factor_sum.idxmax(axis="columns").plot()
dataset_index_4factor = df_4factor_sum.idxmax(axis="columns")



df_5factor = df_factorA + df_factorB + df_factorC + df_factorD + df_factorE
ds_5factor_total = factorA_total + factorB_total + factorC_total + factorD_total + factorE_total
df_5factor_factors = pd.DataFrame([factor_A,factor_B,factor_C,factor_D,factor_E],columns=df_all_data.columns)


#%% VERY EASY AMPLITUDES
#Amplitudes to make it very easy for the model to pick out each factor
amp_A = np.concatenate((np.sin(np.arange(0,math.pi,math.pi/50))*1.5,np.zeros(150))) * 2.5
amp_B = np.concatenate((np.zeros(50),np.sin(np.arange(0,math.pi,math.pi/50))*1.5,np.zeros(100))) * 2.5
amp_C = np.concatenate((np.zeros(100),np.sin(np.arange(0,math.pi,math.pi/50))*1.5,np.zeros(50))) * 2.5
amp_D = np.concatenate((np.zeros(150),np.sin(np.arange(0,math.pi,math.pi/50))*1.5)) * 2.5


num_cols = df_all_data.shape[1]
#No noise
df_factorA = pd.DataFrame((np.random.normal(1, 0, [200,num_cols])) * factor_A).multiply(amp_A,axis=0)
df_factorB = pd.DataFrame((np.random.normal(1, 0, [200,num_cols])) * factor_B).multiply(amp_B,axis=0)
df_factorC = pd.DataFrame((np.random.normal(1, 0, [200,num_cols])) * factor_C).multiply(amp_C,axis=0)
df_factorD = pd.DataFrame((np.random.normal(1, 0, [200,num_cols])) * factor_D).multiply(amp_D,axis=0)


df_factorA.columns = df_all_data_norm1.columns
df_factorB.columns = df_all_data_norm1.columns
df_factorC.columns = df_all_data_norm1.columns
df_factorD.columns = df_all_data_norm1.columns

factorA_total = df_factorA.sum(axis=1)
factorB_total = df_factorB.sum(axis=1)
factorC_total = df_factorC.sum(axis=1)
factorD_total = df_factorD.sum(axis=1)


df_3factor_sum = pd.DataFrame([factorA_total,factorB_total,factorC_total]).T
df_3factor_sum.idxmax(axis="columns").plot()
plt.title("Input test factor labels")
plt.show()

df_3factor = df_factorA + df_factorB + df_factorC

plt.plot(factorA_total,label='0')
plt.plot(factorB_total,label='1')
plt.plot(factorC_total,label='2')
plt.plot(factorD_total,c='k',label='3')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          ncol=4, fancybox=True,title="Factor number")
plt.title("Test data factor time series (easy mode)")
plt.show()

ds_3factor_total = factorA_total + factorB_total + factorC_total



df_2factor = df_factorA + df_factorC
df_2factor = pd.concat([df_2factor.iloc[0:50],df_2factor.iloc[100:150]])
ds_2factor_total = factorA_total + factorC_total

df_4factor = df_factorA + df_factorB + df_factorC + df_factorD
ds_4factor_total = factorA_total + factorB_total + factorC_total + factorD_total
df_4factor_factors = pd.DataFrame([factor_A,factor_B,factor_C,factor_D],columns=df_all_data.columns)
dataset_index_4factor = np.concatenate([np.zeros(50),np.ones(50),np.ones(50)*2,np.ones(50)*3])


#%%Trying my FVAE, factorisation VAE
df_2factor_aug = augment_data_noise(df_2factor,50,1,0)
df_4factor_aug = augment_data_noise(df_4factor,50,1,0)
df_5factor_aug = augment_data_noise(df_5factor,50,1,0)

# ae_input_testdata = df_3factor_aug.values
# ae_input_val_testdata = df_3factor.values
# ae_input_testdata = ae_input_testdata.clip(min=0)
# ae_input_val_testdata = ae_input_val_testdata.clip(min=0)
# input_dim_testdata = ae_input_testdata.shape[1]





df_3factor_01 = df_3factor.clip(lower=0) / df_3factor.max().max()
df_2factor_01 = df_2factor.clip(lower=0) / df_2factor.max().max()
df_4factor_01 = df_4factor.clip(lower=0) / df_4factor.max().max()
df_5factor_01 = df_5factor.clip(lower=0) / df_5factor.max().max()
df_all_data_01 = df_all_data.clip(lower=0) / df_all_data.max().max()
df_2factor_aug_01 = df_2factor_aug.clip(lower=0) / df_2factor_aug.max().max()
df_4factor_aug_01 = df_4factor_aug.clip(lower=0) / df_4factor_aug.max().max()
df_5factor_aug_01 = df_5factor_aug.clip(lower=0) / df_5factor_aug.max().max()
df_all_data_aug_01 = df_all_data_aug.clip(lower=0) / df_all_data_aug.max().max()


#%%Train an entirely new AE for the test data
#%%Work out how many epochs to train for
df_3factor_aug = augment_data_noise(df_3factor,50,1,0)

ae_input_testdata = df_3factor_aug.values
ae_input_val_testdata = df_3factor.values
ae_input_testdata = ae_input_testdata.clip(min=0)
ae_input_val_testdata = ae_input_val_testdata.clip(min=0)
input_dim_testdata = ae_input_testdata.shape[1]

#Based on the above, use an AE with 2 intermediate layers and latent dim of 20
ae_obj_testdata = NMFAE_n_layer(input_dim=input_dim,latent_dim=3,int_layers=1,latent_activation='relu',decoder_output_activation='relu')
history = ae_obj_testdata.fit_model(ae_input_testdata, x_test=ae_input_val_testdata,epochs=100)
val_acc_per_epoch = history.history['val_loss']
fig,ax = plt.subplots(1,figsize=(8,6))
ax.plot(val_acc_per_epoch)
plt.show()
#Now retrain model based on best epoch
best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
ae_obj_testdata = NMFAE_n_layer(input_dim=input_dim,latent_dim=3,int_layers=1,latent_activation='relu',decoder_output_activation='relu')
ae_obj_testdata.fit_model(ae_input_testdata, x_test=ae_input_val_testdata,epochs=best_epoch)
print('Best epoch: %d' % (best_epoch,))

latent_space = ae_obj_testdata.encoder(ae_input_val_testdata).numpy()
latent_space_noisy = np.multiply(latent_space,np.random.normal(1, 0.05, latent_space.shape))

#%%Test data in AE
fig,ax = plt.subplots(1)
plt.scatter(ae_input_val_testdata,ae_obj_testdata.ae(ae_input_val_testdata))
plt.title("AE input vs output")
plt.xlabel('AE input')
plt.ylabel('AE output')
plt.show()

#%%Try decoding each latent space column
lat0 = np.zeros(latent_space.shape)
lat1 = np.zeros(latent_space.shape)
lat2 = np.zeros(latent_space.shape)
#lat3 = np.zeros(latent_space.shape)

lat0[:,0] = latent_space[:,0]
lat1[:,1] = latent_space[:,1]
lat2[:,2] = latent_space[:,2]
#lat3[:,3] = latent_space[:,3]

lat0_decoded = ae_obj_testdata.decoder.predict(lat0)
lat1_decoded = ae_obj_testdata.decoder.predict(lat1)
lat2_decoded = ae_obj_testdata.decoder.predict(lat2)
#lat3_decoded = ae_obj_testdata.decoder.predict(lat3)

lat0_decoded_sum = lat0_decoded.sum(axis=1)
lat1_decoded_sum = lat1_decoded.sum(axis=1)
lat2_decoded_sum = lat2_decoded.sum(axis=1)
#lat3_decoded_sum = lat3_decoded.sum(axis=1)

#decoded_total_mtx = lat0_decoded + lat1_decoded + lat2_decoded# + lat3_decoded

plt.plot(lat0_decoded_sum,c='k')
plt.plot(lat1_decoded_sum,c='r')
plt.plot(lat2_decoded_sum,c='b')
#plt.plot(lat3_decoded_sum,c='y')

#%%
model = NMF(n_components=3)
W = model.fit_transform(latent_space)
H = model.components_

Factor0_lat_mtx = np.outer(W.T[0], H[0])
Factor1_lat_mtx = np.outer(W.T[1], H[1])
Factor2_lat_mtx = np.outer(W.T[2], H[2])
Factor0_mtx_decod = ae_obj_testdata.decoder.predict(Factor0_lat_mtx)
Factor1_mtx_decod = ae_obj_testdata.decoder.predict(Factor1_lat_mtx)
Factor2_mtx_decod = ae_obj_testdata.decoder.predict(Factor2_lat_mtx)

y_pred_factorsum = Factor0_mtx_decod + Factor1_mtx_decod + Factor2_mtx_decod

factor0_decoded_sum = Factor0_mtx_decod.sum(axis=1)
factor1_decoded_sum = Factor1_mtx_decod.sum(axis=1)
factor2_decoded_sum = Factor2_mtx_decod.sum(axis=1)

plt.plot(lat0_decoded_sum,c='k')
plt.plot(lat1_decoded_sum,c='r')
plt.plot(lat2_decoded_sum,c='b')





# W = ae_obj_testdata.nmf.transform(latent_space)
# H = ae_obj_testdata.nmf.components_

# Factor0_lat_mtx = np.outer(W.T[0], H[0])
# Factor1_lat_mtx = np.outer(W.T[1], H[1])
# Factor2_lat_mtx = np.outer(W.T[2], H[2])
# Factor0_mtx_decod = ae_obj_testdata.decoder.predict(Factor0_lat_mtx)
# Factor1_mtx_decod = ae_obj_testdata.decoder.predict(Factor1_lat_mtx)
# Factor2_mtx_decod = ae_obj_testdata.decoder.predict(Factor2_lat_mtx)

# y_pred_factorsum = Factor0_mtx_decod + Factor1_mtx_decod + Factor2_mtx_decod

# factor0_decoded_sum = Factor0_mtx_decod.sum(axis=1)
# factor1_decoded_sum = Factor1_mtx_decod.sum(axis=1)
# factor2_decoded_sum = Factor2_mtx_decod.sum(axis=1)

# plt.plot(lat0_decoded_sum,c='k')
# plt.plot(lat1_decoded_sum,c='r')
# plt.plot(lat2_decoded_sum,c='b')

#%%Try the FAE not the NMFAE

#%%Test with relu for latent space activation function

latent_dims = []
AE1_MSE_best50epoch =[]
AE2_MSE_best50epoch =[]
AE3_MSE_best50epoch =[]
AE4_MSE_best50epoch =[]
best_hps_array = []


verbose = 0

start_time = time.time()


for latent_dim in range(2,20):
    print(latent_dim)
    K.clear_session()
    latent_dims.append(latent_dim)
    
    #Test for 1 intermediate layer
    ae_obj = FAE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=1,latent_activation='linear',decoder_output_activation='relu')
    history = ae_obj.fit_model(ae_input_testdata, x_test=ae_input_val_testdata,epochs=50,verbose=verbose)
    val_acc_per_epoch = history.history['val_loss']
    AE1_MSE_best50epoch.append(min(history.history['val_loss']))
    
    #Test for 2 intermediate layers
    ae_obj = FAE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=2,latent_activation='linear',decoder_output_activation='relu')
    history = ae_obj.fit_model(ae_input_testdata, x_test=ae_input_val_testdata,epochs=50,verbose=verbose)
    val_acc_per_epoch = history.history['val_loss']
    AE2_MSE_best50epoch.append(min(history.history['val_loss']))
    
    #Test for 3 intermediate layers
    ae_obj = FAE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=3,latent_activation='linear',decoder_output_activation='relu')
    history = ae_obj.fit_model(ae_input_testdata, x_test=ae_input_val_testdata,epochs=50,verbose=verbose)
    val_acc_per_epoch = history.history['val_loss']
    AE3_MSE_best50epoch.append(min(history.history['val_loss']))
    
    # #Test for 4 intermediate layers
    # ae_obj = FAE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=4,latent_activation='linear',decoder_output_activation='relu')
    # history = ae_obj.fit_model(ae_input_testdata, x_test=ae_input_val_testdata,epochs=50,verbose=verbose)
    # val_acc_per_epoch = history.history['val_loss']
    # AE4_MSE_best50epoch.append(min(history.history['val_loss']))
    
print("--- %s seconds ---" % (time.time() - start_time))

#%%Plot the data for the different number of layers above
fig,ax=plt.subplots(2,1,figsize=(12,8))
ax[0].set_title('Finding optimum latent dims- FAE with linear latent space')
ax[0].plot(latent_dims,AE1_MSE_best50epoch,c='black')
ax[0].plot(latent_dims,AE2_MSE_best50epoch,c='red')
ax[0].plot(latent_dims,AE3_MSE_best50epoch,c='gray')
#ax[0].plot(latent_dims,AE4_MSE_best50epoch)
ax[0].set_xlabel('Number of latent dims')
ax[0].set_ylabel('Best MSE in first 50 epochs')
ax[1].plot(latent_dims,AE1_MSE_best50epoch,c='black')
ax[1].plot(latent_dims,AE2_MSE_best50epoch,c='red')
ax[1].plot(latent_dims,AE3_MSE_best50epoch,c='gray')
#ax[1].plot(latent_dims,AE4_MSE_best50epoch)
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
#%%
#Based on the above, use an AE with 2 intermediate layers and latent dim of 20
ae_obj_testdata = FAE_n_layer(input_dim=input_dim,latent_dim=3,int_layers=1,latent_activation='linear',decoder_output_activation='linear')
history = ae_obj_testdata.fit_model(ae_input_testdata, x_test=ae_input_val_testdata,epochs=200)
val_acc_per_epoch = history.history['val_loss']
# fig,ax = plt.subplots(1,figsize=(8,6))
# ax.plot(val_acc_per_epoch)
# plt.show()
#Now retrain model based on best epoch
best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
ae_obj_testdata = FAE_n_layer(input_dim=input_dim,latent_dim=3,int_layers=1,latent_activation='linear',decoder_output_activation='linear')
ae_obj_testdata.fit_model(ae_input_testdata, x_test=ae_input_val_testdata,epochs=best_epoch)
print('Best epoch: %d' % (best_epoch,))

latent_space = ae_obj_testdata.encoder(ae_input_val_testdata).numpy()
latent_space_noisy = np.multiply(latent_space,np.random.normal(1, 0.05, latent_space.shape))



#%%Test the zero penalty function
def fn_zeropen(maxmin):
    zeropen = 1-np.tanh(maxmin)
    return zeropen
    
test_x = np.arange(-10,10,dtype=float)
zeropen = fn_zeropen(test_x)
plt.scatter(test_x,zeropen)

#%%Test data in AE
fig,ax = plt.subplots(1)
plt.scatter(ae_input_val_testdata,ae_obj_testdata.ae(ae_input_val_testdata))
plt.title("AE input vs output")
plt.xlabel('AE input')
plt.ylabel('AE output')
plt.show()
#%%Try decoding each latent space column
lat0 = np.zeros(latent_space.shape)
lat1 = np.zeros(latent_space.shape)
lat2 = np.zeros(latent_space.shape)
#lat3 = np.zeros(latent_space.shape)

lat0[:,0] = latent_space[:,0]
lat1[:,1] = latent_space[:,1]
lat2[:,2] = latent_space[:,2]
#lat3[:,3] = latent_space[:,3]

lat0_decoded = ae_obj_testdata.decoder.predict(lat0)
lat1_decoded = ae_obj_testdata.decoder.predict(lat1)
lat2_decoded = ae_obj_testdata.decoder.predict(lat2)
#lat3_decoded = ae_obj_testdata.decoder.predict(lat3)

lat0_decoded_sum = lat0_decoded.sum(axis=1)
lat1_decoded_sum = lat1_decoded.sum(axis=1)
lat2_decoded_sum = lat2_decoded.sum(axis=1)

lat0_decoded_factor = lat0_decoded.sum(axis=0)
lat1_decoded_factor = lat1_decoded.sum(axis=0)
lat2_decoded_factor = lat2_decoded.sum(axis=0)

#decoded_total_mtx = lat0_decoded + lat1_decoded + lat2_decoded# + lat3_decoded

plt.plot(lat0_decoded_sum,c='k')
plt.plot(lat1_decoded_sum,c='r')
plt.plot(lat2_decoded_sum,c='b')
#plt.plot(lat3_decoded_sum,c='y')
#%%How does normal PMF do with this latent space data?
model = NMF(n_components=3)
W = model.fit_transform(latent_space)
H = model.components_

Factor0_mtx = np.outer(W.T[0], H[0])
Factor1_mtx = np.outer(W.T[1], H[1])
Factor2_mtx = np.outer(W.T[2], H[2])

Factor0_decod_mtx = ae_obj_testdata.decoder.predict(Factor0_mtx)
Factor1_decod_mtx = ae_obj_testdata.decoder.predict(Factor1_mtx)
Factor2_decod_mtx = ae_obj_testdata.decoder.predict(Factor2_mtx)

y_pred_factorsum = Factor0_decod_mtx + Factor1_decod_mtx + Factor2_decod_mtx

factor0_decod_sum = Factor0_decod_mtx.sum(axis=1)
factor1_decod_sum = Factor1_decod_mtx.sum(axis=1)
factor2_decod_sum = Factor2_decod_mtx.sum(axis=1)

plt.plot(factor0_decod_sum,c='k')
plt.plot(factor1_decod_sum,c='r')
plt.plot(factor2_decod_sum,c='b')

#%%How does normal PMF do with this real space testdata?
model = NMF(n_components=3)
W = model.fit_transform(ae_input_val_testdata)
H = model.components_

Factor0_mtx = np.outer(W.T[0], H[0])
Factor1_mtx = np.outer(W.T[1], H[1])
Factor2_mtx = np.outer(W.T[2], H[2])


y_pred_factorsum = Factor0_mtx + Factor1_mtx + Factor2_mtx

factor0_sum = Factor0_mtx.sum(axis=1)
factor1_sum = Factor1_mtx.sum(axis=1)
factor2_sum = Factor2_mtx.sum(axis=1)

plt.plot(factor0_sum,c='k')
plt.plot(factor1_sum,c='r')
plt.plot(factor2_sum,c='b')



#%%Try normal PMf with 4factor real space testdata

model = NMF(n_components=4)
W = model.fit_transform(df_4factor_01.to_numpy().clip(min=0))
#W = model.fit_transform(df_5factor_01.to_numpy().clip(min=0))
H = model.components_

Factor0_mtx = np.outer(W.T[0], H[0])
Factor1_mtx = np.outer(W.T[1], H[1])
Factor2_mtx = np.outer(W.T[2], H[2])
Factor3_mtx = np.outer(W.T[3], H[3])
#Factor4_mtx = np.outer(W.T[4], H[4])

#y_pred_factorsum = Factor0_mtx_decod + Factor1_mtx_decod + Factor2_mtx_decod + Factor3_mtx_decod

factor0_sum = Factor0_mtx.sum(axis=1)
factor1_sum = Factor1_mtx.sum(axis=1)
factor2_sum = Factor2_mtx.sum(axis=1)
factor3_sum = Factor3_mtx.sum(axis=1)
#factor4_sum = Factor4_mtx.sum(axis=1)

plt.plot(factor0_sum,c='k')
plt.plot(factor1_sum,c='r')
plt.plot(factor2_sum,c='b')
plt.plot(factor3_sum,c='y')
#plt.plot(factor4_sum,c='m')

#%%Plot 4-factor PMF fit to real space testdata

fig,ax = plt.subplots(5,1,figsize=(6.66,10))
ax[0].stem(mz_columns.to_numpy(),Factor0_mtx.mean(axis=0),markerfmt=' ')
ax[1].stem(mz_columns.to_numpy(),Factor1_mtx.mean(axis=0),markerfmt=' ')
ax[2].stem(mz_columns.to_numpy(),Factor2_mtx.mean(axis=0),markerfmt=' ')
ax[3].stem(mz_columns.to_numpy(),Factor3_mtx.mean(axis=0),markerfmt=' ')
ax[4].stem(mz_columns.to_numpy(),Factor4_mtx.mean(axis=0),markerfmt=' ')
plt.setp(ax, xlim=(100,500))
ax[0].set_title('PMF factors')
ax[4].set_xlabel('m/z')
plt.show()

#%%Plot 4-factor PMF fit to real space testdata, un MinMax

fig,ax = plt.subplots(4,1,figsize=(6.66,10))
ax[0].stem(mz_columns.to_numpy(),MM_4factor.inverse_transform(Factor0_mtx.mean(axis=0).reshape(-1,1).T).T,markerfmt=' ')
ax[1].stem(mz_columns.to_numpy(),MM_4factor.inverse_transform(Factor1_mtx.mean(axis=0).reshape(-1,1).T).T,markerfmt=' ')
ax[2].stem(mz_columns.to_numpy(),MM_4factor.inverse_transform(Factor2_mtx.mean(axis=0).reshape(-1,1).T).T,markerfmt=' ')
ax[3].stem(mz_columns.to_numpy(),MM_4factor.inverse_transform(Factor3_mtx.mean(axis=0).reshape(-1,1).T).T,markerfmt=' ')
plt.setp(ax, xlim=(100,400))
ax[0].set_title('PMF factors')
ax[3].set_xlabel('m/z')
plt.show()









#%%TRAIN 2-FACTOR VAE
#Based on the above, use an AE with 1 intermediate layers and latent dim of 4
ae_obj = FVAE_n_layer(input_dim=input_dim,latent_dim=1,int_layers=1,latent_activation='softsign',decoder_output_activation='relu')
history = ae_obj.fit_model(df_2factor_aug.to_numpy(),x_test=df_2factor_01.to_numpy(),epochs=100,verbose=True)
val_acc_per_epoch = history.history['val_loss']
kl_loss = history.history['kl_loss']
mse_loss = history.history['mse_loss']
# fig,ax = plt.subplots(1,figsize=(8,6))
# ax.plot(val_acc_per_epoch)
# plt.show()
# #Now retrain model based on best epoch
# best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
# ae_obj = FVAE_n_layer(input_dim=input_dim,latent_dim=4,int_layers=1,latent_activation='linear',decoder_output_activation='relu')
# ae_obj.fit_model(df_3factor_01.to_numpy(),epochs=best_epoch,verbose=True)
# print('Best epoch: %d' % (best_epoch,))


fig,ax=plt.subplots(2,1,figsize=(12,8))
ax[0].set_title('Finding optimum epochs- simple relu AE, relu latent space')
ax[0].plot(history.epoch,val_acc_per_epoch,label='loss',c='r')
ax[0].plot(history.epoch,kl_loss,c='k',label='kl_loss')
ax[0].plot(history.epoch,mse_loss,c='b',label='mse')
ax[0].set_xlabel('Number of epochs')
ax[0].set_ylabel('MSE')
ax[1].plot(history.epoch,val_acc_per_epoch,c='r')
ax[1].plot(history.epoch,kl_loss,c='k')
ax[1].plot(history.epoch,mse_loss,c='b')
ax[1].set_xlabel('Number of epochs')
ax[1].set_ylabel('MSE')
ax[1].set_yscale('log')
#loc = plticker.MultipleLocator(base=10.0) # this locator puts ticks at regular intervals
#ax[0].xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
#ax[0].xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
#ax[1].xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
#ax[1].xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
ax[0].legend()
plt.show()

#%%Plot input vs output
latent_space = ae_obj.encoder.predict(df_2factor_01.to_numpy())
decoded_latent_space = ae_obj.decoder.predict(latent_space)

fig,ax = plt.subplots(1)
plt.scatter(df_2factor_01.to_numpy(),ae_obj.vae.predict(df_2factor_01.to_numpy()))
plt.title("AE input vs output")
plt.xlabel('AE input')
plt.ylabel('AE output')
plt.show()

#%%Decode single column latent space
latent_decoded_min1 = ae_obj.decoder.predict([-1]).T
latent_decoded_plus1 = ae_obj.decoder.predict([1]).T

mz_columns = pd.DataFrame(df_all_raw['Molecular Weight'].loc[df_all_data.columns])
fig,ax = plt.subplots(4,1,figsize=(10,10))
ax[0].stem(mz_columns.to_numpy(),factor_A,markerfmt=' ')
ax[1].stem(mz_columns.to_numpy(),factor_C,markerfmt=' ')
ax[2].stem(mz_columns.to_numpy(),latent_decoded_min1,markerfmt=' ')
ax[3].stem(mz_columns.to_numpy(),latent_decoded_plus1,markerfmt=' ')
ax[0].set_xlim(right=400)
ax[1].set_xlim(right=400)
ax[2].set_xlim(right=400)
ax[3].set_xlim(right=400)
plt.show()



#%%Try prepare data with subtract the mean (of the whole thing, then divide by the stdev (of the whole thing))
#Then scale to between 0 and 1?
orig_mean_num = df_4factor.mean().mean()
df_4factor_subm = df_4factor - df_4factor.mean().mean()



#%%TRAIN 4-FACTOR VAE
#Based on the above, use an AE with 1 intermediate layers and latent dim of 4
ae_obj = FVAE_n_layer(input_dim=input_dim,latent_dim=2,int_layers=3,latent_activation='softsign',decoder_output_activation='sigmoid')
history = ae_obj.fit_model(df_4factor_aug_01.to_numpy(),x_test=df_4factor_01.to_numpy(),epochs=100,verbose=True)
val_acc_per_epoch = history.history['val_loss']
kl_loss = history.history['kl_loss']
mse_loss = history.history['mse_loss']
# fig,ax = plt.subplots(1,figsize=(8,6))
# ax.plot(val_acc_per_epoch)
# plt.show()
# #Now retrain model based on best epoch
# best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
# ae_obj = FVAE_n_layer(input_dim=input_dim,latent_dim=4,int_layers=1,latent_activation='linear',decoder_output_activation='relu')
# ae_obj.fit_model(df_3factor_01.to_numpy(),epochs=best_epoch,verbose=True)
# print('Best epoch: %d' % (best_epoch,))


fig,ax=plt.subplots(2,1,figsize=(12,8))
ax[0].set_title('Finding optimum epochs- simple relu AE, relu latent space')
ax[0].plot(history.epoch,val_acc_per_epoch,label='loss',c='r')
ax[0].plot(history.epoch,kl_loss,c='k',label='kl_loss')
ax[0].plot(history.epoch,mse_loss,c='b',label='mse')
ax[0].set_xlabel('Number of epochs')
ax[0].set_ylabel('MSE')
ax[1].plot(history.epoch,val_acc_per_epoch,c='r')
ax[1].plot(history.epoch,kl_loss,c='k')
ax[1].plot(history.epoch,mse_loss,c='b')
ax[1].set_xlabel('Number of epochs')
ax[1].set_ylabel('MSE')
ax[1].set_yscale('log')
#loc = plticker.MultipleLocator(base=10.0) # this locator puts ticks at regular intervals
#ax[0].xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
#ax[0].xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
#ax[1].xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
#ax[1].xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
ax[0].legend()
plt.show()

# SUBTRACT THE MEAN FROM THE INITIAL DATA AND SCALE IT 0 TO 1??
# OR TRY THE THING WITH MINMAXSCALER AND USE A SOFTMAX OUTPUT ACTIVATION

# FIRST TRY JUST A DIFFERNT TIME SERIES OF THE FACTORS, FIRST YOU ARE TRYING TO JUST PICK OUT THE FACTORS
# TO WORK OUT HOW IT WORKS
# THEN TRY ASSESS THE PERFORMANCE AND IMPROVE

#%%Plot input vs output
latent_space = ae_obj.encoder.predict(df_4factor_01.to_numpy())
decoded_latent_space = ae_obj.decoder.predict(latent_space)

fig,ax = plt.subplots(1)
plt.scatter(df_4factor_01.to_numpy(),ae_obj.ae.predict(df_4factor_01.to_numpy()))
plt.title("AE input vs output")
plt.xlabel('AE input')
plt.ylabel('AE output')
plt.show()


#%%Check Pearson's R correlation between input and output factors
latent_grid = np.mgrid[-1:2:1, -1:2:1].reshape(2, -1).T
df_latent_grid_decoded = pd.DataFrame(ae_obj.decoder.predict(latent_grid),columns=df_all_data.columns)
latent_grid_decoded_corr = corr_coeff_rowwise_loops(df_latent_grid_decoded.to_numpy(),df_4factor_factors.to_numpy())
latent_grid_decoded_corr = np.around(latent_grid_decoded_corr,3)

mz_columns = pd.DataFrame(df_all_raw['Molecular Weight'].loc[df_all_data.columns])
fig,ax = plt.subplots(3,3,figsize=(20,10))
axs = ax.ravel()
    
for i in range(9):   
    axs[i].stem(mz_columns.to_numpy(),df_latent_grid_decoded.iloc[i],markerfmt=' ')
    best_R = latent_grid_decoded_corr[i]
    axs[i].text(0.95,0.5,latent_grid_decoded_corr[i].max(),horizontalalignment='right', verticalalignment='top',transform=axs[i].transAxes)
    axs[i].text(0.95,0.4,latent_grid_decoded_corr[i].argmax(),horizontalalignment='right', verticalalignment='top',transform=axs[i].transAxes)
    

plt.setp(ax, xlim=(100,400))
plt.tight_layout()
plt.show()






#%%Decode single column latent space
latent0_decoded_min1 = ae_obj.decoder.predict(np.array([[-1,0]])).T
latent0_decoded_plus1 = ae_obj.decoder.predict(np.array([[1,0]])).T
latent1_decoded_min1 = ae_obj.decoder.predict(np.array([[0,-1]])).T
latent1_decoded_plus1 = ae_obj.decoder.predict(np.array([[0,1]])).T

mz_columns = pd.DataFrame(df_all_raw['Molecular Weight'].loc[df_all_data.columns])
fig,ax = plt.subplots(8,1,figsize=(10,10))
ax[0].stem(mz_columns.to_numpy(),factor_A,markerfmt=' ')
ax[1].stem(mz_columns.to_numpy(),factor_B,markerfmt=' ')
ax[2].stem(mz_columns.to_numpy(),factor_C,markerfmt=' ')
ax[3].stem(mz_columns.to_numpy(),factor_D,markerfmt=' ')
ax[4].stem(mz_columns.to_numpy(),latent0_decoded_min1,markerfmt=' ')
ax[5].stem(mz_columns.to_numpy(),latent0_decoded_plus1,markerfmt=' ')
ax[6].stem(mz_columns.to_numpy(),latent1_decoded_min1,markerfmt=' ')
ax[7].stem(mz_columns.to_numpy(),latent1_decoded_plus1,markerfmt=' ')
ax[0].set_xlim(right=400)
ax[1].set_xlim(right=400)
ax[2].set_xlim(right=400)
ax[3].set_xlim(right=400)
ax[4].set_xlim(right=400)
ax[5].set_xlim(right=400)
ax[6].set_xlim(right=400)
ax[7].set_xlim(right=400)
plt.show()


#%%Decode corners from latent space
latent_mm_decoded = ae_obj.decoder.predict(np.array([[-1,-1]])).T
latent_mp_decoded = ae_obj.decoder.predict(np.array([[-1,1]])).T
latent_pm_decoded = ae_obj.decoder.predict(np.array([[1,-1]])).T
latent_pp_decoded = ae_obj.decoder.predict(np.array([[1,1]])).T
latent_00_decoded = ae_obj.decoder.predict(np.array([[0,0]])).T

mz_columns = pd.DataFrame(df_all_raw['Molecular Weight'].loc[df_all_data.columns])
fig,ax = plt.subplots(9,1,figsize=(10,10))
ax[0].stem(mz_columns.to_numpy(),factor_A,markerfmt=' ')
ax[1].stem(mz_columns.to_numpy(),factor_B,markerfmt=' ')
ax[2].stem(mz_columns.to_numpy(),factor_C,markerfmt=' ')
ax[3].stem(mz_columns.to_numpy(),factor_D,markerfmt=' ')
ax[4].stem(mz_columns.to_numpy(),latent_mm_decoded,markerfmt=' ')
ax[5].stem(mz_columns.to_numpy(),latent_mp_decoded,markerfmt=' ')
ax[6].stem(mz_columns.to_numpy(),latent_pm_decoded,markerfmt=' ')
ax[7].stem(mz_columns.to_numpy(),latent_pp_decoded,markerfmt=' ')
ax[8].stem(mz_columns.to_numpy(),latent_00_decoded,markerfmt=' ')
ax[0].set_xlim(right=400)
ax[1].set_xlim(right=400)
ax[2].set_xlim(right=400)
ax[3].set_xlim(right=400)
ax[4].set_xlim(right=400)
ax[5].set_xlim(right=400)
ax[6].set_xlim(right=400)
ax[7].set_xlim(right=400)
ax[8].set_xlim(right=400)
plt.show()


#%%Now try subtracting the mean from the input and see how that affects things
alpha = df_4factor.mean().mean()

df_4factor_minmen = df_4factor - df_4factor.mean().mean()
df_4factor_minmen_aug = augment_data_noise(df_4factor_minmen,50,1,0)
beta = df_4factor_minmen.min().min()
df_4factor_minmen_01 = (df_4factor_minmen - df_4factor_minmen.min().min())
gamma = df_4factor_minmen_01.max().max()
df_4factor_minmen_01 = df_4factor_minmen_01 / df_4factor_minmen_01.max().max()
df_4factor_minmen_01 = df_4factor_minmen_01*2 - 1

df_4factor_minmen_aug_01 = (df_4factor_minmen_aug - df_4factor_minmen_aug.min().min())
df_4factor_minmen_aug_01 = df_4factor_minmen_aug_01 / df_4factor_minmen_aug_01.max().max()
df_4factor_minmen_aug_01 = df_4factor_minmen_aug_01*2 -1

MM_4factor_aug = MinMaxScaler()
MM_4factor = MinMaxScaler()
df_4factor_MM_aug = pd.DataFrame(MM_4factor_aug.fit_transform(df_4factor_aug.to_numpy()),columns=df_all_data.columns)
df_4factor_MM = pd.DataFrame(MM_4factor.fit_transform(df_4factor.to_numpy()),columns=df_all_data.columns)


QT_4factor_aug = QuantileTransformer()
QT_4factor = QuantileTransformer()
df_4factor_QT_aug = pd.DataFrame(QT_4factor_aug.fit_transform(df_4factor_aug.to_numpy()),columns=df_all_data.columns)
df_4factor_QT = pd.DataFrame(QT_4factor.fit_transform(df_4factor.to_numpy()),columns=df_all_data.columns)

# df_4factor_QT_aug = df_4factor_QT_aug*2 - 1
# df_4factor_QT = df_4factor_QT*2 - 1


from sklearn.preprocessing import PowerTransformer
PT_4factor = PowerTransformer()
PT_4factor_aug = PowerTransformer()
df_4factor_PT_aug = pd.DataFrame(PT_4factor_aug.fit_transform(df_4factor_aug.to_numpy()),columns=df_all_data.columns)
df_4factor_PT = pd.DataFrame(PT_4factor.fit_transform(df_4factor.to_numpy()),columns=df_all_data.columns)


#%%TRAIN 4-FACTOR VAE MINMAX
#Based on the above, use an AE with 1 intermediate layers and latent dim of 4
ae_obj = FVAE_n_layer(input_dim=input_dim,latent_dim=2,int_layers=3,latent_activation='softsign',decoder_output_activation='linear')

#ae_obj = testAE_n_layer(input_dim=input_dim,latent_dim=4,int_layers=1,latent_activation='sigmoid')


#history = ae_obj.fit_model(df_4factor_minmen_aug_01.to_numpy(),x_test=df_4factor_minmen_01.to_numpy(),epochs=100,verbose=True)
#history = ae_obj.fit_model(df_4factor_aug_01.to_numpy(),x_test=df_4factor_01.to_numpy(),epochs=2,verbose=True)
history = ae_obj.fit_model(df_all_data_aug_01.to_numpy(),x_test=df_all_data_01.to_numpy(),epochs=50,verbose=True)
val_acc_per_epoch = history.history['val_loss']
kl_loss = history.history['kl_loss']
mse_loss = history.history['mse_loss']
# fig,ax = plt.subplots(1,figsize=(8,6))
# ax.plot(val_acc_per_epoch)
# plt.show()
# #Now retrain model based on best epoch
# best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
# ae_obj = FVAE_n_layer(input_dim=input_dim,latent_dim=4,int_layers=1,latent_activation='linear',decoder_output_activation='relu')
# ae_obj.fit_model(df_3factor_01.to_numpy(),epochs=best_epoch,verbose=True)
# print('Best epoch: %d' % (best_epoch,))


fig,ax=plt.subplots(2,1,figsize=(12,8))
ax[0].set_title('Finding optimum epochs- simple relu AE, relu latent space')
ax[0].plot(history.epoch,val_acc_per_epoch,label='loss',c='r')
ax[0].plot(history.epoch,kl_loss,c='k',label='kl_loss')
ax[0].plot(history.epoch,mse_loss,c='b',label='mse')
ax[0].set_xlabel('Number of epochs')
ax[0].set_ylabel('MSE')
ax[1].plot(history.epoch,val_acc_per_epoch,c='r')
ax[1].plot(history.epoch,kl_loss,c='k')
ax[1].plot(history.epoch,mse_loss,c='b')
ax[1].set_xlabel('Number of epochs')
ax[1].set_ylabel('MSE')
ax[1].set_yscale('log')
#loc = plticker.MultipleLocator(base=10.0) # this locator puts ticks at regular intervals
#ax[0].xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
#ax[0].xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
#ax[1].xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
#ax[1].xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
ax[0].legend()
plt.show()

#HERE I AM TRYING TO MAKE A FACTOR TO REDUCE THE CORRELATION BETWEEN THE LATENT SPACE TIME SERIES

#%%Plot input vs output df_4factor_MM
#latent_space = ae_obj.encoder.predict(df_4factor_01.to_numpy())
latent_space = ae_obj.encoder.predict(df_all_data_01.to_numpy())
latent_grid = np.mgrid[latent_space[:,0].min():latent_space[:,0].max():3j, latent_space[:,1].min():latent_space[:,1].max():3j].reshape(2, -1).T
df_latent_grid = pd.DataFrame(latent_grid).sort_values([1, 0], ascending=[False, True])
latent_grid = df_latent_grid.to_numpy()

fig,ax = plt.subplots(1)
#plt.scatter(df_4factor_01.to_numpy(),ae_obj.ae.predict(df_4factor_01.to_numpy()))
plt.scatter(df_all_data_01.to_numpy(),ae_obj.ae.predict(df_all_data_01.to_numpy()))
plt.title("AE input vs output")
plt.xlabel('AE input')
plt.ylabel('AE output')
plt.show()

#%%Evaluate AE loss per sample
ds_AE_loss_per_sample = pd.Series(AE_calc_loss_per_sample(ae_obj.ae,df_4factor_01.to_numpy(),df_4factor_01.to_numpy()))


#%%Plot latent space time series on the grid
fig,ax = plt.subplots(1,figsize=(8,6))
# plt.xlim([-1,1])
# plt.ylim([-1,1])
c = np.arange(latent_space.shape[0])
c_timeofday = df_all_data.index.hour
c_dataset = dataset_cat.codes

#sns.scatterplot(ax=ax,x=latent_space[:,0], y=latent_space[:,1],c=c,style=dataset_index_4factor)
ax.set_title('2-D latent space')
plt.xlabel('Latent dimension 0')
plt.ylabel('Latent dimension 1')
#sns.scatterplot(x=latent_space[:,0], y=latent_space[:,1],c=c,style=dataset_cat)
plot = plt.scatter(x=latent_space[:,0], y=latent_space[:,1],c=c_dataset,cmap=plt.cm.get_cmap('tab20b', 4))
cbar = plt.colorbar(plot,label="Dataset")
plt.clim(0, 3)
cbar.ax.set_yticklabels(['Beijing winter', '', 'Beijing summer', '', 'Delhi summer', 'Delhi autumn'])
sns.scatterplot(ax=ax,x=latent_space[:,0], y=latent_space[:,1],c=c_dataset,style=dataset_cat,cmap=plt.cm.get_cmap('tab20b', 4))
ax.plot()

#plt.imshow(I, cmap=plt.cm.get_cmap('Blues', 6))
#plt.colorbar()



#plt.scatter(latent_grid[:,0],latent_grid[:,1])

#%%
fig,ax = plt.subplots(2,1,figsize=(8,5))
ax = ax.ravel()
sns.scatterplot(ax=ax[0],x=c,y=latent_space[:,0],markers='s',c=c_dataset,cmap=plt.cm.get_cmap('tab20b', 4))
sns.scatterplot(ax=ax[1],x=c,y=latent_space[:,1],markers='o',c=c_dataset,cmap=plt.cm.get_cmap('tab20b', 4))
ax[0].set_ylabel('Latent dimension 0')
ax[1].set_ylabel('Latent dimension 1')
ax[1].set_xlabel('~Time (arb units)')
#ax.xlabel('~Time (arb units)')
#ax[0].legend()#(['Beijing winter','Beijing summer','Delhi summer','Delhi autumn','a','b'])


#%%Check Pearson's R correlation between input and output factors
#latent_grid = np.mgrid[-1:2:1, -1:2:1].reshape(2, -1).T

#df_latent_grid_decoded = pd.DataFrame(QT_4factor.inverse_transform((ae_obj.decoder.predict(latent_grid)+1)/2),columns=df_all_data.columns)
#df_latent_grid_decoded = pd.DataFrame(PT_4factor_aug.inverse_transform(ae_obj.decoder.predict(latent_grid)),columns=df_all_data.columns)
df_latent_grid_decoded = pd.DataFrame(ae_obj.decoder.predict(latent_grid),columns=df_all_data.columns)
#df_latent_grid_decoded = df_latent_grid_decoded.fillna(0)
latent_grid_decoded_corr = corr_coeff_rowwise_loops(df_latent_grid_decoded.to_numpy(),df_4factor_factors.to_numpy())
latent_grid_decoded_corr = np.around(latent_grid_decoded_corr,3)

mz_columns = pd.DataFrame(df_all_raw['Molecular Weight'].loc[df_all_data.columns])
fig,ax = plt.subplots(3,3,figsize=(20,10))
axs = ax.ravel()
    
for i in range(9):   
    axs[i].stem(mz_columns.to_numpy(),df_latent_grid_decoded.iloc[i],markerfmt=' ')
    best_R = latent_grid_decoded_corr[i]
    axs[i].text(0.95,0.5,'R = ' + str(latent_grid_decoded_corr[i].max()),horizontalalignment='right', verticalalignment='top',transform=axs[i].transAxes,fontsize=14)
    axs[i].text(0.95,0.4,'Factor ' + str(latent_grid_decoded_corr[i].argmax()),horizontalalignment='right', verticalalignment='top',transform=axs[i].transAxes,fontsize=14)
    

plt.setp(ax, xlim=(100,400))
plt.tight_layout()
plt.show()

#%%Find the best R for the decoded data time series
latent_space_decoded_corr = corr_coeff_rowwise_loops(ae_obj.ae.predict(df_4factor_01.to_numpy()),df_4factor_factors.to_numpy())
latent_space_decoded_corr = np.around(latent_space_decoded_corr,3)
latent_space_decoded_best_corr_id = latent_space_decoded_corr.argmax(axis=1)


input_data_corr = corr_coeff_rowwise_loops(df_4factor_01.to_numpy(),df_4factor_factors.to_numpy())
input_data_corr = np.around(input_data_corr,3)
input_data_best_corr_id = input_data_corr.argmax(axis=1)






#%%Check Pearson's R correlation between input and output factors
latent_grid = np.mgrid[-1:3:1, -1:3:1].reshape(2, -1).T
#df_latent_grid_decoded = pd.DataFrame(QT_4factor.inverse_transform((ae_obj.decoder.predict(latent_grid)+1)/2),columns=df_all_data.columns)
#df_latent_grid_decoded = pd.DataFrame(PT_4factor_aug.inverse_transform(ae_obj.decoder.predict(latent_grid)),columns=df_all_data.columns)
df_latent_grid_decoded = pd.DataFrame(ae_obj.decoder.predict(latent_grid),columns=df_all_data.columns)
#df_latent_grid_decoded = df_latent_grid_decoded.fillna(0)
latent_grid_decoded_corr = corr_coeff_rowwise_loops(df_latent_grid_decoded.to_numpy(),df_4factor_factors.to_numpy())
latent_grid_decoded_corr = np.around(latent_grid_decoded_corr,3)

mz_columns = pd.DataFrame(df_all_raw['Molecular Weight'].loc[df_all_data.columns])
fig,ax = plt.subplots(4,4,figsize=(20,10))
axs = ax.ravel()
    
for i in range(16):   
    axs[i].stem(mz_columns.to_numpy(),df_latent_grid_decoded.iloc[i],markerfmt=' ')
    best_R = latent_grid_decoded_corr[i]
    axs[i].text(0.95,0.5,latent_grid_decoded_corr[i].max(),horizontalalignment='right', verticalalignment='top',transform=axs[i].transAxes)
    axs[i].text(0.95,0.4,latent_grid_decoded_corr[i].argmax(),horizontalalignment='right', verticalalignment='top',transform=axs[i].transAxes)
    

plt.setp(ax, xlim=(100,500))
plt.tight_layout()
plt.show()


#%%Check Pearson's R correlation between input and output factors
latent_grid = np.mgrid[latent_space[:,0].min():latent_space[:,0].max():10j, latent_space[:,1].min():latent_space[:,1].max():10j].reshape(2, -1).T
df_latent_grid = pd.DataFrame(latent_grid).sort_values([1, 0], ascending=[False, True])
latent_grid = df_latent_grid.to_numpy()
#df_latent_grid_decoded = pd.DataFrame(QT_4factor.inverse_transform((ae_obj.decoder.predict(latent_grid)+1)/2),columns=df_all_data.columns)
#df_latent_grid_decoded = pd.DataFrame(PT_4factor_aug.inverse_transform(ae_obj.decoder.predict(latent_grid)),columns=df_all_data.columns)
df_latent_grid_decoded = pd.DataFrame(ae_obj.decoder.predict(latent_grid),columns=df_all_data.columns)
#df_latent_grid_decoded = df_latent_grid_decoded.fillna(0)
latent_grid_decoded_corr = corr_coeff_rowwise_loops(df_latent_grid_decoded.to_numpy(),df_4factor_factors.to_numpy())
latent_grid_decoded_corr = np.around(latent_grid_decoded_corr,3)

mz_columns = pd.DataFrame(df_all_raw['Molecular Weight'].loc[df_all_data.columns])
fig,ax = plt.subplots(10,10,figsize=(20,10))
axs = ax.ravel()
    
for i in range(100):   
    axs[i].stem(mz_columns.to_numpy(),df_latent_grid_decoded.iloc[i],markerfmt=' ')
    best_R = latent_grid_decoded_corr[i]
    axs[i].text(0.95,0.5,latent_grid_decoded_corr[i].max(),horizontalalignment='right', verticalalignment='top',transform=axs[i].transAxes)
    axs[i].text(0.95,0.4,latent_grid_decoded_corr[i].argmax(),horizontalalignment='right', verticalalignment='top',transform=axs[i].transAxes)
    

plt.setp(ax, xlim=(100,500))
plt.tight_layout()
plt.show()

#%%Plot input factors
mz_columns = pd.DataFrame(df_all_raw['Molecular Weight'].loc[df_all_data.columns])
#fig,ax = plt.subplots(4,1,figsize=(6.66,10))
fig,ax = plt.subplots(2,2,figsize=(10,6))
ax = ax.ravel()
ax[0].stem(mz_columns.to_numpy(),factor_A,markerfmt=' ')
ax[1].stem(mz_columns.to_numpy(),factor_B,markerfmt=' ')
ax[2].stem(mz_columns.to_numpy(),factor_C,markerfmt=' ')
ax[3].stem(mz_columns.to_numpy(),factor_D,markerfmt=' ')
ax[0].text(0.95,0.95,'Factor 0',horizontalalignment='right', verticalalignment='top',transform=ax[0].transAxes)
ax[1].text(0.95,0.95,'Factor 1',horizontalalignment='right', verticalalignment='top',transform=ax[1].transAxes)
ax[2].text(0.95,0.95,'Factor 2',horizontalalignment='right', verticalalignment='top',transform=ax[2].transAxes)
ax[3].text(0.95,0.95,'Factor 3',horizontalalignment='right', verticalalignment='top',transform=ax[3].transAxes)
plt.setp(ax, xlim=(100,400))
ax[0].set_title('Input factors')
ax[3].set_xlabel('m/z')
plt.tight_layout()
plt.show()


#%%Some quick clustering

#%%

So, I think I possibly have this VAE in a state where it MIGHT be useful. Might
Need to check some things like what the factor profile look like, if you check one factor as -1 versus 1, etc etc
They should hopefully be independent...if not need to vary beta

What should the activation function be for the decoder? Should really be sigmoid shouldnt it?
Not relu? You end upw ith columns that are not used
So one way to do it would be to use minmaxscaler instead


THEY ARE DIFFERENT
LAT0 AND LAT1 DECODED USE DIFFERENT COLUMNS
When they go from -1 to 1, is that like 2 factors? Each end is a different one???
Not really but sortof, it's differnent to PMF isnt it'

If this gets to a position of it working, compare each end of each factor to things like the O/C, H/C, N fraction, S
Find samples with high and low numbers for each dimension and take AQ averages, and see what they look like
These high and low numbers will be in the latent space
