# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 14:24:32 2021

@author: mbcx5jt5
"""

import numpy as np
np.random.seed(1337) # for reproducibility
import tensorflow as tf
tf.random.set_seed(69)

#Keras from tensorflow
# import tensorflow.keras as keras
# from tensorflow.keras import layers
# from tensorflow.keras import backend as K 

#Keras as keras
import keras as keras
from keras import layers
from keras import backend as K
from keras import metrics


import kerastuner as kt
#from google.colab import drive
import pandas as pd
import glob
import pdb
from sklearn.preprocessing import RobustScaler, StandardScaler,FunctionTransformer
from sklearn.pipeline import Pipeline
#import tensorflow as tf
#import keras
#from tensorflow import keras

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib
import os


path='C:/Users/mbcx5jt5/Google Drive/Shared_York_Man2/'


#1 Load in filter metadata
df_filter_metadata = pd.read_excel(path + 'BJ_UnAmbNeg9.1.1_20210505-Times_Fixed.xlsx',engine='openpyxl',
                                   sheet_name='massloading_Beijing',usecols='A:K',nrows=329,converters={'mid_datetime': str})
#mid_datetime = pd.read_excel('/content/gdrive/MyDrive/Data_YRK_MAN/BJ_UnAmbNeg9.1.1_20210505.xlsx',engine='openpyxl',
                                 #  sheet_name='massloading_Beijing',usecols='E',nrows=329,dtype='str')
#df_filter_metadata["DateTime"] =pd.to_datetime(df_filter_metadata["mid_datetime"])
#df_filter_metadata.set_index('DateTime',inplace=True)
#df_filter_metadata.set_index('Sample.ID',inplace=True)
#2 set index to time
#3 Set that index to the time of the filter data
#4 Normalise filter data by sample volume
#5 AE_input does not contain any column other than the peaks
df_filter_metadata['Sample.ID'] = df_filter_metadata['Sample.ID'].astype(str)
#df_filter_metadata

df_beijing_peaks = pd.read_excel(path + 'BJ_UnAmbNeg9.1.1_20210505-Times_Fixed.xlsx',engine='openpyxl',sheet_name='Compounds')
#df_beijing_peaks


#Line up everything by the sample ID
df_beijing_filters = df_beijing_peaks.iloc[:,list(range(4,321))].copy()
sample_id = df_beijing_filters.columns.str.split('_|.raw').str[2]
#sample_id = sample_id.rename("Sample.ID")
#df_beijing_filters.columns = sample_id
df_beijing_filters = df_beijing_filters.transpose()
df_beijing_filters["Sample.ID"] = sample_id
#df_beijing_filters.set_index("Sample.ID",inplace=True)
df_beijing_filters.columns.rename("compound_num",inplace=True)
df_beijing_filters.columns = df_beijing_filters.columns.astype('str')

#Check for any negative data or NaNs
df_beijing_filters.describe()

#No NaNs
df_beijing_filters.isna().sum().sum()



#Add on the metadata
df_beijing_merged = pd.merge(df_beijing_filters,df_filter_metadata,on="Sample.ID",how='inner')
#Divide the data columns by the sample volume and multiply by the dilution liquid volume (pinonic acid)
df_beijing_filters = df_beijing_merged.iloc[:,0:3783].div(df_beijing_merged['Volume_m3'], axis=0).mul(df_beijing_merged['Dilution_mL'], axis=0)
df_beijing_filters.columns.rename("compound_num",inplace=True)

#JUST THE WINTER DATA
df_beijing_filters = df_beijing_filters.iloc[0:124].copy()



df_beijing_filters['mid_datetime'] = pd.to_datetime(df_beijing_merged['mid_datetime'],yearfirst=True)
df_beijing_filters.set_index('mid_datetime',inplace=True)
#df_beijing_filters



#Some basic checks like the time series of total concentration
beijing_rows_sum = df_beijing_filters.sum(axis=1)
plt.plot(beijing_rows_sum)
plt.title("Total sum of all peaks")
plt.show()


#How many data points?
num_filters = int(df_beijing_filters.shape[0])
num_cols = int(df_beijing_filters.shape[1])


fig = plt.plot(figsize=(30,20))
#plt.pcolormesh(df_beijing_peaks.iloc[:,list(range(4,321))],norm=matplotlib.colors.LogNorm())
plt.pcolormesh(df_beijing_filters.astype(float),norm=matplotlib.colors.LogNorm())
plt.show()
#plt.sjhow()

# %%


#Let's make some fake data with 3 well defined clusters to see if we can pick them apart
max_filter = df_beijing_filters.loc[beijing_rows_sum.idxmax()].ravel()
min_filter = df_beijing_filters.loc[beijing_rows_sum.idxmin()].ravel()

#Now make a pandas frame cointaining just these two, but with added 5% noise
clust0 = pd.DataFrame((np.random.normal(0, 0.05, [int(np.ceil(num_filters/2)),num_cols])+1) * min_filter)
clust1 = pd.DataFrame((np.random.normal(0, 0.05, [int(num_filters/2),num_cols])+1) * max_filter)
df_2clust = clust0.append(clust1)
df_2clust.columns = df_beijing_filters.columns
df_2clust.head()

thousand_filter = np.random.normal(1000, 50, num_cols)
clust2 = pd.DataFrame((np.random.normal(0, 0.05, [int(np.ceil(num_filters/2)),num_cols])+1) * thousand_filter )
df_3clust = clust0.append(clust1.append(clust2))
df_3clust.columns = df_beijing_filters.columns


# %%
# #Let's make 3 factors that vary with time
#Visually these all look a bit different on the time series matrix plot
Cluster_A = df_beijing_filters.iloc[0].ravel()
Cluster_B = df_beijing_filters.iloc[50].ravel()
Cluster_C = df_beijing_filters.iloc[100].ravel()

#Normalise all to 1e6
Cluster_A = 1e6 * Cluster_A / Cluster_A.sum()
Cluster_B = 1e6 * Cluster_B / Cluster_B.sum()
Cluster_C = 1e6 * Cluster_C / Cluster_C.sum()


#For just the winter data, when it's normalised
amp_A = np.append(np.zeros(50),np.arange(0,1,0.01)) * 5
amp_B = np.ones(150)*2.75
amp_C = np.append(np.arange(1,0,-0.01),np.zeros(50)) * 5


df_clustA = pd.DataFrame((np.random.normal(1, 0.01, [150,num_cols])) * Cluster_A).multiply(amp_A,axis=0)
df_clustB = pd.DataFrame((np.random.normal(1, 0.01, [150,num_cols])) * Cluster_B).multiply(amp_B,axis=0)
df_clustC = pd.DataFrame((np.random.normal(1, 0.01, [150,num_cols])) * Cluster_C).multiply(amp_C,axis=0)
df_clustA.columns = df_beijing_filters.columns
df_clustB.columns = df_beijing_filters.columns
df_clustC.columns = df_beijing_filters.columns

clustA_total = df_clustA.sum(axis=1)
clustB_total = df_clustB.sum(axis=1)
clustC_total = df_clustC.sum(axis=1)


df_3clust = df_clustA + df_clustB + df_clustC

plt.plot(clustA_total)
plt.plot(clustB_total)
plt.plot(clustC_total)
plt.show()

# %%

#Augment your data by making many copies of each row, with some added noise percentage
#The sig_noise_pct is how much the peaks each vary relative to each other
#The t_noise_pct is how much they vary with time
#Currently this does not include an unmodified version of the data, it's all with added noise
def augment_data_noise(df,num_copies,sig_noise_pct,t_noise_pct):
    num_rows = df.shape[0]
    num_cols = df.shape[1]
       
    # #Explicit version
    newdf = pd.DataFrame(np.repeat(df.values,num_copies,axis=0))
    newdf.columns = df.columns
    
    
    timenoise = np.random.normal(1, t_noise_pct/100, num_copies*num_rows)
    # timenoise=1
    # #timenoise[0] = 1    #Make it so the first one is just the standard copy
        
    newdf = newdf.multiply(timenoise,axis=0)
    
    # signoise = np.random.normal(1, sig_noise_pct/100, [num_copies*num_rows,num_cols])
    
    # newdf = newdf * signoise
       


    #Efficient version
    #newdf = pd.DataFrame(np.repeat(df.values,num_copies,axis=0)) * np.random.normal(1, sig_noise_pct/100, [num_copies*num_rows,num_cols])
    return newdf
# %%

# df_onerow_test = pd.DataFrame(df_beijing_filters.iloc[0],index=df_beijing_filters.columns).T

# timenoise = np.random.normal(10, 5/100, 1)
# signoise = np.random.normal(1, 5/100, [1,3783])

#df_thisrow = pd.DataFrame(signoise * df_onerow_test).multiply(timenoise,axis=0)



# newdf = pd.DataFrame(np.repeat(df_3clust.values,2,axis=0))
# newdf.columns = df_3clust.columns
    
    
# timenoise = np.random.normal(1, 5/100, 2)
# #timenoise[0] = 1    #Make it so the first one is just the standard copy
# print(timenoise.shape)
# print(type(timenoise))
    
# newdf = newdf.multiply(timenoise,axis=0)
    
# signoise = np.random.normal(1, 5/100, [2,3783])
    
# newdf = newdf * signoise


df_aug = augment_data_noise(df_3clust,100,0,1)
#plt.scatter(df_onerow_test.values,df_aug.values)

# %%

#pipe = Pipeline([('function_transformer', FunctionTransformer(np.log1p, validate=True,inverse_func=np.expm1)), 
#                 ('robust_scalar', RobustScaler())])

pipe = Pipeline([('robust_scalar', RobustScaler())])

#transformer = FunctionTransformer(np.log1p, validate=True,inverse_func=np.expm1)

#transformer2 = RobustScaler()

#SCALE THE DATA FOR AUTOENCODER
#transformer = RobustScaler()
# We are going to scale the raw data before passing to an autoencoder. To do
# this lets create a seperate copy of the dataframe
#scaled_df = pd.DataFrame(transformer.fit_transform(df_beijing_filters), columns=df_beijing_filters.columns,index=df_beijing_filters.index)

#Fit to the 2-cluster test data
#scalery = transformer.fit(df_2clust)
#scaled_df = pd.DataFrame(scalery.transform(df_2clust), columns=df_2clust.columns,index=df_2clust.index)

#Fit to the 3-cluster test data

#scaled_df = pd.DataFrame(transformer.fit_transform(df_3clust), columns=df_3clust.columns,index=df_3clust.index)

#scaled_df = pd.DataFrame(transformer2.fit_transform(transformer.fit_transform(df_3clust)), columns=df_3clust.columns,index=df_3clust.index)
scaled_df = pd.DataFrame(pipe.fit_transform(df_aug), columns=df_aug.columns,index=df_aug.index)
scaled_df_val = pd.DataFrame(pipe.fit_transform(df_3clust), columns=df_3clust.columns,index=df_3clust.index)
ae_input_val = scaled_df_val.to_numpy()
#Inverse of scaling
#df_2clust_inverse = scalery.inverse_transform(scaled_df)

# Now extract all of the data as a scaled array
ae_input=scaled_df.to_numpy()
np.count_nonzero(np.isnan(ae_input))  #No NaNs which is good
ae_input.shape

#Linear space 2-cluster data
plt.pcolormesh(ae_input)
plt.title("Linear space 2-cluster ae_input")
plt.show()
#Log-space 2-cluster data
plt.pcolormesh(ae_input,norm=matplotlib.colors.LogNorm())
plt.title("Log space 2-cluster ae_input")
plt.show()

# %%
original_dim = len(df_beijing_filters.columns)

def sampling(args):
    z_mean, z_log_sigma = args
    batch = K.shape(z_mean)[0]
    dim = K.shape(z_mean)[1]
    #epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
    #                          stddev=epsilon_std)
    epsilon = keras.backend.random_normal(shape=(batch, dim)) 
    return z_mean + K.exp(z_log_sigma / 2) * epsilon

# %%
K.clear_session()

    # intermediate_dim1_hp_units = hp.Int('units1', min_value=400, max_value=800, step=20)
    # intermediate_dim2_hp_units = hp.Int('units2', min_value=200, max_value=400, step=20)
    # intermediate_dim3_hp_units = hp.Int('units3', min_value=100, max_value=200, step=10)
    # latent_dim_units = hp.Int('latent_units', min_value=10, max_value=100, step=5)
intermediate_dim1_hp_units = 800
intermediate_dim2_hp_units = 400
intermediate_dim3_hp_units = 200
latent_dim_units = 100

original_inputs = keras.Input(shape=(original_dim,), name="encoder_input")
layer1_vae = layers.Dense(intermediate_dim1_hp_units, activation="relu")(original_inputs)
layer2_vae = layers.Dense(intermediate_dim2_hp_units, activation="relu")(layer1_vae)
layer3_vae = layers.Dense(intermediate_dim3_hp_units, activation="relu")(layer2_vae)
layer4_vae = layers.Dense(latent_dim_units, activation="relu")(layer3_vae)
z_mean = layers.Dense(latent_dim_units, name="z_mean")(layer4_vae)
z_log_sigma = layers.Dense(latent_dim_units, name="z_log_sigma")(layer4_vae)
z = keras.layers.Lambda(sampling, output_shape=(latent_dim_units,))([z_mean, z_log_sigma])
    #z = Sampling()((z_mean, z_log_sigma))
    #encoder_ae = keras.Model(inputs=original_inputs, outputs=layer4_vae, name="encoder_ae")
encoder_vae = keras.Model(original_inputs, [z_mean, z_log_sigma, z], name='encoder_vae')

    # Define decoder model.
latent_inputs_ae = keras.Input(shape=(latent_dim_units,), name="z_sampling")
dec_layer1_vae = layers.Dense(intermediate_dim3_hp_units, activation="relu")(latent_inputs_ae)
dec_layer2_vae = layers.Dense(intermediate_dim2_hp_units, activation="relu")(dec_layer1_vae)
dec_layer3_vae = layers.Dense(intermediate_dim1_hp_units, activation="relu")(dec_layer2_vae)
outputs_vae = layers.Dense(original_dim, activation="linear")(dec_layer3_vae)
decoder_ae = keras.Model(inputs=latent_inputs_ae, outputs=outputs_vae, name="decoder_vae")

    #Define VAE model.
#outputs = decoder_ae(layer4_vae)
#The [2] here means the outputs is z
#outputs = decoder_ae(encoder_vae(original_inputs)[2])
outputs = decoder_ae(z)
vae = keras.Model(inputs=original_inputs, outputs=outputs, name="vae")


#Define loss
def vae_loss(x, x_decoded_mean):
    #xent_loss = original_dim * metrics.binary_crossentropy(original_inputs, outputs)
    #xent_loss= 0.01 
    #kl_loss = - 0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    kl_loss = -0.5 * tf.reduce_mean(z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma) + 1)#daves version
    return kl_loss
    # Add KL divergence regularization loss.
#kl_loss = -0.5 * tf.reduce_mean(z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma) + 1)#daves version
#kl_loss = - 0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)#version from tutorial
#vae.add_loss(kl_loss)
#vae.add_loss(keras.losses.MeanSquaredError(original_inputs, outputs))#test
  
hp_learning_rate = 1e-4



    # #Custom loss term like in keras tutorial
    # reconstruction_loss = keras.losses.MeanSquaredError(original_inputs, outputs)
    # #reconstruction_loss *= original_dim
    # kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    # kl_loss = K.sum(kl_loss, axis=-1)
    # kl_loss *= -0.5
    # vae_loss = K.mean(reconstruction_loss + kl_loss)
    # vae.add_loss(vae_loss)
optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)
vae.compile(optimizer=optimizer,loss=vae_loss)



# %%

vae.fit(ae_input, ae_input, epochs=30, validation_data=(ae_input_val, ae_input_val))