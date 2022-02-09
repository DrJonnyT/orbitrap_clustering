# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(1337) # for reproducibility
import tensorflow as tf
tf.random.set_seed(7331)
import tensorflow.keras as keras
import kerastuner as kt
#from google.colab import drive
import pandas as pd
#import glob
#import pdb
from sklearn.preprocessing import RobustScaler, StandardScaler,FunctionTransformer,MinMaxScaler
from sklearn.pipeline import Pipeline
#from sklearn.decomposition import PCA
#from sklearn.cluster import AgglomerativeClustering, KMeans
import scipy.cluster.hierarchy as sch
import tensorflow as tf
#import keras
#from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
import matplotlib
import os
from tensorflow.keras import backend as K 
#from dateutil import parser
#import math
from joblib import dump, load
import pdb

import os
os.chdir('C:/Work/Python/Github/Orbitrap_clustering')
from ae_functions import *
    

# %%
path='C:/Users/mbcx5jt5/Google Drive/Shared_York_Man2/'


# #1 Load in filter metadata
# df_beijing_metadata = pd.read_excel(path + 'BJ_UnAmbNeg9.1.1_20210505-Times_Fixed.xlsx',engine='openpyxl',
#                                    sheet_name='massloading_Beijing',usecols='A:K',nrows=329,converters={'mid_datetime': str})

# df_beijing_metadata['Sample.ID'] = df_beijing_metadata['Sample.ID'].astype(str)

# df_beijing_raw = pd.read_excel(path + 'BJ_UnAmbNeg9.1.1_20210505-Times_Fixed.xlsx',engine='openpyxl',sheet_name='Compounds')


# #Load Delhi data and remove columns that are not needed
# df_delhi_raw = pd.read_excel(path + 'Delhi_Amb3.1_MZ.xlsx',engine='openpyxl')
# df_delhi_raw.drop(df_delhi_raw.iloc[:,np.r_[0, 2:11, 14:18]],axis=1,inplace=True)

# df_delhi_metadata = pd.read_excel(path + 'Delhi/Delhi_massloading_autumn_summer.xlsx',engine='openpyxl',
#                                    sheet_name='autumn',usecols='a:N',skiprows=0,nrows=108, converters={'mid_datetime': str})
# df_delhi_metadata.drop(labels="Filter ID.1",axis=1,inplace=True)
# df_delhi_metadata.set_index("Filter ID",inplace=True)
# df_delhi_metadata.drop(labels=["-","S25","S42","S51","S55","S68","S72"],axis=0,inplace=True)




# beijing_peaks_examples = pd.DataFrame([df_beijing_raw["Formula"],
#                                       df_beijing_raw["Formula"].apply(lambda x: filter_by_chemform(x))]).T

df_beijing_raw, df_beijing_filters, df_beijing_metadata = beijing_load(
    path + 'BJ_UnAmbNeg9.1.1_20210505-Times_Fixed.xlsx',path + 'BJ_UnAmbNeg9.1.1_20210505-Times_Fixed.xlsx',
    peaks_sheetname="Compounds",metadata_sheetname="massloading_Beijing")

df_delhi_raw, df_delhi_filters, df_delhi_metadata = delhi_load(
    path + 'Delhi_Amb3.1_MZ.xlsx',path + 'Delhi/Delhi_massloading_autumn_summer.xlsx')


#df_delhi_raw, df_delhi_raw_loaded,df_delhi_metadata, df_delhi_raw_blanks= delhi_load(
#    path + 'Delhi_Amb3.1_MZ.xlsx',path + 'Delhi/Delhi_massloading_autumn_summer.xlsx')



df_beijing_winter = df_beijing_filters.iloc[0:124].copy()
df_beijing_summer = df_beijing_filters.iloc[124:].copy()




# %%Check largest peaks

beijing_chemform_namelist = load_chemform_namelist(path + 'Beijing_Amb3.1_MZ.xlsx')
delhi_chemform_namelist = load_chemform_namelist(path + 'Delhi_Amb3.1_MZ.xlsx')

print("BEIJING SUMMER")
a =cluster_extract_peaks(df_beijing_summer.sum(), df_beijing_raw,10,beijing_chemform_namelist)
print("BEIJING WINTER")
a = cluster_extract_peaks(df_beijing_winter.sum(), df_beijing_raw,10,beijing_chemform_namelist)
print("DELHI AUTUMN")
a = cluster_extract_peaks(df_delhi_filters.sum(), df_delhi_raw,10,delhi_chemform_namelist)

print("BEIJING BLANK")
a = cluster_extract_peaks(df_beijing_raw.iloc[:,320].transpose(), df_beijing_raw,10,beijing_chemform_namelist)

print("DELHI BLANKS MEAN")
a = cluster_extract_peaks(df_delhi_raw_blanks.transpose().mean(), df_delhi_raw,10,delhi_chemform_namelist)


# %%



#Some basic checks like the time series of total concentration
beijing_rows_sum = df_beijing_filters.sum(axis=1)
plt.plot(beijing_rows_sum)
plt.title("Total sum of all peaks")
plt.show()



fig = plt.plot(figsize=(30,20))
#plt.pcolormesh(df_beijing_raw.iloc[:,list(range(4,321))],norm=matplotlib.colors.LogNorm())
plt.pcolormesh(df_beijing_filters.astype(float))
plt.title("Filter data, linear scale")
plt.show()

fig = plt.plot(figsize=(30,20))
plt.pcolormesh(df_beijing_filters.astype(float),norm=matplotlib.colors.LogNorm())
plt.title("Filter data, log scale")
plt.show()

#
print("The index/column of max data point is " + str(df_beijing_filters.stack().idxmax()))


# %%Combine Beijing and Delhi
#df_all_filters = pd.concat([df_beijing_filters, df_delhi_filters], axis=1)
df_all_filters = df_beijing_filters.append(df_delhi_filters,sort=True)



# %%
#Define the scaling and train it on the full dataset

#MinMaxScaler- scale each feature between 0 and 1
#pipe = Pipeline([('function_transformer', FunctionTransformer(np.log1p, validate=True,inverse_func=np.expm1)), 
 #                ('robust_scalar', RobustScaler())])
#pipe = Pipeline([('function_transformer', FunctionTransformer(np.log1p, validate=True,inverse_func=np.expm1))])
#pipe = MinMaxScaler()
#pipe = StandardScaler()
pipe = RobustScaler()
pipe.fit(df_all_filters)
df_all_scaled = pd.DataFrame(pipe.transform(df_all_filters))
df_all_scaled = df_all_scaled.fillna(0)
fig = plt.plot(figsize=(30,20))
plt.pcolormesh(df_all_scaled.astype(float))
plt.title("Filter data, logged and MinMaxScaled")
plt.show()



#Make a class for the AE input. This is a bit convoluted but it's because you need
#to make sure you deal with NaNs right, ie you need to turn NaNs to zero AFTER you scale the data
class ae_input:
  def __init__(self, ae_input, val, pipe):
    self.pipe = pipe
    self.scaled = pipe.transform(ae_input)
    self.val = age
    self.unscaled

  def myfunc(self):
    print("Hello my name is " + self.name)



# %%
# Now we can perform agglomerative cluster analysis on this new ouput, assigning each row to a particular cluster. We have to define the number of clusters but thats ok for now

agglom_native = AgglomerativeClustering(n_clusters = 3, linkage = 'ward')
#clustering = agglom.fit(latent_space)
clustering = agglom_native.fit(df_beijing_filters.values)
plt.scatter(df_beijing_filters.index,clustering.labels_)
plt.title("Agglom clustering labels, unencoded data")
plt.show()

#And what are the clusters?
#The cluster labels can just be moved straight out the latent space
cluster0_uncoded = df_beijing_filters[clustering.labels_==0].mean()
cluster1_uncoded = df_beijing_filters[clustering.labels_==1].mean()
cluster2_uncoded = df_beijing_filters[clustering.labels_==2].mean()


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
    
    # if(t_noise_pct == 0):
    #     #SOMETHING NOT RIGHT HERE??
    #     print("Warning! Not sure this is working in augment_data_noise")
    #     timenoise = np.ones(num_copies*num_rows)
    # else:
    timenoise = np.random.normal(1, t_noise_pct/100, num_copies*num_rows)
    # timenoise=1
    # #timenoise[0] = 1    #Make it so the first one is just the standard copy
    newdf = newdf.multiply(timenoise,axis=0)
    if(sig_noise_pct == 0):
        signoise = np.ones([num_copies*num_rows,num_cols])
    else:
        signoise = np.random.normal(1, sig_noise_pct/100, [num_copies*num_rows,num_cols])
        print("using sig noise")
    
    newdf = newdf * signoise
       


    #Efficient version
    #newdf = pd.DataFrame(np.repeat(df.values,num_copies,axis=0)) * np.random.normal(1, sig_noise_pct/100, [num_copies*num_rows,num_cols])
    return newdf
# %%


#Augment the data
df_aug = augment_data_noise(df_all_filters,50,1,0)
#plt.scatter(df_onerow_test.values,df_aug.values)

# plt.pcolormesh(df_aug.astype(float))
# plt.title("Linear space scaled data")
# plt.show()

# plt.pcolormesh(df_aug.astype(float),norm=matplotlib.colors.LogNorm())
# plt.title("Log space scaled data")
# plt.show()

# %%
#Scale the data for input into AE
pipe.fit(df_aug)
scaled_df = pd.DataFrame(pipe.transform(df_aug),columns=df_all_filters.columns)
scaled_df = scaled_df.fillna(0)
ae_input=scaled_df.to_numpy()
scaled_df_val = pd.DataFrame(pipe.transform(df_all_filters), columns=df_all_filters.columns,index=df_all_filters.index)
scaled_df_val = scaled_df_val.fillna(0)
ae_input_val = scaled_df_val.to_numpy()

plt.pcolormesh(ae_input_val)
plt.title("Scaled space ae_input_val")
plt.show()


#Some basic checks like the time series of total concentration
scaled_df_sum = scaled_df.sum(axis=1)
plt.plot(scaled_df_sum)
plt.title("Total sum of all peaks")
plt.show()


# %%
#Lets try a dendrogram to work out the optimal number of clusters

fig,axes = plt.subplots(1,1,figsize=(20,10))
plt.title("dendrogram")
dendrogram = sch.dendrogram(sch.linkage(scaled_df_val, method='ward'))
plt.title("Dendrogram of scaled data")
plt.show()



# %%#How many clusters should we have? Real-space
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
min_clusters = 2
max_clusters = 10

num_clusters_index = range(min_clusters,(max_clusters+1),1)
ch_score = np.empty(len(num_clusters_index))
db_score = np.empty(len(num_clusters_index))

for num_clusters in num_clusters_index:
    agglom_native = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    #agglom_native = KMeans(n_clusters = num_clusters)
    clustering = agglom_native.fit(df_beijing_filters.fillna(0).values)
    ch_score[num_clusters-min_clusters] = calinski_harabasz_score(df_beijing_filters.fillna(0).values, clustering.labels_)
    db_score[num_clusters-min_clusters] = davies_bouldin_score(df_beijing_filters.fillna(0).values, clustering.labels_)
fig,ax1 = plt.subplots()
ax2=ax1.twinx()
ax1.plot(num_clusters_index,ch_score,label="CH score")
ax2.plot(num_clusters_index,db_score,c='red',label="DB score")
ax1.set_xlabel("Num clusters")
ax1.set_ylabel("CH score")
ax2.set_ylabel("DB score")
plt.title("How many clusters? Real-space input data")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc=2)
plt.show()


# %%#How many clusters should we have? Scaled-space data

min_clusters = 2
max_clusters = 10

num_clusters_index = range(min_clusters,(max_clusters+1),1)
ch_score = np.empty(len(num_clusters_index))
db_score = np.empty(len(num_clusters_index))

for num_clusters in num_clusters_index:
    agglom_native = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    #agglom_native = KMeans(n_clusters = num_clusters)
    clustering = agglom_native.fit(scaled_df_val.values)
    ch_score[num_clusters-min_clusters] = calinski_harabasz_score(scaled_df_val.values, clustering.labels_)
    db_score[num_clusters-min_clusters] = davies_bouldin_score(scaled_df_val.values, clustering.labels_)
fig,ax1 = plt.subplots()
ax2=ax1.twinx()
ax1.plot(num_clusters_index,ch_score,label="CH score")
ax2.plot(num_clusters_index,db_score,c='red',label="DB score")
ax1.set_xlabel("Num clusters")
ax1.set_ylabel("CH score")
ax2.set_ylabel("DB score")
plt.title("How many clusters? RobustScaled input data")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc=2)
plt.show()
# %%Define AE


############################################################
#AUTOENCODER DEINITION
#STANDARD AE WORKING
############################################################
original_dim = ae_input.shape[1]
#intermediate_dim1_vae = original_dim//2
#intermediate_dim2_vae = original_dim//5
#intermediate_dim3_vae = original_dim//10
#latent_dim_vae = 8

# - uncomment the following if using a Variational autoencoder. 
#class Sampling(layers.Layer):
#    """Uses (z_mean, z_log_var) to sample z."""#
#
#    def call(self, inputs):
#        z_mean, z_log_var = inputs
#        batch = tf.shape(z_mean)[0]
#        dim = tf.shape(z_mean)[1]
#        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
#        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def model_builder(hp):

    intermediate_dim1_hp_units = hp.Int('units1', min_value=400, max_value=800, step=20)
    intermediate_dim2_hp_units = hp.Int('units2', min_value=200, max_value=400, step=20)
    intermediate_dim3_hp_units = hp.Int('units3', min_value=100, max_value=200, step=10)
    latent_dim_units = hp.Int('latent_units', min_value=10, max_value=100, step=5)
    # intermediate_dim1_hp_units = hp.Int('units1', min_value=800, max_value=1600, step=40)
    # intermediate_dim2_hp_units = hp.Int('units2', min_value=400, max_value=800, step=40)
    # intermediate_dim3_hp_units = hp.Int('units3', min_value=200, max_value=400, step=20)
    # latent_dim_units = hp.Int('latent_units', min_value=10, max_value=200, step=10)

    original_inputs = tf.keras.Input(shape=(original_dim,), name="encoder_input")
    layer1_vae = layers.Dense(intermediate_dim1_hp_units, activation="relu")(original_inputs)
    layer2_vae = layers.Dense(intermediate_dim2_hp_units, activation="relu")(layer1_vae)
    layer3_vae = layers.Dense(intermediate_dim3_hp_units, activation="relu")(layer2_vae)
    #layer4_vae = layers.Dense(latent_dim_units, activation="relu")(layer3_vae)
    #JT changed, was originally relu. Sigmoid so latent space is between 0 - 1
    layer4_vae = layers.Dense(latent_dim_units, activation="sigmoid")(layer3_vae)
    
    #z_mean = layers.Dense(latent_dim_units, name="z_mean")(layer3_vae)
    #z_log_var = layers.Dense(latent_dim_units, name="z_log_var")(layer3_vae)
    #z = Sampling()((z_mean, z_log_var))
    encoder_ae = tf.keras.Model(inputs=original_inputs, outputs=layer4_vae, name="encoder_ae")

    # Define decoder model.
    latent_inputs_ae = tf.keras.Input(shape=(latent_dim_units,), name="z_sampling")
    dec_layer1_vae = layers.Dense(intermediate_dim3_hp_units, activation="relu")(latent_inputs_ae)
    dec_layer2_vae = layers.Dense(intermediate_dim2_hp_units, activation="relu")(dec_layer1_vae)
    dec_layer3_vae = layers.Dense(intermediate_dim1_hp_units, activation="relu")(dec_layer2_vae)
    outputs_vae = layers.Dense(original_dim, activation="linear")(dec_layer3_vae)
    decoder_ae = tf.keras.Model(inputs=latent_inputs_ae, outputs=outputs_vae, name="decoder_vae")

    #Define VAE model.
    outputs = decoder_ae(layer4_vae)
    ae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name="vae")
    # Add KL divergence regularization loss.
    #kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    #vae.add_loss(kl_loss)
    
      
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])

    optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
    
    #COMPILING
    #Standard compilation
    #ae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
    #Compile weighted to reduce number of columns
    ae.compile(optimizer, loss=['mean_squared_error'],loss_weights=[np.sqrt(latent_dim_units)])

    return ae

# %%Define VAE

############################################################
#AUTOENCODER DEINITION
#VARIATIONAL AE NOT CURRENTLY WORKING
############################################################
original_dim = ae_input.shape[1]
#intermediate_dim1_vae = original_dim//2
#intermediate_dim2_vae = original_dim//5
#intermediate_dim3_vae = original_dim//10
#latent_dim_vae = 8

# # - uncomment the following if using a Variational autoencoder. 
# class Sampling(layers.Layer):
#     """Uses (z_mean, z_log_var) to sample z."""#

#     def call(self, inputs):
#         z_mean, z_log_var = inputs
#         batch = tf.shape(z_mean)[0]
#         dim = tf.shape(z_mean)[1]
#         epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
#         return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
# def sampling(args):
#     z_mean, z_log_sigma = args
#     epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
#                               mean=0., stddev=0.1)
#     return z_mean + K.exp(z_log_sigma) * epsilon

def model_builder(hp):

    intermediate_dim1_hp_units = hp.Int('units1', min_value=400, max_value=800, step=20)
    intermediate_dim2_hp_units = hp.Int('units2', min_value=200, max_value=400, step=20)
    intermediate_dim3_hp_units = hp.Int('units3', min_value=100, max_value=200, step=10)
    latent_dim_units = hp.Int('latent_units', min_value=10, max_value=100, step=5)
    # intermediate_dim1_hp_units = hp.Int('units1', min_value=800, max_value=1600, step=40)
    # intermediate_dim2_hp_units = hp.Int('units2', min_value=400, max_value=800, step=40)
    # intermediate_dim3_hp_units = hp.Int('units3', min_value=200, max_value=400, step=20)
    # latent_dim_units = hp.Int('latent_units', min_value=10, max_value=200, step=10)

    original_inputs = tf.keras.Input(shape=(original_dim,), name="encoder_input")
    layer1_vae = layers.Dense(intermediate_dim1_hp_units, activation="relu")(original_inputs)
    layer2_vae = layers.Dense(intermediate_dim2_hp_units, activation="relu")(layer1_vae)
    layer3_vae = layers.Dense(intermediate_dim3_hp_units, activation="relu")(layer2_vae)
    #layer4_vae = layers.Dense(latent_dim_units, activation="relu")(layer3_vae)
    #JT changed, was originally relu. Sigmoid so latent space is between 0 - 1
    layer4_vae = layers.Dense(latent_dim_units, activation="sigmoid")(layer3_vae)
    
    # z_mean = layers.Dense(latent_dim_units, name="z_mean")(layer3_vae)
    # z_log_sigma = layers.Dense(latent_dim_units, name="z_log_var")(layer3_vae)
    # #z = Sampling()((z_mean, z_log_var))#Dave's original line
    # z = layers.Lambda(sampling)([z_mean, z_log_sigma])
    
    
    encoder_ae = tf.keras.Model(inputs=original_inputs, outputs=layer4_vae, name="encoder_ae")
    
    #encoder_vae = keras.Model(original_inputs, [z_mean, z_log_sigma, z], name='encoder')
    

    # Define decoder model.
    latent_inputs_ae = tf.keras.Input(shape=(latent_dim_units,), name="z_sampling")
    dec_layer1_vae = layers.Dense(intermediate_dim3_hp_units, activation="relu")(layer4_vae)
    dec_layer2_vae = layers.Dense(intermediate_dim2_hp_units, activation="relu")(dec_layer1_vae)
    dec_layer3_vae = layers.Dense(intermediate_dim1_hp_units, activation="relu")(dec_layer2_vae)
    outputs_vae = layers.Dense(original_dim, activation="linear")(dec_layer3_vae)
    decoder_ae = tf.keras.Model(inputs=latent_inputs_ae, outputs=outputs_vae, name="decoder_vae")

    #Define VAE model.
    outputs = decoder_ae(layer4_vae)
    #outputs_vae = decoder(encoder(inputs)[2])
    
    ae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name="vae")
    # Add KL divergence regularization loss.
    #kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    #vae.add_loss(kl_loss)
    
      
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])

    optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)


    reconstruction_loss = keras.losses.MeanSquaredError(inputs=original_inputs, outputs=outputs)
    #reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
    #reconstruction_loss *= original_dim

    vae = keras.Model(original_inputs, outputs, name='vae_mlp')
    #vae_loss = K.mean(reconstruction_loss + kl_loss)
    #vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')




    
    #COMPILING
    #Standard compilation
    #ae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
    #Compile weighted to reduce number of columns
    ae.compile(optimizer, loss=['mean_squared_error'],loss_weights=[latent_dim_units])

    return vae
    #return ae


#FOR VAE CAN DO IT THIS WAY
# model.compile(
#     optimizer='rmsprop',
#     loss=['binary_crossentropy', 'mean_squared_error'],
#     loss_weights=[1., 0.2]
# )









# %%Hypertune!
K.clear_session()
##############################
##TUNING HYPERPARAMETERS
##############################
tuner = kt.Hyperband(model_builder,
                     objective='val_loss',
                    max_epochs=10,
                    factor=3,
                    directory=os.path.normpath('C:/work/temp/keras'),
                    overwrite=True)

#This gives an error at the end on windows but don't worry about it
tuner.search(ae_input, ae_input, epochs=30, validation_data=(ae_input_val, ae_input_val))

# %% Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units1')}, the second {best_hps.get('units2')}, third {best_hps.get('units3')}, latent {best_hps.get('latent_units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# %%Build the model with the optimal hyperparameters and train it on the data for 30 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(ae_input, ae_input, epochs=30, validation_data=(ae_input_val, ae_input_val))
val_acc_per_epoch = history.history['val_loss']
best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

# %%hypermodel

hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
history_new = hypermodel.fit(ae_input, ae_input, epochs=best_epoch, validation_data=(ae_input_val, ae_input_val))
# plot loss history
loss = history_new.history['loss']
val_loss = history_new.history['val_loss']
epochs = range(len(loss))

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# %%Retrain model with optimal parameters

#########################################################################################
# Perform cluster analysis
#Now that we have our optimised AE, we can predict the latent space.
#To do that, we need to rebuild our autoencoder since we cannot access the encoder part of our model.
#However, not to worry since we have recorded the best set of parameters.
#In the following code block we retrain the autoencoder and then we are going to extract the output from only
#the encoder model and then use this for cluster analysis. 

# Define encoder model.
original_inputs = tf.keras.Input(shape=(original_dim,), name="encoder_input")
layer1_vae = layers.Dense(best_hps.get('units1'), activation="relu")(original_inputs)
layer2_vae = layers.Dense(best_hps.get('units2'), activation="relu")(layer1_vae)
layer3_vae = layers.Dense(best_hps.get('units3'), activation="relu")(layer2_vae)
layer4_vae = layers.Dense(best_hps.get('latent_units'), activation="sigmoid")(layer3_vae)
#z_mean = layers.Dense(best_hps.get('latent_units'), name="z_mean")(layer3_vae)
#z_log_var = layers.Dense(best_hps.get('latent_units'), name="z_log_var")(layer3_vae)
#z = Sampling()((z_mean, z_log_var))
encoder_ae = tf.keras.Model(inputs=original_inputs, outputs=layer4_vae, name="encoder_ae")

# Define decoder model.
latent_inputs_ae = tf.keras.Input(shape=(best_hps.get('latent_units'),), name="decoder_input")
dec_layer1_vae = layers.Dense(best_hps.get('units3'), activation="relu")(latent_inputs_ae)
dec_layer2_vae = layers.Dense(best_hps.get('units2'), activation="relu")(dec_layer1_vae)
dec_layer3_vae = layers.Dense(best_hps.get('units1'), activation="relu")(dec_layer2_vae)
outputs_ae = layers.Dense(original_dim, activation="linear")(dec_layer3_vae)
decoder_ae = tf.keras.Model(inputs=latent_inputs_ae, outputs=outputs_ae, name="decoder_ae")

# Define VAE model.
outputs = decoder_ae(layer4_vae)
ae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name="ae")

# Add KL divergence regularization loss.
#kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
#vae.add_loss(kl_loss)
# Model summary
print(ae.summary())
#train_data,test_data,_,_ = train_test_split(vae_input,vae_input,test_size=0.2)
optimizer = tf.keras.optimizers.Adam(learning_rate=best_hps.get('learning_rate'))
ae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())

training_history = ae.fit(ae_input, ae_input, epochs=best_epoch, validation_data=(ae_input_val, ae_input_val))


#######################################################################################
#Now we use the trained encoder part of our model to extract the deep learning latent space.
#We can then perform cluster analysis on this latent space.
#If we define the number of clusters we then add these cluster labels back into our original dataset [data-frame].
#We can then, for example, see what time of the day, certain clusters appear.
#First, lets call the encoder and extract our new latent space.
#Following this, we apply cluster analysis to this new space.
#Just to confirm, each row in in new space is the new latent representation of 
#our traffic network at a given point in time.

# %%Make the latent space
## Call the now re-trained encoder part of our model
latent_space = encoder_ae.predict(ae_input_val)
latent_space.shape
df_latent_space = pd.DataFrame(latent_space)
latent_space_sum = latent_space.sum(axis=1)


# %%#Save the model
encoder_ae.save(r'C:\Work\Python\Github\Orbitrap_clustering\Models\encoder_ae')
#encoder_ae.save(r'C:\Work\Python\Github\Orbitrap_clustering\Models\encoder_ae\encoder_ae.h5')

decoder_ae.save(r'C:\Work\Python\Github\Orbitrap_clustering\Models\decoder_ae')
#decoder_ae.save(r'C:\Work\Python\Github\Orbitrap_clustering\Models\decoder_ae\decoder_ae.h5')
ae.save(r'C:\Work\Python\Github\Orbitrap_clustering\Models\ae')
#ae.save(r'C:\Work\Python\Github\Orbitrap_clustering\Models\ae\ae.h5')


dump(pipe, r'C:\Work\Python\Github\Orbitrap_clustering\Models\ae_pipe\pipe.joblib') 


# %%

#How good is our encode/decode?
#This is the really key plot. It's shit!! So need to improve it I think, otherwise it's not useful for getting the factors out of the latent space
#And if you don't know what the factors out, then you can't use them
#plt.scatter(ae_input_val,ae.predict(ae_input_val))
plt.scatter(df_all_filters.values,pipe.inverse_transform(ae.predict(ae_input_val)))
plt.xlabel("Input data")
plt.ylabel("Reconstructed data")
plt.title("Point-by-point autoencoder performance")
plt.show()

# %%
#Is the sum total linear with input vs output?
plt.scatter(df_all_filters.sum(axis=1),pipe.inverse_transform(ae.predict(ae_input_val)).sum(axis=1),c=df_all_filters.index)
plt.xlabel("AE input")
plt.ylabel("AE output")


# %%
#What does the latent space look like?
plt.pcolormesh(latent_space)
plt.title("AE Latent space")
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
    pca_ae = PCA(n_components = num_components)
    prin_comp = pca.fit_transform(df_all_filters.fillna(0).to_numpy())
    pca_variance_explained[num_components-min_components] = pca.explained_variance_ratio_.sum()
    prin_comp = pca.fit_transform(ae_input_val)
    pca_scaled_variance_explained[num_components-min_components] = pca.explained_variance_ratio_.sum()
    prin_comp_ae = pca_ae.fit_transform(latent_space)
    pca_ae_variance_explained[num_components-min_components] = pca_ae.explained_variance_ratio_.sum()
    
fig,ax = plt.subplots()
ax.plot(num_components_index,pca_variance_explained,label="PCA on unscaled data")
ax.plot(num_components_index,pca_scaled_variance_explained,label="PCA on scaled AE input")
ax.plot(num_components_index,pca_ae_variance_explained,c='red',label="PCA on latent space")
ax.set_xlabel("Num PCA components")
ax.set_ylabel("Fraction of variance explained")
plt.legend(loc="lower right")
plt.title("How many PCA components?")
plt.show()



# %%


############################################################################################
#CLUSTERING AND FACTOR ANALYSIS OF REAL SPACE DATA
############################################################################################
#fig,axes = plt.subplots(1,1,figsize=(20,10))
plt.title("dendrogram")
dendrogram = sch.dendrogram(sch.linkage(df_beijing_filters.to_numpy(), method='ward'))
plt.title("Real space dendrogram")
plt.show()


# %%
# Now we can perform agglomerative cluster analysis on this new ouput, assigning each row to a particular cluster. We have to define the number of clusters but thats ok for now
n_clusters = 5
agglom = AgglomerativeClustering(n_clusters = n_clusters, linkage = 'ward')
#clustering = agglom.fit(latent_space)
clustering = agglom.fit(df_beijing_filters.to_numpy())
plt.plot(df_beijing_filters.index,clustering.labels_,marker=".")
plt.title("Real space cluster labels, " + str(n_clusters) + " clusters")
#plt.plot(df_beijing_filters.index,clustering.labels_)


# %%
############################################################################################
#CLUSTERING AND FACTOR ANALYSIS OF SCALEDDATA
############################################################################################
#fig,axes = plt.subplots(1,1,figsize=(20,10))

# %%
fig = plt.plot(figsize=(30,20))
#plt.pcolormesh(df_beijing_raw.iloc[:,list(range(4,321))],norm=matplotlib.colors.LogNorm())
plt.pcolormesh(ae_input_val)
plt.title("Filter data, MinMaxScaled")
plt.show()


plt.title("dendrogram")
dendrogram = sch.dendrogram(sch.linkage(ae_input_val, method='ward'))
plt.title("Scaled space dendrogram")
plt.show()





# %%
# Now we can perform agglomerative cluster analysis on this new ouput, assigning each row to a particular cluster. We have to define the number of clusters but thats ok for now
n_clusters = 5
agglom = AgglomerativeClustering(n_clusters = n_clusters, linkage = 'ward')
#clustering = agglom.fit(latent_space)
clustering = agglom.fit(ae_input_val)
plt.plot(df_beijing_filters.index,clustering.labels_,marker=".")
plt.title("Scaled space cluster labels, " + str(n_clusters) + " clusters")
#plt.plot(df_beijing_filters.index,clustering.labels_)

# %%
############################################################################################
#CLUSTERING AND FACTOR ANALYSIS OF LATENT SPACE DATA
############################################################################################

#Lets try a dendrogram to work out the optimal number of clusters


plt.title("dendrogram")
dendrogram = sch.dendrogram(sch.linkage(latent_space, method='ward'))
plt.title("Latent space dendrogram")
plt.show()



# %%#How many clusters should we have? Latent space
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
min_clusters = 2
max_clusters = 30

num_clusters_index = range(min_clusters,(max_clusters+1),1)
ch_score = np.empty(len(num_clusters_index))
db_score = np.empty(len(num_clusters_index))

for num_clusters in num_clusters_index:
    agglom_native = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    #agglom_native = KMeans(n_clusters = num_clusters)
    clustering = agglom_native.fit(latent_space)
    ch_score[num_clusters-min_clusters] = calinski_harabasz_score(latent_space, clustering.labels_)
    db_score[num_clusters-min_clusters] = davies_bouldin_score(latent_space, clustering.labels_)
fig,ax1 = plt.subplots()
ax2=ax1.twinx()
ax1.plot(num_clusters_index,ch_score,label="CH score")
ax2.plot(num_clusters_index,db_score,c='red',label="DB score")
ax1.set_xlabel("Num clusters")
ax1.set_ylabel("CH score")
ax2.set_ylabel("DB score")
plt.title("How many clusters? Latent-space data")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)
plt.show()

# from lifelines.utils import concordance_index
# cph = CoxPHFitter().fit(df, 'T', 'E')
# concordance_index(df['T'], -cph.predict_partial_hazard(df), df['E'])

# %%#How many clusters should we have? 3-component PCA output

pca3 = PCA(n_components = 3)
prin_comp3 = pca.fit_transform(latent_space)

from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
min_clusters = 2
max_clusters = 10

num_clusters_index = range(min_clusters,(max_clusters+1),1)
ch_score = np.empty(len(num_clusters_index))
db_score = np.empty(len(num_clusters_index))

for num_clusters in num_clusters_index:
    agglom_native = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    #agglom_native = KMeans(n_clusters = num_clusters)
    clustering = agglom_native.fit(prin_comp3)
    ch_score[num_clusters-min_clusters] = calinski_harabasz_score(prin_comp3, clustering.labels_)
    db_score[num_clusters-min_clusters] = davies_bouldin_score(prin_comp3, clustering.labels_)
fig,ax1 = plt.subplots()
ax2=ax1.twinx()
ax1.plot(num_clusters_index,ch_score,label="CH score")
ax2.plot(num_clusters_index,db_score,c='red',label="DB score")
ax1.set_xlabel("Num clusters")
ax1.set_ylabel("CH score")
ax2.set_ylabel("DB score")
plt.title("How many clusters? 3-component PCA output")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)
plt.show()




# %%Calculate mean shift on PCA
from sklearn.cluster import MeanShift
ms = MeanShift(bandwidth=1)
ms.fit(latent_space)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)


# %%
# Now we can perform agglomerative cluster analysis on this new ouput, assigning each row to a particular cluster. We have to define the number of clusters but thats ok for now
n_clusters = 5
agglom = AgglomerativeClustering(n_clusters = n_clusters, linkage = 'ward')
#clustering = agglom.fit(latent_space)
clustering = agglom.fit(latent_space)
plt.plot(df_beijing_filters.index,clustering.labels_,marker=".")
plt.title("Latent space cluster labels, " + str(n_clusters) + " clusters")
#plt.plot(df_beijing_filters.index,clustering.labels_)
# %%
#And what are the clusters?
#The cluster labels can just be moved straight out the latent space
cluster0_decoded = df_beijing_filters[clustering.labels_==0].mean()
cluster1_decoded = df_beijing_filters[clustering.labels_==1].mean()
cluster2_decoded = df_beijing_filters[clustering.labels_==2].mean()

#Latent space clusters
cluster0_lat = df_latent_space[clustering.labels_==0].mean()
cluster1_lat = df_latent_space[clustering.labels_==1].mean()
cluster2_lat = df_latent_space[clustering.labels_==2].mean()


# %%

#Now lets decode the clusters from latent space and see if it still works
#Cluster0_decod = transformer.inverse_transform(decoder_ae.predict(np.expand_dims(cluster0_lat, axis=0)))
#Cluster1_decod = transformer.inverse_transform(decoder_ae.predict(np.expand_dims(cluster1_lat, axis=0)))
#Cluster2_decod = transformer.inverse_transform(decoder_ae.predict(np.expand_dims(cluster2_lat, axis=0)))

Cluster0_decod = pipe.inverse_transform((decoder_ae.predict(np.expand_dims(cluster0_lat, axis=0))))
Cluster1_decod = pipe.inverse_transform((decoder_ae.predict(np.expand_dims(cluster1_lat, axis=0))))
Cluster2_decod = pipe.inverse_transform((decoder_ae.predict(np.expand_dims(cluster2_lat, axis=0))))

#THESE SHOULD ALWAYS ALWAYS BE A STRAIGHT LINE OTHERWISE SOMEHTING IT NOT WORKING RIGHT IN THE AE OR TRANSFORMER
plt.scatter(cluster0_decoded,Cluster0_decod)

plt.scatter(cluster1_decoded,Cluster1_decod)

plt.scatter(cluster2_decoded,Cluster2_decod)

# %%
#Now do some NMF clustering
from sklearn.decomposition import NMF
num_nmf_factors = 5
nmf_model = NMF(n_components=num_nmf_factors)
#model.fit(latent_space)

#Get the different components
#nmf_features = model.transform(latent_space)
#print(model.components_)

W = nmf_model.fit_transform(latent_space)
#W = model.fit_transform(df_3clust.values)
H = nmf_model.components_

# %%
prefix = "Beijing_summer_"
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
Factor2_lat = H[2]
Factor3_lat = H[3]
Factor4_lat = H[4]


Factor0_lat_mtx = np.outer(W.T[0], H[0])
Factor1_lat_mtx = np.outer(W.T[1], H[1])
Factor2_lat_mtx = np.outer(W.T[2], H[2])
Factor3_lat_mtx = np.outer(W.T[3], H[3])
Factor4_lat_mtx = np.outer(W.T[4], H[4])
#Now need to decode these matrices to get the time series matrix of each factor

Factor0_decod = pipe.inverse_transform(decoder_ae.predict(np.expand_dims(Factor0_lat, axis=0)))
Factor1_decod = pipe.inverse_transform(decoder_ae.predict(np.expand_dims(Factor1_lat, axis=0)))
Factor2_decod = pipe.inverse_transform(decoder_ae.predict(np.expand_dims(Factor2_lat, axis=0)))
Factor3_decod = pipe.inverse_transform(decoder_ae.predict(np.expand_dims(Factor3_lat, axis=0)))
Factor4_decod = pipe.inverse_transform(decoder_ae.predict(np.expand_dims(Factor4_lat, axis=0)))

Factor0_mtx_decod = pipe.inverse_transform(decoder_ae.predict(Factor0_lat_mtx))
Factor1_mtx_decod = pipe.inverse_transform(decoder_ae.predict(Factor1_lat_mtx))
Factor2_mtx_decod = pipe.inverse_transform(decoder_ae.predict(Factor2_lat_mtx))
Factor3_mtx_decod = pipe.inverse_transform(decoder_ae.predict(Factor3_lat_mtx))
Factor4_mtx_decod = pipe.inverse_transform(decoder_ae.predict(Factor4_lat_mtx))

Factor0_decod_sum = Factor0_mtx_decod.sum(axis=1)
Factor1_decod_sum = Factor1_mtx_decod.sum(axis=1)
Factor2_decod_sum = Factor2_mtx_decod.sum(axis=1)
Factor3_decod_sum = Factor3_mtx_decod.sum(axis=1)
Factor4_decod_sum = Factor4_mtx_decod.sum(axis=1)

plt.plot(Factor0_decod_sum)
plt.plot(Factor1_decod_sum)
plt.plot(Factor2_decod_sum)
plt.plot(Factor3_decod_sum)
plt.plot(Factor4_decod_sum)
plt.ylim(bottom=0)
plt.show()


# %%
from hyperspy.signals import Signal1D
s = Signal1D(np.random.randn(10, 10, 200))
s.decomposition()

# %%
#The latent-space factors and clusters are the same (not the same labels though)
plt.scatter(cluster2_lat,Factor1_lat)

plt.scatter(cluster0_lat,Factor2_lat)

plt.scatter(cluster1_lat,Factor0_lat)

# %%

#Now lets compare the decoded factors to the imput factors
#This one correlates well
plt.scatter(Cluster_B,Factor2_decod)
#plt.scatter(max_filter,Factor0_decod)
plt.xlabel("Input peak height")
plt.ylabel("Output peak height")

#This one does not
plt.scatter(Cluster_A,Factor1_decod)
#plt.scatter(min_filter,Factor1_decod)
plt.xlabel("Input peak height")
plt.ylabel("Output peak height")

#This one does not either
plt.scatter(Cluster_B,Factor0_decod)
#plt.scatter(thousand_filter,Factor2_decod)
plt.xlabel("Input peak height")
plt.ylabel("Output peak height")



#Latent space factors do not correlate
plt.scatter(Factor0_lat,Factor1_lat)
plt.scatter(Factor0_lat,Factor2_lat)
plt.scatter(Factor1_lat,Factor2_lat)


#But decoded factors correlate really well
plt.scatter(Factor0_decod,Factor1_decod)
plt.scatter(Factor0_decod,Factor2_decod)
plt.scatter(Factor1_decod,Factor2_decod)

plt.xlabel("Input peak height")
plt.ylabel("Output peak height")





#What about before unscaling with pipeline?
Factor0_half_decod = decoder_ae.predict(np.expand_dims(Factor0_lat, axis=0))
Factor1_half_decod = decoder_ae.predict(np.expand_dims(Factor1_lat, axis=0))
Factor2_half_decod = decoder_ae.predict(np.expand_dims(Factor2_lat, axis=0))
plt.scatter(Factor0_half_decod,Factor1_half_decod)
plt.scatter(Factor0_half_decod,Factor2_half_decod)
plt.scatter(Factor1_half_decod,Factor2_half_decod)
