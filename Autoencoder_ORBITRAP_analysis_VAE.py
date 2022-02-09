# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(1337) # for reproducibility
import tensorflow as tf
tf.random.set_seed(69)
import keras
import kerastuner as kt
#from google.colab import drive
import pandas as pd
import glob
import pdb
from sklearn.preprocessing import RobustScaler, StandardScaler,FunctionTransformer
from sklearn.pipeline import Pipeline
#import tensorflow as tf
import keras
#from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib
import os
from keras import backend as K 

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


############################################################
#AUTOENCODER DEINITION
#Based on keras tutorial
#https://blog.keras.io/building-autoencoders-in-keras.html
############################################################
original_dim = len(df_beijing_filters.columns)
#batch_size = 30
#intermediate_dim1_vae = original_dim//2
#intermediate_dim2_vae = original_dim//5
#intermediate_dim3_vae = original_dim//10
#latent_dim_vae = 8

#- uncomment the following if using a Variational autoencoder. 
def sampling(args):
    z_mean, z_log_sigma = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    #epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
     #                         stddev=epsilon_std)
    epsilon = keras.backend.random_normal(shape=(batch, dim)) 
    return z_mean + K.exp(z_log_sigma / 2) * epsilon
def model_builder(hp):

    # intermediate_dim1_hp_units = hp.Int('units1', min_value=400, max_value=800, step=20)
    # intermediate_dim2_hp_units = hp.Int('units2', min_value=200, max_value=400, step=20)
    # intermediate_dim3_hp_units = hp.Int('units3', min_value=100, max_value=200, step=10)
    # latent_dim_units = hp.Int('latent_units', min_value=10, max_value=100, step=5)
    intermediate_dim1_hp_units = hp.Int('units1', min_value=800, max_value=1600, step=40)
    intermediate_dim2_hp_units = hp.Int('units2', min_value=400, max_value=800, step=40)
    intermediate_dim3_hp_units = hp.Int('units3', min_value=200, max_value=400, step=20)
    latent_dim_units = hp.Int('latent_units', min_value=10, max_value=200, step=10)

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
    #Not quite sure what this [2] is, it came from keras tutorial
    outputs = decoder_ae(encoder_vae(original_inputs)[2])
    vae = keras.Model(inputs=original_inputs, outputs=outputs, name="vae")
    # Add KL divergence regularization loss.
    kl_loss = -0.5 * tf.reduce_mean(z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma) + 1)
    vae.add_loss(kl_loss)
    #vae.add_loss(keras.losses.MeanSquaredError(original_inputs, outputs))#test
  
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])



    # #Custom loss term like in keras tutorial
    # reconstruction_loss = keras.losses.MeanSquaredError(original_inputs, outputs)
    # #reconstruction_loss *= original_dim
    # kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    # kl_loss = K.sum(kl_loss, axis=-1)
    # kl_loss *= -0.5
    # vae_loss = K.mean(reconstruction_loss + kl_loss)
    # vae.add_loss(vae_loss)
    optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)
    vae.compile(optimizer=optimizer)


    #optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)
    #ae.compile(optimizer, loss=f.keras.losses.MeanSquaredError())

    return vae




# %%
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

    # %%
# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units1')}, the second {best_hps.get('units2')}, third {best_hps.get('units3')}, latent {best_hps.get('latent_units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# %%



# Build the model with the optimal hyperparameters and train it on the data for 30 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(ae_input, ae_input, epochs=50, validation_data=(ae_input_val, ae_input_val))
val_acc_per_epoch = history.history['val_loss']
best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

# %%

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

    # %%

#########################################################################################
# Perform cluster analysis
#Now that we have our optimised AE, we can predict the latent space.
#To do that, we need to rebuild our autoencoder since we cannot access the encoder part of our model.
#However, not to worry since we have recorded the best set of parameters.
#In the following code block we retrain the autoencoder and then we are going to extract the output from only
#the encoder model and then use this for cluster analysis. 

# Define encoder model.
original_inputs = keras.Input(shape=(original_dim,), name="encoder_input")
layer1_vae = layers.Dense(best_hps.get('units1'), activation="relu")(original_inputs)
layer2_vae = layers.Dense(best_hps.get('units2'), activation="relu")(layer1_vae)
layer3_vae = layers.Dense(best_hps.get('units3'), activation="relu")(layer2_vae)
layer4_vae = layers.Dense(best_hps.get('latent_units'), activation="relu")(layer3_vae)
#z_mean = layers.Dense(best_hps.get('latent_units'), name="z_mean")(layer3_vae)
#z_log_sigma = layers.Dense(best_hps.get('latent_units'), name="z_log_sigma")(layer3_vae)
#z = Sampling()((z_mean, z_log_sigma))
encoder_ae = keras.Model(inputs=original_inputs, outputs=layer4_vae, name="encoder_ae")

# Define decoder model.
latent_inputs_ae = keras.Input(shape=(best_hps.get('latent_units'),), name="decoder_input")
dec_layer1_vae = layers.Dense(best_hps.get('units3'), activation="relu")(latent_inputs_ae)
dec_layer2_vae = layers.Dense(best_hps.get('units2'), activation="relu")(dec_layer1_vae)
dec_layer3_vae = layers.Dense(best_hps.get('units1'), activation="relu")(dec_layer2_vae)
outputs_ae = layers.Dense(original_dim, activation="linear")(dec_layer3_vae)
decoder_ae = keras.Model(inputs=latent_inputs_ae, outputs=outputs_ae, name="decoder_ae")

# Define VAE model.
outputs = decoder_ae(layer4_vae)
ae = keras.Model(inputs=original_inputs, outputs=outputs, name="ae")

# Add KL divergence regularization loss.
#kl_loss = -0.5 * tf.reduce_mean(z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma) + 1)
#vae.add_loss(kl_loss)
# Model summary
print(ae.summary())
#train_data,test_data,_,_ = train_test_split(vae_input,vae_input,test_size=0.2)
optimizer = keras.optimizers.Adam(learning_rate=best_hps.get('learning_rate'))
ae.compile(optimizer, loss=keras.losses.MeanSquaredError())
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

## Call the now re-trained encoder part of our model
latent_space = encoder_ae.predict(ae_input_val)
latent_space.shape
df_latent_space = pd.DataFrame(latent_space)

# %%

#How good is our encode/decode?
#This is the really key plot. It's shit!! So need to improve it I think, otherwise it's not useful for getting the factors out of the latent space
#And if you don't know what the factors out, then you can't use them
#plt.scatter(ae_input_val,ae.predict(ae_input_val))
plt.scatter(df_3clust.values,pipe.inverse_transform(ae.predict(ae_input_val)))
plt.xlabel("AE input")
plt.ylabel("AE output")

# %%
#Is the sum total linear with input vs output?
plt.scatter(df_3clust.sum(axis=1),pipe.inverse_transform(ae.predict(ae_input_val)).sum(axis=1),c=df_3clust.index)
plt.xlabel("AE input")
plt.ylabel("AE output")


# %%

loss, accuracy, f1_score, precision, recall = ae.evaluate(ae_input_val, ae_input_val, verbose=0)
# %%


############################################################################################
#CLUSTERING AND FACTOR ANALYSIS OF LATENT SPACE DATA
############################################################################################

#Lets try a dendrogram to work out the optimal number of clusters
import scipy.cluster.hierarchy as sch
fig,axes = plt.subplots(1,1,figsize=(20,10))
plt.title("dendrogram")
dendrogram = sch.dendrogram(sch.linkage(latent_space, method='ward'))
plt.show()
#It comes out with 2 clusters for the 2 cluster solution so that's good

# Now we can perform agglomerative cluster analysis on this new ouput, assigning each row to a particular cluster. We have to define the number of clusters but thats ok for now
from sklearn.cluster import AgglomerativeClustering 
agglom = AgglomerativeClustering(n_clusters = 3, linkage = 'ward')
clustering = agglom.fit(latent_space)
plt.plot(clustering.labels_)
#plt.plot(df_beijing_filters.index,clustering.labels_)

#And what are the clusters?
#The cluster labels can just be moved straight out the latent space
cluster0_decoded = df_3clust[clustering.labels_==0].mean()
cluster1_decoded = df_3clust[clustering.labels_==1].mean()
cluster2_decoded = df_3clust[clustering.labels_==2].mean()

#Latent space clusters
cluster0_lat = df_latent_space[clustering.labels_==0].mean()
cluster1_lat = df_latent_space[clustering.labels_==1].mean()
cluster2_lat = df_latent_space[clustering.labels_==2].mean()

# %%

#Compare to the input clusters, should be almost identical
#They are! Awesome!
plt.figure()
plt.scatter(Cluster_A,cluster0_decoded)
#plt.scatter(max_filter,cluster0_decoded)
plt.xlabel("Cluster A")
plt.xlabel("Cluster 0 output")
plt.show()

plt.figure()
#plt.scatter(Cluster_C,cluster1_decoded)
plt.scatter(Cluster_A,cluster1_decoded)
plt.xlabel("Min filter")
plt.xlabel("Cluster 2 output")
plt.show()


plt.figure()
#plt.scatter(Cluster_C,cluster2_decoded)
plt.scatter(Cluster_A,cluster2_decoded)
plt.xlabel("Thou filter")
plt.xlabel("Cluster 1 output")
plt.show()

# %%

#Now lets decode the clusters from latent space and see if it still works
#Cluster0_decod = transformer.inverse_transform(decoder_ae.predict(np.expand_dims(cluster0_lat, axis=0)))
#Cluster1_decod = transformer.inverse_transform(decoder_ae.predict(np.expand_dims(cluster1_lat, axis=0)))
#Cluster2_decod = transformer.inverse_transform(decoder_ae.predict(np.expand_dims(cluster2_lat, axis=0)))

Cluster0_decod = pipe.inverse_transform((decoder_ae.predict(np.expand_dims(cluster0_lat, axis=0))))
Cluster1_decod = pipe.inverse_transform((decoder_ae.predict(np.expand_dims(cluster1_lat, axis=0))))
Cluster2_decod = pipe.inverse_transform((decoder_ae.predict(np.expand_dims(cluster2_lat, axis=0))))

#THESE SHOULD ALWAYS ALWAYS BE A STRAIGH LINE OTHERWISE SOMEHTING IT NOT WORKING RIGHT IN THE AE OR TRANSFORMER
plt.scatter(cluster0_decoded,Cluster0_decod)

plt.scatter(cluster1_decoded,Cluster1_decod)

plt.scatter(cluster2_decoded,Cluster2_decod)

# %%
#Now do some NMF clustering
from sklearn.decomposition import NMF
model = NMF(n_components=3)
#model.fit(latent_space)

#Get the different components
#nmf_features = model.transform(latent_space)
#print(model.components_)

W = model.fit_transform(latent_space)
#W = model.fit_transform(df_3clust.values)
H = model.components_

#1 What is the time series of the 2 factors? Need each factor as a t series
Factor0_lat = H[0]
Factor1_lat = H[1]
Factor2_lat = H[2]

Factor0_lat_mtx = np.outer(W.T[0], H[0])
Factor1_lat_mtx = np.outer(W.T[1], H[1])
Factor2_lat_mtx = np.outer(W.T[2], H[2])
#Now need to decode these matrices to get the time series matrix of each factor

Factor0_decod = pipe.inverse_transform(decoder_ae.predict(np.expand_dims(Factor0_lat, axis=0)))
Factor1_decod = pipe.inverse_transform(decoder_ae.predict(np.expand_dims(Factor1_lat, axis=0)))
Factor2_decod = pipe.inverse_transform(decoder_ae.predict(np.expand_dims(Factor2_lat, axis=0)))

Factor0_mtx_decod = pipe.inverse_transform(decoder_ae.predict(Factor0_lat_mtx))
Factor1_mtx_decod = pipe.inverse_transform(decoder_ae.predict(Factor1_lat_mtx))
Factor2_mtx_decod = pipe.inverse_transform(decoder_ae.predict(Factor2_lat_mtx))

Factor0_decod_sum = Factor0_mtx_decod.sum(axis=1)
Factor1_decod_sum = Factor1_mtx_decod.sum(axis=1)
Factor2_decod_sum = Factor2_mtx_decod.sum(axis=1)

plt.plot(Factor0_decod_sum)
plt.plot(Factor1_decod_sum)
plt.plot(Factor2_decod_sum)
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
plt.scatter(Cluster_A,Factor0_decod)
#plt.scatter(min_filter,Factor1_decod)
plt.xlabel("Input peak height")
plt.ylabel("Output peak height")

#This one does not either
plt.scatter(Cluster_A,Factor1_decod)
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
