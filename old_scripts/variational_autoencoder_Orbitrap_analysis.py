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
from sklearn.preprocessing import RobustScaler, StandardScaler,FunctionTransformer,MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
import scipy.cluster.hierarchy as sch
import tensorflow as tf
import keras
from tensorflow.keras import layers
import os
os.chdir("C:/Work/Python/Github/Orbitrap_clustering")
from ae_functions import *

#from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib
from keras import backend as K 
from dateutil import parser
import math
from joblib import dump, load
import datetime
# %%
#A class for chemical formula
class chemform:
  def __init__(self, formula):
    #fiddle with the string so you can get the number of each element out, including 1 and 0
    formula = formula + " "
    formula = formula.replace(" ","1")
    formula = "0" + formula
    
    self.C = int(formula[formula.find("C")+1])
    self.H = int(formula[formula.find("H")+1])
    self.O = int(formula[formula.find("O")+1])
    self.N = int(formula[formula.find("N")+1])
    self.S = int(formula[formula.find("S")+1])
    

# #Take a string and work out the chemical formula, then return true or false if it's good or bad   
def filter_by_chemform(formula):
    chemformula = chemform(formula)
    if(chemformula.S >= 1 and chemformula.N >= 1 and chemformula.O > chemformula.C*7):
        return False
    elif(chemformula.S >= 1 and chemformula.N == 0 and chemformula.O > chemformula.C*4):
        return False
    elif(chemformula.N >= 1 and chemformula.S == 0 and chemformula.O > chemformula.C*3):
        return False
    elif(chemformula.N == 0 and chemformula.S == 0 and chemformula.O > chemformula.C*3.5):
        return False
    elif(chemformula.H > chemformula.C*3):
        return False
    else:
        return True
    

# %%
path='C:/Users/mbcx5jt5/Google Drive/Shared_York_Man2/'

# %%
df_beijing_peaks1, df_beijing_filters1, df_beijing_metadata1 = beijing_load(
    path + 'BJ_UnAmbNeg9.1.1_20210505-Times_Fixed.xlsx',path + 'BJ_UnAmbNeg9.1.1_20210505-Times_Fixed.xlsx',
    peaks_sheetname="Compounds",metadata_sheetname="massloading_Beijing")



df_delhi_peaks1, df_delhi_filters1, df_delhi_metadata1 = delhi_load(
    path + 'Delhi_Amb3.1_MZ.xlsx',path + 'Delhi/Delhi_massloading_autumn_summer.xlsx')


# %%



#Load Delhi data and remove columns that are not needed
df_delhi_peaks = pd.read_excel(path + 'Delhi_Amb3.1_MZ.xlsx',engine='openpyxl')
df_delhi_peaks.drop(df_delhi_peaks.iloc[:,np.r_[0, 2:11, 14:18]],axis=1,inplace=True)

df_delhi_metadata = pd.read_excel(path + 'Delhi/Delhi_massloading_autumn_summer.xlsx',engine='openpyxl',
                                   sheet_name='autumn',usecols='a:N',skiprows=0,nrows=108, converters={'mid_datetime': str})
df_delhi_metadata.drop(labels="Filter ID.1",axis=1,inplace=True)
df_delhi_metadata.set_index("Filter ID",inplace=True)
df_delhi_metadata.drop(labels=["-","S25","S42","S51","S55","S68","S72"],axis=0,inplace=True)



# %%
df_beijing_peaks.drop(df_beijing_peaks[df_beijing_peaks["Formula"] == "C12 H26 O3 S"].index,inplace=True)
df_delhi_peaks.drop(df_delhi_peaks[df_delhi_peaks["Formula"] == "C12 H26 O3 S"].index,inplace=True)
# %%
#Filter out peaks with strange formula
df_delhi_peaks = df_delhi_peaks[df_delhi_peaks["Formula"].apply(lambda x: filter_by_chemform(x))]
df_beijing_peaks = df_beijing_peaks[df_beijing_peaks["Formula"].apply(lambda x: filter_by_chemform(x))]

    
    
    
# %%
#Merge compound peaks that have the same m/z and retention time
#Round m/z to nearest integer and RT to nearest 2, as in 1/3/5/7/9 etc
#Also remove anything with RT > 20min

df_delhi_peaks.drop(df_delhi_peaks[df_delhi_peaks["RT [min]"] > 20].index, inplace=True)
# %%

def custom_round(x, base=5):
    return int(base * round(float(x)/base))

#round to nearest odd number
def round_odd(x):
    return (2*math.floor(x/2)+1)


RT_round =  df_delhi_peaks["RT [min]"].apply(lambda x: round_odd(x))
mz_round = df_delhi_peaks["m/z"].apply(lambda x: round(x, 3))

#Join the peaks with the same rounded m/z and RT
df_delhi_peaks = df_delhi_peaks.iloc[:,np.r_[0:4]].groupby([mz_round,RT_round]).aggregate("first").join(df_delhi_peaks.iloc[:,np.r_[4:len(df_delhi_peaks.columns)]].groupby([mz_round,RT_round]).aggregate("sum") )

RT_round =  df_beijing_peaks["RT [min]"].apply(lambda x: round_odd(x))
mz_round = df_beijing_peaks["m/z"].apply(lambda x: round(x, 3))
df_beijing_peaks = df_beijing_peaks.iloc[:,np.r_[0:4]].groupby([mz_round,RT_round]).aggregate("first").join(df_beijing_peaks.iloc[:,np.r_[4:len(df_beijing_peaks.columns)]].groupby([mz_round,RT_round]).aggregate("sum") )

#aggregation_functions = {'price': 'sum', 'amount': 'sum', 'name': 'first'}
#df_new = df.groupby(df['id']).aggregate(aggregation_functions)

# %%#
#combine the delhi with beijing data somehow
#combined_df=pd.merge(frame_traff,frame_aq, left_index=True, right_index=True)


# %%
#Line up everything by the sample ID
df_beijing_filters = df_beijing_peaks.iloc[:,list(range(4,len(df_beijing_peaks.columns)))].copy()

sample_id = df_beijing_filters.columns.str.split('_|.raw').str[2]
df_beijing_filters.columns = sample_id

df_beijing_filters = df_beijing_filters.transpose()
df_beijing_filters["Sample.ID"] = sample_id

#These two lines were there when I was using line in excel spreadsheet as the compound index
#df_beijing_filters.columns.rename("compound_num",inplace=True)
#df_beijing_filters.columns = df_beijing_filters.columns.astype('str')

#Check for NaNs
df_beijing_filters.isna().sum().sum()



#Add on the metadata
#This gives a warning but it's fine
df_beijing_merged = pd.merge(df_beijing_filters,df_beijing_metadata,on="Sample.ID",how='inner')
#Divide the data columns by the sample volume and multiply by the dilution liquid volume (pinonic acid)
#df_beijing_filters = df_beijing_merged.iloc[:,0:3783].div(df_beijing_merged['Volume_m3'], axis=0).mul(df_beijing_merged['Dilution_mL'], axis=0)
df_beijing_filters = df_beijing_merged.iloc[:,1:len(df_beijing_filters.columns)].div(df_beijing_merged['Volume_m3'], axis=0).mul(df_beijing_merged['Dilution_mL'], axis=0)
#df_beijing_filters.columns.rename("compound_num",inplace=True)

#JUST THE WINTER DATA
#df_beijing_filters = df_beijing_filters.iloc[0:124].copy()

#JUST THE SUMMER DATA
#df_beijing_filters = df_beijing_filters.iloc[124:].copy()


df_beijing_filters['mid_datetime'] = pd.to_datetime(df_beijing_merged['mid_datetime'],yearfirst=True)
df_beijing_filters.set_index('mid_datetime',inplace=True)


df_beijing_filters = df_beijing_filters.astype(float)

# %% Subtract blank sample data
df_beijing_filters = df_beijing_filters - df_beijing_filters.iloc[-1]
df_beijing_filters.drop(df_beijing_filters.tail(1).index,inplace=True)



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
Cluster_A = df_beijing_filters.iloc[89].ravel() #Q5 photochemical age, Timestamp('2017-06-11 09:02:30')
Cluster_B = df_beijing_filters.iloc[158].ravel() #Q50 photochemical age, Timestamp('2017-06-20 16:43:00')
Cluster_C = df_beijing_filters.iloc[106].ravel()    #Q95 photochemical age, Timestamp('2017-06-12 15:56:30')






#Normalise all to 1
Cluster_A = 1 * Cluster_A / Cluster_A.sum()
Cluster_B = 1 * Cluster_B / Cluster_B.sum()
Cluster_C = 1 * Cluster_C / Cluster_C.sum()


#For just the winter data, when it's normalised
amp_A = np.append(np.zeros(83),np.arange(0,1,0.015)) * 5
amp_B = np.ones(150)*2.75
amp_C = np.append(np.arange(1,0,-0.015),np.zeros(83)) * 5


df_clustA = pd.DataFrame((np.random.normal(1, 0.01, [150,num_cols])) * Cluster_A).multiply(amp_A,axis=0)
df_clustB = pd.DataFrame((np.random.normal(1, 0.01, [150,num_cols])) * Cluster_B).multiply(amp_B,axis=0)
df_clustC = pd.DataFrame((np.random.normal(1, 0.01, [150,num_cols])) * Cluster_C).multiply(amp_C,axis=0)
df_clustA.columns = df_beijing_filters.columns
df_clustB.columns = df_beijing_filters.columns
df_clustC.columns = df_beijing_filters.columns

clustA_total = df_clustA.sum(axis=1)
clustB_total = df_clustB.sum(axis=1)
clustC_total = df_clustC.sum(axis=1)


df_3clust_sum = pd.DataFrame([clustA_total,clustB_total,clustC_total]).T
df_3clust_sum.idxmax(axis="columns").plot()
plt.title("Input test cluster labels")
plt.show()

#df_3clust = df_clustA + df_clustB + df_clustC

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
    
    signoise = np.random.normal(1, sig_noise_pct/100, [num_copies*num_rows,num_cols])
    newdf = newdf * signoise
       


    #Efficient version
    #newdf = pd.DataFrame(np.repeat(df.values,num_copies,axis=0)) * np.random.normal(1, sig_noise_pct/100, [num_copies*num_rows,num_cols])
    return newdf
# %%

#AUGMENT THE DATA
#df_aug = augment_data_noise(df_3clust,100,0,1)

#Don't augment the data
#df_aug = augment_data_noise(df_3clust,100,3,0)
df_aug = augment_data_noise(df_beijing_filters,50,3,0)

# %%

#pipe = Pipeline([('function_transformer', FunctionTransformer(np.log1p, validate=True,inverse_func=np.expm1)), 
#                 ('robust_scalar', RobustScaler())])

#pipe = Pipeline([('robust_scalar', RobustScaler())])
pipe = Pipeline([('standard_scalar', StandardScaler())])

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
#scaled_df_val = pd.DataFrame(pipe.fit_transform(df_3clust), columns=df_3clust.columns,index=df_3clust.index)
scaled_df_val = pd.DataFrame(pipe.transform(df_beijing_filters), columns=df_beijing_filters.columns,index=df_beijing_filters.index)
ae_input_val = scaled_df_val.to_numpy()
#Inverse of scaling
#df_2clust_inverse = scalery.inverse_transform(scaled_df)

# Now extract all of the data as a scaled array
ae_input=scaled_df.to_numpy()
np.count_nonzero(np.isnan(ae_input))  #No NaNs which is good
ae_input.shape

#Linear space 2-cluster data
plt.pcolormesh(ae_input_val)
plt.title("Linear space 2-cluster ae_input")
plt.show()
#Log-space 2-cluster data
plt.pcolormesh(ae_input_val,norm=matplotlib.colors.LogNorm())
plt.title("Log space 2-cluster ae_input")
plt.show()


# %%


############################################################
#AUTOENCODER DEINITION
############################################################
# original_dim = len(df_beijing_filters.columns)
# #intermediate_dim1_vae = original_dim//2
# #intermediate_dim2_vae = original_dim//5
# #intermediate_dim3_vae = original_dim//10
# #latent_dim_vae = 8
# # - uncomment the following if using a Variational autoencoder. 
# class Sampling(layers.Layer):
#     """Uses (z_mean, z_log_var) to sample z."""#

#     def call(self, inputs):
#         z_mean, z_log_var = inputs
#         batch = tf.shape(z_mean)[0]
#         dim = tf.shape(z_mean)[1]
#         epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
#         return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# def model_builder(hp):

#     intermediate_dim1_hp_units = hp.Int('units1', min_value=420, max_value=840, step=30)
#     intermediate_dim2_hp_units = hp.Int('units2', min_value=140, max_value=280, step=20)
#     intermediate_dim3_hp_units = hp.Int('units3', min_value=45, max_value=95, step=10)
#     intermediate_dim4_hp_units = hp.Int('units4', min_value=15, max_value=30, step=3)
#     latent_dim_units = hp.Int('latent_units', min_value=2, max_value=10, step=1)

#     original_inputs = keras.Input(shape=(original_dim,), name="encoder_input")
#     layer1_vae = layers.Dense(intermediate_dim1_hp_units, activation="relu")(original_inputs)
#     layer2_vae = layers.Dense(intermediate_dim2_hp_units, activation="relu")(layer1_vae)
#     layer3_vae = layers.Dense(intermediate_dim3_hp_units, activation="relu")(layer2_vae)
#     layer4_vae = layers.Dense(intermediate_dim4_hp_units, activation="relu")(layer3_vae)
#     layer5_vae = layers.Dense(latent_dim_units, activation="relu")(layer4_vae)
#     z_mean = layers.Dense(latent_dim_units, name="z_mean")(layer5_vae)
#     z_log_sigma = layers.Dense(latent_dim_units, name="z_log_sigma")(layer5_vae)
#     z = keras.layers.Lambda(Sampling, output_shape=(latent_dim_units,))([z_mean, z_log_sigma])
#     #z = Sampling()((z_mean, z_log_sigma))
#     #encoder_ae = keras.Model(inputs=original_inputs, outputs=layer4_vae, name="encoder_ae")
#     encoder_vae = keras.Model(original_inputs, [z_mean, z_log_sigma, z], name='encoder_vae')

#     # Define decoder model.
#     latent_inputs_ae = keras.Input(shape=(latent_dim_units,), name="z_sampling")
#     dec_layer1_vae = layers.Dense(intermediate_dim4_hp_units, activation="relu")(latent_inputs_ae)
#     dec_layer2_vae = layers.Dense(intermediate_dim3_hp_units, activation="relu")(dec_layer1_vae)
#     dec_layer3_vae = layers.Dense(intermediate_dim2_hp_units, activation="relu")(dec_layer2_vae)
#     dec_layer4_vae = layers.Dense(intermediate_dim1_hp_units, activation="relu")(dec_layer3_vae)
#     outputs_vae = layers.Dense(original_dim, activation="linear")(dec_layer4_vae)
#     decoder_ae = keras.Model(inputs=latent_inputs_ae, outputs=outputs_vae, name="decoder_vae")



#   # #MY CODE EXPERIMENTING WITH LOSSES
#   # # Define VAE model.
#   # outputs = decoder_ae(z)
#   # ae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name="vae")
#   # # Add KL divergence regularization loss.
#   # kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
#   # #ae.add_loss(kl_loss)
#   # #ae.add_loss(tf.keras.losses.MeanSquaredError())
  
#   # mse_loss = tf.reduce_mean(tf.keras.metrics.mean_squared_error(original_inputs,outputs))
#   # vae_loss2 = K.mean(mse_loss + kl_loss)
#   # ae.add_loss(vae_loss2)

#   # hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])


#   # optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
#   # #ae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
#   # ae.compile(optimizer)
  
  
#   ##DAVE'S ORIGINAL CODE
#   # Define VAE model.

#     outputs = decoder_ae(z)
#     ae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name="vae")
#     # Add KL divergence regularization loss.
#     kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)

#     ae.add_loss(kl_loss)
#     hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])

#     optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
#     ae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
    
#     return ae


# #Define loss function
# # Calculate custom loss in separate function
# def vae_loss(original_inputs, outputs):
#     mse_loss = tf.keras.metrics.mean_squared_error(original_inputs,outputs)
#     #xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
#     kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
#     kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
#     vae_loss = K.mean(mse_loss + kl_loss)
#     return vae_loss

# # %%
# ############################
# ###DAVE'S ORIGINAL VAE CODE
# ############################
# original_dim = len(df_beijing_filters.columns)
# #intermediate_dim1_vae = original_dim//2
# #intermediate_dim2_vae = original_dim//5
# #intermediate_dim3_vae = original_dim//10
# #latent_dim_vae = 8

# # - uncomment the following if using a Variational autoencoder. 
# class Sampling(layers.Layer):
#     """Uses (z_mean, z_log_var) to sample z."""#

#     def call(self, inputs):
#         z_mean, z_log_var = inputs
#         batch = tf.shape(z_mean)[0]
#         dim = tf.shape(z_mean)[1]
#         epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
#         return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# def model_builder(hp):

#     #DAVE'S ORIGINAL VERSION
#     intermediate_dim1_hp_units = hp.Int('units1', min_value=400, max_value=800, step=20)
#     intermediate_dim2_hp_units = hp.Int('units2', min_value=200, max_value=400, step=20)
#     intermediate_dim3_hp_units = hp.Int('units3', min_value=100, max_value=200, step=10)
#     latent_dim_units = hp.Int('latent_units', min_value=10, max_value=100, step=5)

#     original_inputs = tf.keras.Input(shape=(original_dim,), name="encoder_input")
#     layer1_vae = layers.Dense(intermediate_dim1_hp_units, activation="relu")(original_inputs)
#     layer2_vae = layers.Dense(intermediate_dim2_hp_units, activation="relu")(layer1_vae)
#     layer3_vae = layers.Dense(intermediate_dim3_hp_units, activation="relu")(layer2_vae)
#     #layer4_vae = layers.Dense(latent_dim_units, activation="relu")(layer3_vae)
#     z_mean = layers.Dense(latent_dim_units, name="z_mean")(layer3_vae)
#     z_log_var = layers.Dense(latent_dim_units, name="z_log_var")(layer3_vae)
#     z = Sampling()((z_mean, z_log_var))
#     encoder_ae = tf.keras.Model(inputs=original_inputs, outputs=z, name="encoder_ae")

#     # Define decoder model.
#     latent_inputs_ae = tf.keras.Input(shape=(latent_dim_units,), name="z_sampling")
#     dec_layer1_vae = layers.Dense(intermediate_dim3_hp_units, activation="relu")(latent_inputs_ae)
#     dec_layer2_vae = layers.Dense(intermediate_dim2_hp_units, activation="relu")(dec_layer1_vae)
#     dec_layer3_vae = layers.Dense(intermediate_dim1_hp_units, activation="relu")(dec_layer2_vae)
#     outputs_vae = layers.Dense(original_dim, activation="linear")(dec_layer3_vae)
#     decoder_ae = tf.keras.Model(inputs=latent_inputs_ae, outputs=outputs_vae, name="decoder_vae")
    
    
# # #MY VERSION NOT WORING LOSS ISALWAYS 1
# #   intermediate_dim1_hp_units = hp.Int('units1', min_value=420, max_value=840, step=30)
# #   intermediate_dim2_hp_units = hp.Int('units2', min_value=140, max_value=280, step=20)
# #   intermediate_dim3_hp_units = hp.Int('units3', min_value=45, max_value=95, step=10)
# #   intermediate_dim4_hp_units = hp.Int('units4', min_value=15, max_value=30, step=3)
# #   latent_dim_units = hp.Int('latent_units', min_value=2, max_value=10, step=1)

# #   original_inputs = tf.keras.Input(shape=(original_dim,), name="encoder_input")
# #   layer1_vae = layers.Dense(intermediate_dim1_hp_units, activation="relu")(original_inputs)
# #   layer2_vae = layers.Dense(intermediate_dim2_hp_units, activation="relu")(layer1_vae)
# #   layer3_vae = layers.Dense(intermediate_dim3_hp_units, activation="relu")(layer2_vae)
# #   layer4_vae = layers.Dense(intermediate_dim4_hp_units, activation="relu")(layer3_vae)
# #   #layer4_vae = layers.Dense(latent_dim_units, activation="relu")(layer3_vae)
# #   z_mean = layers.Dense(latent_dim_units, name="z_mean")(layer4_vae)
# #   z_log_var = layers.Dense(latent_dim_units, name="z_log_var")(layer4_vae)
# #   z = Sampling()((z_mean, z_log_var))
# #   encoder_ae = tf.keras.Model(inputs=original_inputs, outputs=z, name="encoder_ae")

# #    # Define decoder model.
# #   latent_inputs_ae = tf.keras.Input(shape=(latent_dim_units,), name="z_sampling")
# #   dec_layer1_vae = layers.Dense(intermediate_dim4_hp_units, activation="relu")(latent_inputs_ae)
# #   dec_layer2_vae = layers.Dense(intermediate_dim2_hp_units, activation="relu")(dec_layer1_vae)
# #   dec_layer3_vae = layers.Dense(intermediate_dim1_hp_units, activation="relu")(dec_layer2_vae)
# #   dec_layer4_vae = layers.Dense(intermediate_dim1_hp_units, activation="relu")(dec_layer3_vae)
# #   outputs_vae = layers.Dense(original_dim, activation="linear")(dec_layer4_vae)
# #   decoder_ae = tf.keras.Model(inputs=latent_inputs_ae, outputs=outputs_vae, name="decoder_vae")
  


#     # Define VAE model.
#     outputs = decoder_ae(z)
#     ae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name="vae")
#     # Add KL divergence regularization loss.
#     kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
#     ae.add_loss(kl_loss)

#     hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])

#     optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
#     ae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())

#     return ae

# %% The regular AE (not VAE)
############################################################
#AUTOENCODER DEINITION
############################################################
original_dim = len(df_beijing_filters.columns)
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
    layer4_vae = layers.Dense(latent_dim_units, activation="relu")(layer3_vae)
    
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
    ae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())

    return ae



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
history = model.fit(ae_input, ae_input, epochs=30, validation_data=(ae_input_val, ae_input_val))
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
original_inputs = tf.keras.Input(shape=(original_dim,), name="encoder_input")
layer1_vae = layers.Dense(best_hps.get('units1'), activation="relu")(original_inputs)
layer2_vae = layers.Dense(best_hps.get('units2'), activation="relu")(layer1_vae)
layer3_vae = layers.Dense(best_hps.get('units3'), activation="relu")(layer2_vae)
layer4_vae = layers.Dense(best_hps.get('units4'), activation="relu")(layer3_vae)
#layer4_vae = layers.Dense(best_hps.get('latent_units'), activation="relu")(layer3_vae)
z_mean = layers.Dense(best_hps.get('latent_units'), name="z_mean")(layer3_vae)
z_log_var = layers.Dense(best_hps.get('latent_units'), name="z_log_var")(layer3_vae)
z = Sampling()((z_mean, z_log_var))
encoder_ae = tf.keras.Model(inputs=original_inputs, outputs=z, name="encoder_ae")

# Define decoder model.
latent_inputs_ae = tf.keras.Input(shape=(best_hps.get('latent_units'),), name="decoder_input")
dec_layer1_vae = layers.Dense(best_hps.get('units4'), activation="relu")(latent_inputs_ae)
dec_layer2_vae = layers.Dense(best_hps.get('units3'), activation="relu")(dec_layer1_vae)
dec_layer3_vae = layers.Dense(best_hps.get('units2'), activation="relu")(dec_layer2_vae)
dec_layer4_vae = layers.Dense(best_hps.get('units1'), activation="relu")(dec_layer3_vae)
outputs_ae = layers.Dense(original_dim, activation="linear")(dec_layer4_vae)
decoder_ae = tf.keras.Model(inputs=latent_inputs_ae, outputs=outputs_ae, name="decoder_ae")

# Define VAE model.
outputs = decoder_ae(z)
ae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name="ae")

# Add KL divergence regularization loss.
kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
ae.add_loss(kl_loss)

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

## Call the now re-trained encoder part of our model
latent_space = encoder_ae.predict(ae_input_val)
latent_space.shape
df_latent_space = pd.DataFrame(latent_space)

# %%

#How good is our encode/decode?
#This is the really key plot. It's shit!! So need to improve it I think, otherwise it's not useful for getting the factors out of the latent space
#And if you don't know what the factors out, then you can't use them
#plt.scatter(ae_input_val,ae.predict(ae_input_val))
plt.scatter(df_beijing_filters.values,pipe.inverse_transform(ae.predict(ae_input_val)))
plt.xlabel("AE input")
plt.ylabel("AE output")

# %%
#Is the sum total linear with input vs output?
plt.scatter(df_3clust.sum(axis=1),pipe.inverse_transform(ae.predict(ae_input_val)).sum(axis=1),c=df_3clust.index)
plt.xlabel("AE input")
plt.ylabel("AE output")


# %%
#What does the latent space look like?
plt.pcolormesh(latent_space)

# %%
#What does one column of the latent space look like in real space?

decoded_col0 = pipe.inverse_transform(decoder_ae.predict([[1,0,0,0,0,0,0,0,0,0]]))
decoded_col1 = pipe.inverse_transform(decoder_ae.predict([[0,1,0,0,0,0,0,0,0,0]]))

decoded_col9 = pipe.inverse_transform(decoder_ae.predict([[0,0,0,0,0,0,0,0,0,9]]))

plt.plot(decoded_col1)
plt.show()
















# %%
#Principal component analysis
pca = PCA(n_components = 3)
prin_comp = pca.fit_transform(latent_space)
plt.pcolormesh(prin_comp)

# %%

#loss, accuracy, f1_score, precision, recall = ae.evaluate(ae_input_val, ae_input_val, verbose=0)
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
#clustering = agglom.fit(ae_input_val)
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

latent_space = latent_space + 5
W = model.fit_transform(latent_space)

#W = model.fit_transform(latent_space+latent_space.min())#To make sure it's positive
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
