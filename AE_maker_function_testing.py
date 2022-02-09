# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:37:36 2021

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
from sklearn.preprocessing import RobustScaler, StandardScaler,FunctionTransformer,MinMaxScaler
from sklearn.pipeline import Pipeline

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


import os
os.chdir('C:/Work/Python/Github/Orbitrap_clustering')
from ae_functions import *

# %%Load data

path='C:/Users/mbcx5jt5/Google Drive/Shared_York_Man2/'
df_beijing_raw, df_beijing_filters, df_beijing_metadata = beijing_load(
    path + 'BJ_UnAmbNeg9.1.1_20210505-Times_Fixed.xlsx',path + 'BJ_UnAmbNeg9.1.1_20210505-Times_Fixed.xlsx',
    peaks_sheetname="Compounds",metadata_sheetname="massloading_Beijing")


#%%Scale data


#pipe = Pipeline([('function_transformer', FunctionTransformer(np.log1p, validate=True,inverse_func=np.expm1))])
scalefactor = 1e6
pipe = FunctionTransformer(lambda x: np.divide(x,scalefactor),inverse_func = lambda x: np.multiply(x,scalefactor))
#pipe = MinMaxScaler()
#pipe = StandardScaler()
#pipe = RobustScaler()
#pipe = MaxAbsScaler()
pipe.fit(df_beijing_filters.to_numpy())
df_beijing_scaled = pd.DataFrame(pipe.transform(df_beijing_filters.to_numpy()))
df_beijing_scaled = df_beijing_scaled.fillna(0)
ae_input=df_beijing_scaled.to_numpy()

#For testing
ae_input_val = ae_input

#%%

# #%%Model builder for 2 layer AE
# class kt_model_builder_2int_AE(kt.HyperModel):

#     def __init__(self,input_dim,latent_dim='DEFAULT'):
#         self.input_dim = input_dim
#         self.latent_dim = latent_dim               

#     def build(self,hp):
#         #Define hyperparameters to scan
#         hp_input_dim = hp.Fixed('input_dim', self.input_dim)
#         if(self.latent_dim=='DEFAULT'):
#             hp_latent_dim = hp.Int('latent_dim', min_value=5, max_value=50, step=5)
#         else:
#             hp_latent_dim = hp.Fixed('latent_dim', self.latent_dim)
        
#         hp_intermediate_dim1 = hp.Int('intermediate_dim1', min_value=250, max_value=500, step=50)
#         hp_intermediate_dim2 = hp.Int('intermediate_dim2', min_value=50, max_value=200, step=30)
        
#         hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
        
#         hp_output_activation = hp.Choice('decoder_output_activation',values=['linear','sigmoid'])

#         #Define encoder model
#         encoder_input_layer = layers.Input(shape=(hp_input_dim,), name="encoder_input_layer")
#         encoder_layer1 = layers.Dense(hp_intermediate_dim1, activation="relu",name='intermediate_layer1')(encoder_input_layer)
#         encoder_layer2 = layers.Dense(hp_intermediate_dim2, activation="relu",name='intermediate_layer2')(encoder_layer1)    
#         latent_layer = layers.Dense(hp_latent_dim, activation="linear",name='latent_layer')(encoder_layer2)
        
#         #encoder_ae = Model(inputs=input_layer, outputs=latent_layer, name="encoder_ae")
            
#         # Define decoder model.
#         #latent_inputs_ae = tf.keras.Input(shape=(latent_dim_units,), name="decoder_input_layer")
#         decoder_layer1 = layers.Dense(hp_intermediate_dim2, activation="relu",name='decoder_layer1')(latent_layer)
#         decoder_layer2 = layers.Dense(hp_intermediate_dim1, activation="relu",name='decoder_layer2')(decoder_layer1)
#         decoder_output_layer = layers.Dense(hp_input_dim, activation=hp_output_activation,name='decoder_output_layer')(decoder_layer2)
#         #decoder_ae = Model(inputs=latent_inputs_ae, outputs=decoder_output_layer, name="decoder_ae")

#         # #Define VAE model.
#         # outputs = decoder_ae(layer4_vae)
#         # #outputs_vae = decoder(encoder(inputs)[2])
        
#         ae = Model(inputs=encoder_input_layer, outputs=decoder_output_layer, name="ae")        
#         optimizer = optimizers.Adam(learning_rate=hp_learning_rate)
       
#         #COMPILING
#         #Standard compilation
#         ae.compile(optimizer, loss='mse')
#         #Compile weighted to reduce number of columns
#         #ae.compile(optimizer, loss=['mean_squared_error'],loss_weights=[latent_dim_units])

#         return ae
    
# class kt_model_builder_3int_AE(kt.HyperModel):

#     def __init__(self,input_dim,latent_dim='DEFAULT'):
#         self.input_dim = input_dim
#         self.latent_dim = latent_dim               

#     def build(self,hp):
#         #Define hyperparameters to scan
#         hp_input_dim = hp.Fixed('input_dim', self.input_dim)
#         if(self.latent_dim=='DEFAULT'):
#             hp_latent_dim = hp.Int('latent_dim', min_value=5, max_value=50, step=5)
#         else:
#             hp_latent_dim = hp.Fixed('latent_dim', self.latent_dim)
        
#         hp_intermediate_dim1 = hp.Int('intermediate_dim1', min_value=250, max_value=500, step=50)
#         hp_intermediate_dim2 = hp.Int('intermediate_dim2', min_value=50, max_value=200, step=30)
#         hp_intermediate_dim3 = hp.Int('intermediate_dim3', min_value=20, max_value=50, step=10)
        
#         hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
        
#         hp_output_activation = hp.Choice('decoder_output_activation',values=['linear','sigmoid'])

#         #Define encoder model
#         encoder_input_layer = layers.Input(shape=(hp_input_dim,), name="encoder_input_layer")
#         encoder_layer1 = layers.Dense(hp_intermediate_dim1, activation="relu",name='intermediate_layer1')(encoder_input_layer)
#         encoder_layer2 = layers.Dense(hp_intermediate_dim2, activation="relu",name='intermediate_layer2')(encoder_layer1)    
#         encoder_layer3 = layers.Dense(hp_intermediate_dim3, activation="relu",name='intermediate_layer3')(encoder_layer2)
#         latent_layer = layers.Dense(hp_latent_dim, activation="linear",name='latent_layer')(encoder_layer3)
        
#         #encoder_ae = Model(inputs=input_layer, outputs=latent_layer, name="encoder_ae")
            
#         # Define decoder model.
#         #latent_inputs_ae = tf.keras.Input(shape=(latent_dim_units,), name="decoder_input_layer")
#         decoder_layer1 = layers.Dense(hp_intermediate_dim3, activation="relu",name='decoder_layer1')(latent_layer)
#         decoder_layer2 = layers.Dense(hp_intermediate_dim2, activation="relu",name='decoder_layer2')(decoder_layer1)
#         decoder_layer3 = layers.Dense(hp_intermediate_dim1, activation="relu",name='decoder_layer3')(decoder_layer2)
#         decoder_output_layer = layers.Dense(hp_input_dim, activation=hp_output_activation,name='decoder_output_layer')(decoder_layer3)
#         #decoder_ae = Model(inputs=latent_inputs_ae, outputs=decoder_output_layer, name="decoder_ae")

#         # #Define VAE model.
#         # outputs = decoder_ae(layer4_vae)
#         # #outputs_vae = decoder(encoder(inputs)[2])
        
#         ae = Model(inputs=encoder_input_layer, outputs=decoder_output_layer, name="ae")        
#         optimizer = optimizers.Adam(learning_rate=hp_learning_rate)
       
#         #COMPILING
#         #Standard compilation
#         ae.compile(optimizer, loss='mse')
#         #Compile weighted to reduce number of columns
#         #ae.compile(optimizer, loss=['mean_squared_error'],loss_weights=[latent_dim_units])

#         return ae
    
#%%
class kt_model_builder_AE_n_layer(kt.HyperModel):

    def __init__(self,input_dim,int_layers=2,latent_dim='DEFAULT'):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.int_layers = int_layers  
        self.int_layer_dims = []             

    def build(self,hp):
        #Define hyperparameters to scan
        hp_input_dim = hp.Fixed('input_dim', self.input_dim)
        if(self.latent_dim=='DEFAULT'):
            hp_latent_dim = hp.Int('latent_dim', min_value=5, max_value=20, step=5)
        else:
            hp_latent_dim = hp.Fixed('latent_dim', self.latent_dim)
        
        hp_int_layers = hp.Fixed('intermediate_layers',self.int_layers)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
        hp_output_activation = hp.Choice('decoder_output_activation',values=['linear','sigmoid'])    
        encoder_input_layer = layers.Input(shape=(hp_input_dim,), name="encoder_input_layer")

        #Work out logarithmically spaced encoder layers
        #The dimensions of the intermediate layers are stored in self.int_layer_dims
        #There are also fixed hyperparameters that are output for the layer dimensions
        self.int_layer_dims = []
        if(self.int_layers>0):
            layer1_dim_mid = (self.input_dim**self.int_layers *self.latent_dim)**(1/(self.int_layers+1))
            hp_layer1_dim_mid = hp.Fixed('intermediate_dim_1', round(layer1_dim_mid))
            encoder_layer = layers.Dense(round(layer1_dim_mid), activation="relu",name='intermediate_layer_1')(encoder_input_layer)
            self.int_layer_dims.append(round(layer1_dim_mid))
            if(self.int_layers>1):
                for int_layer in range(2,self.int_layers+1):
                    #pdb.set_trace()
                    thislayer_dim_mid = round(layer1_dim_mid**(int_layer) / (self.input_dim**(int_layer-1)))
                    hp_thislayer_dim_mid = hp.Fixed('intermediate_dim_'+str(int_layer), thislayer_dim_mid)
                    self.int_layer_dims.append(thislayer_dim_mid)
                    encoder_layer = layers.Dense(thislayer_dim_mid, activation="relu",name='intermediate_layer_'+str(int_layer))(encoder_layer)

        latent_layer = layers.Dense(hp_latent_dim, activation="linear",name='latent_layer')(encoder_layer)
        
        # Define decoder model.
        if(self.int_layers>0):
            layer1_dim_mid = self.int_layer_dims[-1]
            decoder_layer = layers.Dense(round(layer1_dim_mid), activation="relu",name='decoder_layer_1')(latent_layer)
            if(self.int_layers>1):
                for this_int_layer in range(2,self.int_layers+1):
                    #pdb.set_trace()
                    thislayer_dim_mid = self.int_layer_dims[-this_int_layer]
                    decoder_layer = layers.Dense(round(thislayer_dim_mid), activation="relu",name='decoder_layer_'+str(this_int_layer))(decoder_layer)
       
        decoder_output_layer = layers.Dense(hp_input_dim, activation=hp_output_activation,name='decoder_output_layer')(decoder_layer)
        #decoder_ae = Model(inputs=latent_inputs_ae, outputs=decoder_output_layer, name="decoder_ae")

        # #Define VAE model.
        # outputs = decoder_ae(layer4_vae)
        # #outputs_vae = decoder(encoder(inputs)[2])
        
        ae = Model(inputs=encoder_input_layer, outputs=decoder_output_layer, name="ae")        
        optimizer = optimizers.Adam(learning_rate=hp_learning_rate)
       
        #COMPILING
        #Standard compilation
        ae.compile(optimizer, loss='mse')
        #Compile weighted to reduce number of columns
        #ae.compile(optimizer, loss=['mean_squared_error'],loss_weights=[latent_dim_units])

        return ae
        

# # %%Possibly needed for VAE
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()
# %%Hypertune!
K.clear_session()
##############################
##TUNING HYPERPARAMETERS
##############################
latent_dim = 1

my_hyper_model = kt_model_builder_AE_n_layer(input_dim=ae_input.shape[1],latent_dim = latent_dim,int_layers=3)

#tuner = kt.Hyperband(model_builder,
tuner = kt.Hyperband(my_hyper_model,
                     objective='val_loss',
                    max_epochs=5,
                    factor=3,
                    directory=os.path.normpath('C:/work/temp/keras'),
                    overwrite=True)

tuner.search(ae_input, ae_input, epochs=30, validation_data=(ae_input, ae_input))

# %%# Get the optimal hyperparameters

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

[val.value for key, val in best_hps._space.items() if 'intermediate_dim' in key]

# print(f"""
# The hyperparameter search is complete. The optimal number of units in the first densely-connected
# layer is {best_hps.get('intermediate_dim1')}, the second {best_hps.get('intermediate_dim2')}, latent {best_hps.get('latent_dim')} and the optimal learning rate for the optimizer
# is {best_hps.get('learning_rate')}, and the output activation is {best_hps.get('decoder_output_activation')}.
# """)



# # %%Basic autoencoder with 2 intermediate layers
# #THIS WORKS DONT MESS WITH IT, ALTHOUGH LOSS IS QUITE HIGH
# def make_2int_AE(hp='DEFAULT'):
#     #def __init__(self):
#     if(hp=='DEFAULT'):#Use parameters from the list
#         input_dim = 964
#         latent_dim = 30
#         intermediate_dim1 = 300
#         intermediate_dim2 = 100
#         decoder_output_activation = 'linear'
    
#     else:   #Use kerastuner hyperparameters
#         input_dim = hp.get('input_dim')
#         latent_dim = hp.get('latent_dim')
#         intermediate_dim1 = hp.get('intermediate_dim1')
#         intermediate_dim2 = hp.get('intermediate_dim2')
#         learning_rate = hp.get('learning_rate')
#         decoder_output_activation = hp.get('decoder_output_activation')
    

#     #Define encoder model
#     encoder_input_layer = keras.Input(shape=(input_dim,), name="encoder_input_layer")
#     encoder_layer1 = layers.Dense(intermediate_dim1, activation="relu",name='intermediate_layer1')(encoder_input_layer)
#     encoder_layer2 = layers.Dense(intermediate_dim2, activation="relu",name='intermediate_layer2')(encoder_layer1)    
#     latent_layer = layers.Dense(latent_dim, activation="linear",name='latent_layer')(encoder_layer2)
#     encoder_ae = Model(inputs=encoder_input_layer, outputs=latent_layer, name="encoder_ae")

#     #latent_space = encoder_ae()
#     # Define decoder model.
#     decoder_input_layer = keras.Input(shape=(latent_dim,), name="decoder_input_layer")
#     decoder_layer1 = layers.Dense(intermediate_dim2, activation="relu",name='decoder_layer1')(decoder_input_layer)
#     #decoder_layer1 = layers.Dense(intermediate_dim2, activation="relu",name='decoder_layer1')(latent_layer)
#     decoder_layer2 = layers.Dense(intermediate_dim1, activation="relu",name='decoder_layer2')(decoder_layer1)
#     decoder_output_layer = layers.Dense(input_dim, activation=decoder_output_activation,name='decoder_output_layer')(decoder_layer2)
#     decoder_ae = Model(inputs=decoder_input_layer, outputs=decoder_output_layer, name="decoder_ae")

#     # #Define VAE model.
#     # outputs = decoder_ae(layer4_vae)
#     # #outputs_vae = decoder(encoder(inputs)[2])
#     #decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
    
#     #latent_inputs_ae = tf.keras.Input(shape=(best_hps.get('latent_units'),), name="decoder_input")
#     #decoder_ae = tf.keras.Model(inputs=latent_inputs_ae, outputs=outputs_ae, name="decoder_ae")
    
#     outputs = decoder_ae(latent_layer)
    
    
#     ae = Model(inputs=encoder_input_layer, outputs=outputs, name="ae")
#     #ae = Model(inputs=encoder_input_layer, outputs=decoder_output_layer, name="ae")
#     optimizer = optimizers.Adam(learning_rate=learning_rate)

#     #COMPILING
#     #Standard compilation
#     ae.compile(optimizer, loss='mse')
#     #Compile weighted to reduce number of columns
#     #ae.compile(optimizer, loss=['mean_squared_error'],loss_weights=[latent_dim_units])

#     return ae, encoder_ae, decoder_ae


# # %%Basic 2layer autoencoder class
# class AE_2int():
#     def __init__(self,hp):
#         if(hp=='DEFAULT'):#Use parameters from the list
#             self.input_dim = 964
#             self.latent_dim = 30
#             self.int_layer_dims = [300,100]
#             self.intermediate_dim1 = 300
#             self.intermediate_dim2 = 100
#             self.decoder_output_activation = 'linear'
        
#         else:   #Use kerastuner hyperparameters
#             self.input_dim = hp.get('input_dim')
#             self.latent_dim = hp.get('latent_dim')
#             self.int_layer_dims = [val.value for key, val in best_hps._space.items() if 'intermediate_dim' in key]
#             self.learning_rate = hp.get('learning_rate')
#             self.decoder_output_activation = hp.get('decoder_output_activation')
        
#         self.ae = None
#         self.encoder = None
#         self.decoder = None
#         self.build_model()
        
#     def build_model(self):

#         #Define encoder model
#         encoder_input_layer = keras.Input(shape=(self.input_dim,), name="encoder_input_layer")
#         encoder_layer1 = layers.Dense(self.intermediate_dim1, activation="relu",name='intermediate_layer1')(encoder_input_layer)
#         encoder_layer2 = layers.Dense(self.intermediate_dim2, activation="relu",name='intermediate_layer2')(encoder_layer1)    
#         latent_layer = layers.Dense(self.latent_dim, activation="linear",name='latent_layer')(encoder_layer2)
#         self.encoder = Model(inputs=encoder_input_layer, outputs=latent_layer, name="encoder_ae")
    
#         #latent_space = encoder_ae()
#         # Define decoder model.
#         decoder_input_layer = keras.Input(shape=(self.latent_dim,), name="decoder_input_layer")
#         decoder_layer1 = layers.Dense(self.intermediate_dim2, activation="relu",name='decoder_layer1')(decoder_input_layer)
#         decoder_layer2 = layers.Dense(self.intermediate_dim1, activation="relu",name='decoder_layer2')(decoder_layer1)
#         decoder_output_layer = layers.Dense(self.input_dim, activation=self.decoder_output_activation,name='decoder_output_layer')(decoder_layer2)
#         self.decoder = Model(inputs=decoder_input_layer, outputs=decoder_output_layer, name="decoder_ae")
        
#         outputs = self.decoder(latent_layer)
        
        
#         self.ae = Model(inputs=encoder_input_layer, outputs=outputs, name="ae")
#         optimizer = optimizers.Adam(learning_rate=self.learning_rate)
    
#         #COMPILING
#         self.ae.compile(optimizer, loss='mse')
    
#     def fit_model(self, x_train,x_test='DEFAULT',batch_size=100,epochs=30):
#         if(x_test=='DEFAULT'):
#             _history = self.ae.fit(x_train,x_train,
#                          shuffle=True,
#                          epochs=epochs,
#                          batch_size=batch_size,
#                          validation_data=(x_train,x_train))
#         else:
#             _history = self.ae.fit(x_train,x_train,
#                         shuffle=True,
#                         epochs=epochs,
#                         batch_size=batch_size,
#                         validation_data=(x_test,x_test))
            
#         return _history


#     def encode(self, data,batch_size=100):
#         #self.encoder = Model(self.input, self.encoded)
#         return self.encoder.predict(data, batch_size=batch_size)
    
#     def decode(self, data,batch_size=100):
#         #encoded_input = Input(shape=(self.latent_dim,))
#         #decoder_layer = self.ae.layers[-1]
#         # Create the decoder model
#         #self.decoder = Model(encoded_input, decoder_layer(encoded_input))
#         return self.decoder.predict(data, batch_size=batch_size) 
            
# # %%Basic 3layer autoencoder class
# class AE_3int():
#     def __init__(self,hp):
#         if(hp=='DEFAULT'):#Use parameters from the list
#             self.input_dim = 964
#             self.latent_dim = 30
#             self.intermediate_dim1 = 500
#             self.intermediate_dim2 = 300
#             self.intermediate_dim3 = 100
#             self.decoder_output_activation = 'linear'
        
#         else:   #Use kerastuner hyperparameters
#             self.input_dim = hp.get('input_dim')
#             self.latent_dim = hp.get('latent_dim')
#             self.intermediate_dim1 = hp.get('intermediate_dim1')
#             self.intermediate_dim2 = hp.get('intermediate_dim2')
#             self.intermediate_dim2 = hp.get('intermediate_dim3')
#             self.learning_rate = hp.get('learning_rate')
#             self.decoder_output_activation = hp.get('decoder_output_activation')
        
#         self.ae = None
#         self.encoder = None
#         self.decoder = None
#         self.build_model()
        
#     def build_model(self):

#         #Define encoder model
#         encoder_input_layer = keras.Input(shape=(self.input_dim,), name="encoder_input_layer")
#         encoder_layer1 = layers.Dense(self.intermediate_dim1, activation="relu",name='intermediate_layer1')(encoder_input_layer)
#         encoder_layer2 = layers.Dense(self.intermediate_dim2, activation="relu",name='intermediate_layer2')(encoder_layer1)   
#         encoder_layer3 = layers.Dense(self.intermediate_dim2, activation="relu",name='intermediate_layer2')(encoder_layer2)
#         latent_layer = layers.Dense(self.latent_dim, activation="linear",name='latent_layer')(encoder_layer3)
#         self.encoder = Model(inputs=encoder_input_layer, outputs=latent_layer, name="encoder_ae")
    
#         #latent_space = encoder_ae()
#         # Define decoder model.
#         decoder_input_layer = keras.Input(shape=(self.latent_dim,), name="decoder_input_layer")
#         decoder_layer1 = layers.Dense(self.intermediate_dim2, activation="relu",name='decoder_layer1')(decoder_input_layer)
#         decoder_layer2 = layers.Dense(self.intermediate_dim1, activation="relu",name='decoder_layer2')(decoder_layer1)
#         decoder_layer3 = layers.Dense(self.intermediate_dim1, activation="relu",name='decoder_layer2')(decoder_layer2)
#         decoder_output_layer = layers.Dense(self.input_dim, activation=self.decoder_output_activation,name='decoder_output_layer')(decoder_layer3)
#         self.decoder = Model(inputs=decoder_input_layer, outputs=decoder_output_layer, name="decoder_ae")
        
#         outputs = self.decoder(latent_layer)
        
        
#         self.ae = Model(inputs=encoder_input_layer, outputs=outputs, name="ae")
#         optimizer = optimizers.Adam(learning_rate=self.learning_rate)
    
#         #COMPILING
#         self.ae.compile(optimizer, loss='mse')
    
#     def fit_model(self, x_train,x_test='DEFAULT',batch_size=100,epochs=30):
#         if(x_test=='DEFAULT'):
#             _history = self.ae.fit(x_train,x_train,
#                          shuffle=True,
#                          epochs=epochs,
#                          batch_size=batch_size,
#                          validation_data=(x_train,x_train))
#         else:
#             _history = self.ae.fit(x_train,x_train,
#                         shuffle=True,
#                         epochs=epochs,
#                         batch_size=batch_size,
#                         validation_data=(x_test,x_test))
            
#         return _history


#     def encode(self, data,batch_size=100):
#         return self.encoder.predict(data, batch_size=batch_size)
    
#     def decode(self, data,batch_size=100):
#         return self.decoder.predict(data, batch_size=batch_size) 
    
    
    
# %%Basic n-layer autoencoder class
class AE_n_layer():
    def __init__(self,hp='DEFAULT',input_dim=964,latent_dim=15,int_layers=2,int_layer_dims='DEFAULT'):
        if(hp=='DEFAULT'):#Use parameters from the list
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            self.int_layers = int_layers
            
            #Make logspace int layer dims if required
            if(int_layer_dims=='DEFAULT'):
                self.int_layer_dims = []
                if(self.int_layers>0):
                    layer1_dim_mid = (self.input_dim**self.int_layers *self.latent_dim)**(1/(self.int_layers+1))
                    self.int_layer_dims.append(round(layer1_dim_mid))
                    if(self.int_layers>1):
                        for int_layer in range(2,self.int_layers+1):
                            thislayer_dim_mid = round(layer1_dim_mid**(int_layer) / (self.input_dim**(int_layer-1)))
                            self.int_layer_dims.append(thislayer_dim_mid)
                            
            else:
                self.int_layer_dims = int_layer_dims
                
            self.decoder_output_activation = 'linear'
            self.learning_rate = 1e-3
        
        else:   #Use kerastuner hyperparameters
            self.input_dim = hp.get('input_dim')
            self.latent_dim = hp.get('latent_dim')
            self.int_layers = hp.get('intermediate_layers')
            self.int_layer_dims = [val.value for key, val in hp._space.items() if 'intermediate_dim' in key]
            self.learning_rate = hp.get('learning_rate')
            self.decoder_output_activation = hp.get('decoder_output_activation')
        
        self.ae = None
        self.encoder = None
        self.decoder = None
        self.build_model()
        
    def build_model(self):

        #Define encoder model
        encoder_input_layer = keras.Input(shape=(self.input_dim,), name="encoder_input_layer")
        
        #Create the encoder layers
        #The dimensions of the intermediate layers are stored in self.int_layer_dims
        #The number of intermediate layers is stored in self.int_layers
        if(self.int_layers>0):
            layer1_dim_mid = self.int_layer_dims[0]
            encoder_layer = layers.Dense(layer1_dim_mid, activation="relu",name='intermediate_layer_1')(encoder_input_layer)
            if(self.int_layers>1):
                for int_layer in range(2,self.int_layers+1):
                    thislayer_dim = self.int_layer_dims[int_layer-1]
                    encoder_layer = layers.Dense(thislayer_dim, activation="relu",name='intermediate_layer_'+str(int_layer))(encoder_layer)

        latent_layer = layers.Dense(self.latent_dim, activation="linear",name='latent_layer')(encoder_layer)
        self.encoder = Model(inputs=encoder_input_layer, outputs=latent_layer, name="encoder_ae")
    
        # Define decoder model.
        decoder_input_layer = keras.Input(shape=(self.latent_dim,), name="decoder_input_layer")
        if(self.int_layers>0):
            layer1_dim_mid = self.int_layer_dims[-1]
            decoder_layer = layers.Dense(layer1_dim_mid, activation="relu",name='decoder_layer_1')(decoder_input_layer)
            if(self.int_layers>1):
                for this_int_layer in range(2,self.int_layers+1):
                    thislayer_dim_mid = self.int_layer_dims[-this_int_layer]
                    decoder_layer = layers.Dense(round(thislayer_dim_mid), activation="relu",name='decoder_layer_'+str(this_int_layer))(decoder_layer)
           
        decoder_output_layer = layers.Dense(self.input_dim, activation=self.decoder_output_activation,name='decoder_output_layer')(decoder_layer)
        self.decoder = Model(inputs=decoder_input_layer, outputs=decoder_output_layer, name="decoder_ae")
        
        outputs = self.decoder(latent_layer)
        
        
        self.ae = Model(inputs=encoder_input_layer, outputs=outputs, name="ae")
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
    
        #COMPILING
        self.ae.compile(optimizer, loss='mse')
    
    def fit_model(self, x_train,x_test='DEFAULT',batch_size=100,epochs=30):
        if(x_test=='DEFAULT'):
            _history = self.ae.fit(x_train,x_train,
                         shuffle=True,
                         epochs=epochs,
                         batch_size=batch_size,
                         validation_data=(x_train,x_train))
        else:
            _history = self.ae.fit(x_train,x_train,
                        shuffle=True,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_test,x_test))
            
        return _history


    def encode(self, data,batch_size=100):
        return self.encoder.predict(data, batch_size=batch_size)
    
    def decode(self, data,batch_size=100):
        return self.decoder.predict(data, batch_size=batch_size) 



#%%AE n layers
ae_obj = AE_n_layer(best_hps)
history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=300)
val_acc_per_epoch = history.history['val_loss']
best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
ae_obj = AE_n_layer(best_hps)
ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=best_epoch)
print('Best epoch: %d' % (best_epoch,))


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss)+1)
fig,ax=plt.subplots(2,1,figsize=(12,8))
ax[0].plot(epochs, loss, 'bo', label='Training loss')
ax[0].plot(epochs, val_loss, 'b', label='Validation loss')
ax[0].set_title('Training and validation loss')
ax[1].plot(epochs, loss, 'bo', label='Training loss')
ax[1].plot(epochs, val_loss, 'b', label='Validation loss')
ax[1].set_title('Training and validation loss')
ax[1].set_yscale('log')
plt.legend()
plt.show()



#%%How many epochs?
ae_obj = AE_2int(best_hps)
history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=50)
val_acc_per_epoch = history.history['val_loss']
best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))
ae_obj = AE_2int(best_hps)
ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=best_epoch)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss)+1)
fig,ax=plt.subplots(2,1,figsize=(12,8))
ax[0].plot(epochs, loss, 'bo', label='Training loss')
ax[0].plot(epochs, val_loss, 'b', label='Validation loss')
ax[0].set_title('Training and validation loss- 2 intermediate layer AE')
ax[0].set_ylabel('MSE')
ax[0].set_xlabel('Epochs')
ax[1].plot(epochs, loss, 'bo', label='Training loss')
ax[1].plot(epochs, val_loss, 'b', label='Validation loss')
ax[1].set_yscale('log')
ax[1].set_ylabel('MSE')
ax[1].set_xlabel('Epochs')
plt.legend()
plt.show()



#%%Plot input vs output
latent_space = ae_obj.encoder(ae_input).numpy()
decoded_latent_space = ae_obj.decoder(latent_space)

fig,ax = plt.subplots(1)
plt.scatter(ae_obj.ae(ae_input),ae_input)
plt.title("AE input vs output")
plt.xlabel('AE input')
plt.ylabel('AE output')
plt.show()

#%%Now compare loss for different latent dimensions
#This is NOT using kerastuner, and using log-space int layers
latent_dims = []
AE1_MSE_best50epoch =[]
AE2_MSE_best50epoch =[]
AE3_MSE_best50epoch =[]
AE4_MSE_best50epoch =[]
best_hps_array = []

input_dim = ae_input.shape[1]

for latent_dim in range(1,3):
    K.clear_session()
    latent_dims.append(latent_dim)
    
    #Test for 1 intermediate layer
    ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=1)
    history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=100)
    val_acc_per_epoch = history.history['val_loss']
    AE1_MSE_best50epoch.append(min(history.history['val_loss']))
    
    #Test for 2 intermediate layers
    ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=2)
    history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=100)
    val_acc_per_epoch = history.history['val_loss']
    AE2_MSE_best50epoch.append(min(history.history['val_loss']))
    
    #Test for 3 intermediate layers
    ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=3)
    history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=100)
    val_acc_per_epoch = history.history['val_loss']
    AE3_MSE_best50epoch.append(min(history.history['val_loss']))
    
    #Test for 4 intermediate layers
    ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=4)
    history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=100)
    val_acc_per_epoch = history.history['val_loss']
    AE4_MSE_best50epoch.append(min(history.history['val_loss']))
    
    

# #%%Now compare loss for different latent dimensions
# #HOPEFULLY THIS WAS THE ISSUE, I WAS JUST SAYING RANGE(5) WHICH STARTS AT ZERO, SO IT WAS TRYING TO 
# #DO LATENT DIMS AS 0. HOPEFULLY THATS THE ISSUE. DONT KNOW WHY THAT WORKS IN CPU MODE THOUGH...
# latent_dims = []
# AE1_MSE_best50epoch =[]
# AE2_MSE_best50epoch =[]
# AE3_MSE_best50epoch =[]
# AE4_MSE_best50epoch =[]
# best_hps_array = []

# directory = os.path.normpath('C:/Work/temp/keras')

# for latent_dim in range(1,30):
#     K.clear_session()
#     latent_dims.append(latent_dim)
#     ##TUNING HYPERPARAMETERS for 1 latent dim
#     my_hyper_model = kt_model_builder_AE_n_layer(input_dim=ae_input.shape[1],latent_dim = latent_dim,int_layers=1)
#     tuner = kt.Hyperband(my_hyper_model,
#                       objective='val_loss',
#                     max_epochs=5,
#                     factor=3,
#                     directory=directory,
#                     overwrite=True)
#     tuner.search(ae_input, ae_input, epochs=30, validation_data=(ae_input, ae_input))
#     best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
#     # #Make AE with best hyperparameters
#     ae_obj = AE_n_layer(best_hps)
#     history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=100)
#     val_acc_per_epoch = history.history['val_loss']
#     AE1_MSE_best50epoch.append(min(history.history['val_loss']))
    
#     ##TUNING HYPERPARAMETERS for 2 latent dim
#     my_hyper_model = kt_model_builder_AE_n_layer(input_dim=ae_input.shape[1],latent_dim = latent_dim,int_layers=2)
#     tuner = kt.Hyperband(my_hyper_model,
#                       objective='val_loss',
#                     max_epochs=5,
#                     factor=3,
#                     directory=directory,
#                     overwrite=True)
#     tuner.search(ae_input, ae_input, epochs=30, validation_data=(ae_input, ae_input))
#     best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
#     # #Make AE with best hyperparameters
#     ae_obj = AE_n_layer(best_hps)
#     history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=100)
#     val_acc_per_epoch = history.history['val_loss']
#     AE2_MSE_best50epoch.append(min(history.history['val_loss']))
    
#     ##TUNING HYPERPARAMETERS for 3 latent dim
#    # THIS NEEDS OVERWRITE=FALSE OTHERWISE IT KACKS OUT
#     my_hyper_model = kt_model_builder_AE_n_layer(input_dim=ae_input.shape[1],latent_dim = latent_dim,int_layers=3)
#     tuner = kt.Hyperband(my_hyper_model,
#                       objective='val_loss',
#                     max_epochs=5,
#                     factor=3,
#                     directory=directory,
#                     overwrite=False)
#     tuner.search(ae_input, ae_input, epochs=30, validation_data=(ae_input, ae_input))
#     best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
#     # #Make AE with best hyperparameters
#     ae_obj = AE_n_layer(best_hps)
#     history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=100)
#     val_acc_per_epoch = history.history['val_loss']
#     AE3_MSE_best50epoch.append(min(history.history['val_loss']))
    
#     ##TUNING HYPERPARAMETERS for 4 latent dim
#     ##THIS NEEDS OVERWRITE=FALSE OTHERWISE IT KACKS OUT
#     my_hyper_model = kt_model_builder_AE_n_layer(input_dim=ae_input.shape[1],latent_dim = latent_dim,int_layers=4)
#     tuner = kt.Hyperband(my_hyper_model,
#                       objective='val_loss',
#                     max_epochs=5,
#                     factor=3,
#                     directory=directory,
#                     overwrite=False)
#     tuner.search(ae_input, ae_input, epochs=30, validation_data=(ae_input, ae_input))
#     best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
#     # #Make AE with best hyperparameters
#     ae_obj = AE_n_layer(best_hps)
#     history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=100)
#     val_acc_per_epoch = history.history['val_loss']
#     AE4_MSE_best50epoch.append(min(history.history['val_loss']))   
#     best_hps_array.append(best_hps)
    
    
#%%
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

import matplotlib.ticker as plticker
loc = plticker.MultipleLocator(base=10.0) # this locator puts ticks at regular intervals
ax[0].xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
ax[0].xaxis.set_minor_locator(plticker.MultipleLocator(base=5.0))
ax[1].xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
ax[1].xaxis.set_minor_locator(plticker.MultipleLocator(base=5.0))

plt.show()

#pd.DataFrame({'latent_dims':latent_dims,'AE3_MSE_best50epoch':AE3_MSE_best50epoch}).to_csv("C:/work/Python/Github/Orbitrap_clustering/Performance_metrics/AE_3layer_MSE.csv")
