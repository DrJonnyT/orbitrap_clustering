# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 14:18:43 2021

@author: mbcx5jt5
"""

#####################################
####IMPORT STATEMENTS FOR TESTING####
#####################################
#Set random seed for repeatability
from numpy.random import seed
seed(1337)
import tensorflow as tf
tf.random.set_seed(1338)

import pandas as pd
import math
import datetime as dt
import numpy as np
import pdb
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras import metrics
import kerastuner as kt
from sklearn.preprocessing import RobustScaler, StandardScaler,FunctionTransformer,MinMaxScaler
from sklearn.pipeline import Pipeline


import re
import os

#%%

#################################################
#####AUTOENCODER GENERATORS###################
##############################################
# %%Basic n-layer autoencoder class
class AE_n_layer():
    def __init__(self,hp='DEFAULT',input_dim=964,latent_dim=15,int_layers=2,int_layer_dims='DEFAULT',latent_activation='linear'):
        if(hp=='DEFAULT'):#Use parameters from the list
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            self.int_layers = int_layers
            self.latent_activation=latent_activation
            
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
            self.latent_activation = hp.get('latent_activation')
        
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

        latent_layer = layers.Dense(self.latent_dim, activation=self.latent_activation,name='latent_layer')(encoder_layer)
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
    
    def fit_model(self, x_train,x_test='DEFAULT',batch_size=100,epochs=30,verbose='auto'):
        if(str(x_test)=='DEFAULT'):
            _history = self.ae.fit(x_train,x_train,
                         shuffle=True,
                         epochs=epochs,
                         batch_size=batch_size,
                         validation_data=(x_train,x_train),verbose=verbose)
        else:
            _history = self.ae.fit(x_train,x_train,
                        shuffle=True,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_test,x_test),verbose=verbose)
            
        return _history


    def encode(self, data,batch_size=100):
        return self.encoder.predict(data, batch_size=batch_size)
    
    def decode(self, data,batch_size=100):
        return self.decoder.predict(data, batch_size=batch_size)


# %%Basic n-layer variational autoencoder class

class My_KL_Layer(layers.Layer):
    '''
    @note: Returns the input layers ! Required to allow for z-point calculation
           in a final Lambda layer of the Encoder model    
    '''
    # Standard initialization of layers 
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(My_KL_Layer, self).__init__(*args, **kwargs)

    # The implementation interface of the Layer
    def call(self, inputs, beta = 4.5e-4):
        mu      = inputs[0]
        log_var = inputs[1]
        # Note: from other analysis we know that the backend applies tf.math.functions 
        # "fact" must be adjusted - for MNIST reasonable values are in the range of 0.65e-4 to 6.5e-4
        kl_mean_batch = - beta * K.mean(1 + log_var - K.square(mu) - K.exp(log_var))
        # We add the loss via the layer's add_loss() - it will be added up to other losses of the model     
        self.add_loss(kl_mean_batch, inputs=inputs)
        # We add the loss information to the metrics displayed during training 
        self.add_metric(kl_mean_batch, name='kl', aggregation='mean')
        #self.add_metric(beta, name='beta3', aggregation='mean')
        return inputs


class VAE_n_layer():
    def __init__(self,hp='DEFAULT',input_dim=964,latent_dim=2,int_layers=2,beta_schedule=np.array([0,0.1]),int_layer_dims='DEFAULT',
                 learning_rate=1e-3,latent_activation='linear',decoder_output_activation='linear'):
        if(hp=='DEFAULT'):#Use parameters from the list
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            self.int_layers = int_layers
            self.decoder_output_activation = decoder_output_activation
            self.latent_activation = latent_activation
            self.learning_rate = learning_rate 
            
            
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
   
        else:   #Use kerastuner hyperparameters
            quit("VAE_n_layer implementation with kerastuner hyperparameters not currently implemented")
            # self.input_dim = hp.get('input_dim')
            # self.latent_dim = hp.get('latent_dim')
            # self.int_layers = hp.get('intermediate_layers')
            # self.int_layer_dims = [val.value for key, val in hp._space.items() if 'intermediate_dim' in key]
            # self.learning_rate = hp.get('learning_rate')
            # self.decoder_output_activation = hp.get('decoder_output_activation')
            # self.latent_activation = hp.get('decoder_output_activation')
        
        self.mu = None
        self.log_var = None
        self.z_mean = None
        self.beta=tf.Variable(10.)
        self.beta_schedule=beta_schedule       
        # self.alpha5=K.variable(1.)
        # self.beta5=K.variable(0.001)        
        
        self.ae = None
        self.encoder = None
        self.decoder = None
        self.build_model()
        
    def build_model(self):
        print("Look at commented code in ae_functions for vae_beta_scheduler, custom function is required for each VAE object")
        # Preparation: We later need a function to calculate the z-points in the latent space 
        # this function will be used by an eventual Lambda layer of the Encoder 
        def z_point_sampling(args):
            '''
            A point in the latent space is calculated statistically 
            around an optimized mu for each sample 
            '''
            mu, log_var = args # Note: These are 1D tensors !
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            return mu + K.exp(log_var / 2) * epsilon        


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

        
        # "Variational" part - create 2 Dense layers for a statistical distribution of z-points  
        self.mu      = layers.Dense(self.latent_dim, name='mu')(encoder_layer)
        self.log_var = layers.Dense(self.latent_dim, name='log_var')(encoder_layer)
        
        self.mu, self.log_var = My_KL_Layer()([self.mu, self.log_var], beta=self.beta)
        self.z_mean = layers.Lambda(z_point_sampling, name='encoder_output')([self.mu, self.log_var])        
        self.encoder = Model(inputs=encoder_input_layer, outputs=self.z_mean, name="encoder_vae")
        
        #This is the reparameterization trick
        def sampling(args):
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon


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
        self.decoder = Model(inputs=decoder_input_layer, outputs=decoder_output_layer, name="decoder_vae")
        
        outputs = self.decoder(self.z_mean)
        
        
        self.ae = Model(inputs=encoder_input_layer, outputs=outputs, name="vae")
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)

        self.ae.compile(optimizer, loss='mse',metrics=[tf.keras.metrics.MeanSquaredError(name='msemetric')])
        #self.ae.add_metric(tf.Variable(0.),name='betametric',aggregation='mean')
    
    def fit_model(self, x_train,x_test=None,batch_size=100,epochs=30,verbose='auto',callbacks=[]):
        if(x_test is None):
            _history = self.ae.fit(x_train,x_train,
                         shuffle=True,
                         epochs=epochs,
                         batch_size=batch_size,
                         validation_data=(x_train,x_train),callbacks=callbacks,verbose=verbose)
        else:
            _history = self.ae.fit(x_train,x_train,
                        shuffle=True,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_test,x_test),callbacks=callbacks,verbose=verbose)
            
        return _history


    def encode(self, data,batch_size=100):
        return self.encoder.predict(data, batch_size=batch_size)
    
    def decode(self, data,batch_size=100):
        return self.decoder.predict(data, batch_size=batch_size)
    



# #Callback for VAE to make beta change with training epoch
# class vae_beta_scheduler(keras.callbacks.Callback):
#     """Callback for VAE to make beta change with training epoch

#   Arguments:
#       schedule: this is a dummy and doesn't actually do anything.
#       You need to tailor this to your vae_n_layer object, for example here it's called vae_obj'
#   """

#     def __init__(self,schedule):
#         super(vae_beta_scheduler, self).__init__()
#         self.schedule=schedule

#     def on_epoch_begin(self,  epoch, logs=None):
#         beta_schedule=vae_obj.beta_schedule
#         if(epoch >= beta_schedule.shape[0]):
#             new_beta = beta_schedule[-1]
#         else:
#             new_beta = beta_schedule[epoch]
#         tf.keras.backend.set_value(vae_obj.beta, new_beta)         
#         print("\nEpoch %05d: beta is %6.4f." % (epoch, new_beta))
        
        


    
# # %%AE with single layer decoder
# class testAE_n_layer():
#     def __init__(self,hp='DEFAULT',input_dim=964,latent_dim=15,int_layers=2,int_layer_dims='DEFAULT',learning_rate=1e-3,latent_activation='relu'):
#         if(hp=='DEFAULT'):#Use parameters from the list
#             self.input_dim = input_dim
#             self.latent_dim = latent_dim
#             self.int_layers = int_layers
#             self.latent_activation = latent_activation
#             self.learning_rate = learning_rate 
            
            
#             #Make logspace int layer dims if required
#             if(int_layer_dims=='DEFAULT'):
#                 self.int_layer_dims = []
#                 if(self.int_layers>0):
#                     layer1_dim_mid = (self.input_dim**self.int_layers *self.latent_dim)**(1/(self.int_layers+1))
#                     self.int_layer_dims.append(round(layer1_dim_mid))
#                     if(self.int_layers>1):
#                         for int_layer in range(2,self.int_layers+1):
#                             thislayer_dim_mid = round(layer1_dim_mid**(int_layer) / (self.input_dim**(int_layer-1)))
#                             self.int_layer_dims.append(thislayer_dim_mid)
                            
#             else:
#                 self.int_layer_dims = int_layer_dims
   
            
            
            
        
#         else:   #Use kerastuner hyperparameters
#             self.input_dim = hp.get('input_dim')
#             self.latent_dim = hp.get('latent_dim')
#             self.int_layers = hp.get('intermediate_layers')
#             self.int_layer_dims = [val.value for key, val in hp._space.items() if 'intermediate_dim' in key]
#             self.learning_rate = hp.get('learning_rate')
#             self.latent_activation = hp.get('decoder_output_activation')
        
#         self.ae = None
#         self.encoder = None
#         self.decoder = None
#         self.build_model()
        
#     def build_model(self):

#         #Define encoder model
#         encoder_input_layer = keras.Input(shape=(self.input_dim,), name="encoder_input_layer")
        
#         #Create the encoder layers
#         #The dimensions of the intermediate layers are stored in self.int_layer_dims
#         #The number of intermediate layers is stored in self.int_layers
#         if(self.int_layers>0):
#             layer1_dim_mid = self.int_layer_dims[0]
#             encoder_layer = layers.Dense(layer1_dim_mid, activation="relu",name='intermediate_layer_1',use_bias=False)(encoder_input_layer)
#             if(self.int_layers>1):
#                 for int_layer in range(2,self.int_layers+1):
#                     thislayer_dim = self.int_layer_dims[int_layer-1]
#                     encoder_layer = layers.Dense(thislayer_dim, activation="relu",name='intermediate_layer_'+str(int_layer),use_bias=False)(encoder_layer)

#         #z is the latent layer
#         #z_mean = layers.Dense(self.latent_dim,activation=self.latent_activation,name='latent_mean',use_bias=False)(encoder_layer)
#         z_mean = layers.Dense(self.latent_dim,activation=self.latent_activation,name='latent_mean',use_bias=False)(encoder_input_layer)
#         z_log_var = layers.Dense(self.latent_dim,name='latent_log_var',use_bias=False)(encoder_layer)
#         self.z_mean = z_mean
#         self.z_log_var = z_log_var
#         self.encoder = Model(inputs=encoder_input_layer, outputs=z_mean, name="encoder_vae")
        
#         # # #Resample
#         # def sampling(args):
#         #     z_mean, z_log_var = args
#         #     batch = tf.shape(z_mean)[0]
#         #     dim = tf.shape(z_mean)[1]
#         #     #epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
#         #                               #stddev=epsilon_std)
#         #     epsilon = K.random_normal(shape=(batch, dim), mean=0.,
#         #                               stddev=1)
#         #     return z_mean + K.exp(z_log_var / 2) * epsilon
        
#         #This is the reparameterization trick
#         def sampling(args):
#             z_mean, z_log_var = args
#             batch = K.shape(z_mean)[0]
#             dim = K.int_shape(z_mean)[1]
#             epsilon = K.random_normal(shape=(batch, dim))
#             return z_mean + K.exp(0.5 * z_log_var) * epsilon

        
#         latent_resampled = layers.Lambda(sampling, output_shape=(self.latent_dim,),name='resampled_latent_mean')([z_mean, z_log_var])
        
    
#         # Define decoder model.
#         decoder_input_layer = keras.Input(shape=(self.latent_dim,), name="decoder_input_layer")
#         decoder_output_layer = layers.Dense(self.input_dim, activation='linear',name='decoder_output_layer',use_bias=False)(decoder_input_layer)
#         self.decoder = Model(inputs=decoder_input_layer, outputs=decoder_output_layer, name="decoder_vae")
        
#         #outputs = self.decoder(latent_resampled)
#         outputs = self.decoder(z_mean)
        
        
#         self.ae = Model(inputs=encoder_input_layer, outputs=outputs, name="ae")
#         #optimizer = optimizers.Adam(learning_rate=self.learning_rate)
#         optimizer = optimizers.RMSprop(learning_rate=self.learning_rate)

        
#         def mse_loss(y_true,y_pred):
#             mse_total = tf.reduce_mean(tf.metrics.mean_squared_error(y_true, y_pred))
#             return mse_total

#         # rate = K.variable(0.0,name='KL_Annealing')
#         # annealing_rate = 0.0001
#         self.beta = 0.0001
#         def kl_loss(y_true, y_pred):
#             mean=self.z_mean
#             log_var = self.z_log_var
#             kl_loss = self.beta * -0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis = 1)
#             self.beta = self.beta*1.3
#             return kl_loss


#         def FAE_loss(y_true, y_pred):
#             the_latent_space = self.encoder(y_true)
#             latent_columns_decoded = tf.zeros(tf.shape(y_true))            
            
#             for i in np.arange(self.latent_dim):
#             #TRY each column in latent space is a factor
#                 split1, split2, split3 = tf.split(the_latent_space,[i,1,(-1+self.latent_dim-i)],1)    
#                 zeros1 = tf.zeros(tf.shape(split1))
#                 zeros3 = tf.zeros(tf.shape(split3))
#                 onecol_withzeros = tf.concat([zeros1,split2,zeros3],axis=1)
            
#                 thiscol_decoded = self.decoder(onecol_withzeros)
#                 latent_columns_decoded = latent_columns_decoded + thiscol_decoded
                
#             #pdb.set_trace()
#             aa = tfp.stats.correlation(the_latent_space)
#             bb = tf.math.reduce_min(aa)
#             tf.print(aa)
#             tf.print(bb)
            
            
#             mse_cols = tf.reduce_mean(tf.metrics.mean_squared_error(y_true, latent_columns_decoded))
#             return mse_cols
        
        
#         def zero_nudge(y_true,y_pred):
#             the_latent_space = self.encoder(y_true)
#             latent_zeros = tf.zeros(tf.shape(the_latent_space))
#             decoded_zeros = self.decoder(latent_zeros)
#             mse_zeros = tf.reduce_mean(tf.metrics.mean_squared_error(decoded_zeros, tf.zeros(tf.shape(y_true))))
#             return mse_zeros
        
#         # def mse2(y_true, y_pred):
#         #     the_latent_space = self.encoder(y_true)
#         #     y_decoded = self.decoder(the_latent_space)
#         #     #mse_cols = tf.reduce_mean(tf.metrics.mean_squared_error(y_true, y_decoded))
#         #     #mse_cols=  tf.metrics.mean_squared_error(y_true, y_decoded)
#         #     return K.mean(K.square(y_true-y_decoded))
#         #     #return mse_cols


#         def FVAE_loss(y_true, y_pred):
#             a = mse_loss(y_true,y_pred)
#             b = kl_loss(y_true, y_pred)
#             c = FAE_loss(y_true,y_pred)
#             d = zero_nudge(y_true,y_pred)
#             return a + 10*c# + b + c + d#+ d#+ c# + d
        

        
#         #self.vae.compile(optimizer=optimizer, loss=vae_loss,metrics=[mse_loss,kl_loss])
#         #COMPILING
#         self.ae.compile(optimizer, loss=FVAE_loss,experimental_run_tf_function=False,metrics=[mse_loss,kl_loss])
#         #self.ae.compile(optimizer, loss='mse',experimental_run_tf_function=False,metrics=[mse_loss,kl_loss])
    
#     def fit_model(self, x_train,x_test='DEFAULT',batch_size=100,epochs=30,verbose='auto'):
#         if(x_test=='DEFAULT'):
#             _history = self.ae.fit(x_train,x_train,
#                          shuffle=True,
#                          epochs=epochs,
#                          batch_size=batch_size,
#                          validation_data=(x_train,x_train),verbose=verbose)
#         else:
#             _history = self.ae.fit(x_train,x_train,
#                         shuffle=True,
#                         epochs=epochs,
#                         batch_size=batch_size,
#                         validation_data=(x_test,x_test),verbose=verbose)
            
#         return _history


#     def encode(self, data,batch_size=100):
#         return self.encoder.predict(data, batch_size=batch_size)
    
#     def decode(self, data,batch_size=100):
#         return self.decoder.predict(data, batch_size=batch_size)
   
    
    
    
    
# # %%n-layer factorization autoencoder class
# class FAE_n_layer():
#     def __init__(self,hp='DEFAULT',input_dim=964,latent_dim=15,int_layers=2,int_layer_dims='DEFAULT',latent_activation='linear',decoder_output_activation='linear'):
#         if(hp=='DEFAULT'):#Use parameters from the list
#             self.input_dim = input_dim
#             self.latent_dim = latent_dim
#             self.int_layers = int_layers
#             self.latent_activation=latent_activation
#             self.decoder_output_activation = decoder_output_activation
#             self.learning_rate = 1e-2
            
#             #Make logspace int layer dims if required
#             if(int_layer_dims=='DEFAULT'):
#                 self.int_layer_dims = []
#                 if(self.int_layers>0):
#                     layer1_dim_mid = (self.input_dim**self.int_layers *self.latent_dim)**(1/(self.int_layers+1))
#                     self.int_layer_dims.append(round(layer1_dim_mid))
#                     if(self.int_layers>1):
#                         for int_layer in range(2,self.int_layers+1):
#                             thislayer_dim_mid = round(layer1_dim_mid**(int_layer) / (self.input_dim**(int_layer-1)))
#                             self.int_layer_dims.append(thislayer_dim_mid)
                            
#             else:
#                 self.int_layer_dims = int_layer_dims
                
            
        
#         else:   #Use kerastuner hyperparameters
#             self.input_dim = hp.get('input_dim')
#             self.latent_dim = hp.get('latent_dim')
#             self.int_layers = hp.get('intermediate_layers')
#             self.int_layer_dims = [val.value for key, val in hp._space.items() if 'intermediate_dim' in key]
#             self.learning_rate = hp.get('learning_rate')
#             self.decoder_output_activation = hp.get('decoder_output_activation')
#             self.latent_activation = hp.get('latent_activation')
        
#         self.ae = None
#         self.encoder = None
#         self.decoder = None
#         self.build_model()
        
#     def build_model(self):

#         #Define encoder model
#         encoder_input_layer = keras.Input(shape=(self.input_dim,), name="encoder_input_layer")
        
#         #Create the encoder layers
#         #The dimensions of the intermediate layers are stored in self.int_layer_dims
#         #The number of intermediate layers is stored in self.int_layers
#         if(self.int_layers>0):
#             layer1_dim_mid = self.int_layer_dims[0]
#             encoder_layer = layers.Dense(layer1_dim_mid, activation="relu",name='intermediate_layer_1')(encoder_input_layer)
#             if(self.int_layers>1):
#                 for int_layer in range(2,self.int_layers+1):
#                     thislayer_dim = self.int_layer_dims[int_layer-1]
#                     encoder_layer = layers.Dense(thislayer_dim, activation="relu",name='intermediate_layer_'+str(int_layer))(encoder_layer)

#         latent_layer = layers.Dense(self.latent_dim, activation=self.latent_activation,name='latent_layer')(encoder_layer)
#         self.encoder = Model(inputs=encoder_input_layer, outputs=latent_layer, name="encoder_ae")
    
#         # Define decoder model.
#         decoder_input_layer = keras.Input(shape=(self.latent_dim,), name="decoder_input_layer")
#         if(self.int_layers>0):
#             layer1_dim_mid = self.int_layer_dims[-1]
#             decoder_layer = layers.Dense(layer1_dim_mid, activation="relu",name='decoder_layer_1')(decoder_input_layer)
#             if(self.int_layers>1):
#                 for this_int_layer in range(2,self.int_layers+1):
#                     thislayer_dim_mid = self.int_layer_dims[-this_int_layer]
#                     decoder_layer = layers.Dense(round(thislayer_dim_mid), activation="relu",name='decoder_layer_'+str(this_int_layer))(decoder_layer)
           
#         decoder_output_layer = layers.Dense(self.input_dim, activation=self.decoder_output_activation,name='decoder_output_layer')(decoder_layer)
#         self.decoder = Model(inputs=decoder_input_layer, outputs=decoder_output_layer, name="decoder_ae")
        
#         outputs = self.decoder(latent_layer)#Original outputs line        
        
        
#         def FAE_loss(y_true, y_pred):
#             the_latent_space = self.encoder(y_true)
#             #np_latent_space = the_latent_space.numpy()
            
#             column_max_min_tf = tf.reduce_min(tf.reduce_max(the_latent_space,axis=0))
            
#             #zeropen = 1 - np_latent_space.max(axis=0).min()
            
#             #zeropentf = 1 - tf.math.tanh(    )
            
#             zeropentf = 1# + tf.experimental.numpy.heaviside(column_max_min_tf,10) + tf.experimental.numpy.heaviside(-column_max_min_tf,10)
            
#             #print(zeropen)
#             #print(zeropentf)

            
#             latent_columns_decoded = tf.zeros(tf.shape(y_true))            
#             # latent_factors_decoded = tf.zeros([self.latent_dim,y_true.shape.as_list()[1]]).numpy()
            
#             for i in np.arange(self.latent_dim):
                
#             #TRY each column in latent space is a factor
#                 split1, split2, split3 = tf.split(the_latent_space,[i,1,(-1+self.latent_dim-i)],1)    
#                 zeros1 = tf.zeros(tf.shape(split1))
#                 zeros3 = tf.zeros(tf.shape(split3))
#                 onecol_withzeros = tf.concat([zeros1,split2,zeros3],axis=1)
            
#                 thiscol_decoded = self.decoder(onecol_withzeros)
#                 latent_columns_decoded = latent_columns_decoded + thiscol_decoded
#                 # thiscol_decoded_mean = tf.reduce_mean(thiscol_decoded,axis=0)
#                 # latent_factors_decoded[i,:] = thiscol_decoded_mean.numpy()

            
#             mse_cols = tf.reduce_mean(tf.metrics.mean_squared_error(y_true, latent_columns_decoded))
#             mse_total = tf.reduce_mean(tf.metrics.mean_squared_error(y_true, y_pred))
            
#             #Need to penalise it for having any columns that are all zero in the latent space
#             #zeropen = 1 - np_latent_space.max(axis=0).min()
#             #print(zeropen)
            
#             #cross_correlation between columns when decoded, 1 minus the r2
#             #pdb.set_trace()
            
#             return zeropentf*(mse_total + mse_cols)
#         #    return mse_cols 
        
        
        
#         self.ae = Model(inputs=encoder_input_layer, outputs=outputs, name="ae")
#         optimizer = optimizers.Adam(learning_rate=self.learning_rate)
    
#         #COMPILING
#         #self.ae.compile(optimizer, loss='mse')
#         self.ae.compile(optimizer, loss=FAE_loss,run_eagerly=True)
#         #self.ae.add_loss(tf.keras.losses.MeanSquaredError)
    
#     #Default batch size is the whole dataset
#     def fit_model(self, x_train,x_test='DEFAULT',batch_size=int(1e10),epochs=30,verbose='auto'):
#         if(str(x_test)=='DEFAULT'):
#             _history = self.ae.fit(x_train,x_train,
#                          shuffle=True,
#                          epochs=epochs,
#                          batch_size=batch_size,
#                          validation_data=(x_train,x_train),verbose=verbose)
#         else:
#             _history = self.ae.fit(x_train,x_train,
#                         shuffle=True,
#                         epochs=epochs,
#                         batch_size=batch_size,
#                         validation_data=(x_test,x_test),verbose=verbose)
            
#         return _history


#     def encode(self, data,batch_size=100):
#         return self.encoder.predict(data, batch_size=batch_size)
    
#     def decode(self, data,batch_size=100):
#         return self.decoder.predict(data, batch_size=batch_size)
    
    
# # %%n-layer factorization variational autoencoder class
# class FVAE_n_layer():
#     def __init__(self,hp='DEFAULT',input_dim=964,latent_dim=15,int_layers=2,int_layer_dims='DEFAULT',latent_activation='linear',decoder_output_activation='linear'):
#         if(hp=='DEFAULT'):#Use parameters from the list
#             self.input_dim = input_dim
#             self.latent_dim = latent_dim
#             self.int_layers = int_layers
#             self.latent_activation=latent_activation
#             self.decoder_output_activation = decoder_output_activation
            
#             #Make logspace int layer dims if required
#             if(int_layer_dims=='DEFAULT'):
#                 self.int_layer_dims = []
#                 if(self.int_layers>0):
#                     layer1_dim_mid = (self.input_dim**self.int_layers *self.latent_dim)**(1/(self.int_layers+1))
#                     self.int_layer_dims.append(round(layer1_dim_mid))
#                     if(self.int_layers>1):
#                         for int_layer in range(2,self.int_layers+1):
#                             thislayer_dim_mid = round(layer1_dim_mid**(int_layer) / (self.input_dim**(int_layer-1)))
#                             self.int_layer_dims.append(thislayer_dim_mid)
                            
#             else:
#                 self.int_layer_dims = int_layer_dims
                
#             self.decoder_output_activation = 'linear'
#             self.learning_rate = 1e-2
        
#         else:   #Use kerastuner hyperparameters
#             self.input_dim = hp.get('input_dim')
#             self.latent_dim = hp.get('latent_dim')
#             self.int_layers = hp.get('intermediate_layers')
#             self.int_layer_dims = [val.value for key, val in hp._space.items() if 'intermediate_dim' in key]
#             self.learning_rate = hp.get('learning_rate')
#             self.decoder_output_activation = hp.get('decoder_output_activation')
#             self.latent_activation = hp.get('latent_activation')
        
#         self.ae = None
#         self.encoder = None
#         self.decoder = None
#         self.build_model()
        
#     def build_model(self):

#         #Define encoder model
#         encoder_input_layer = keras.Input(shape=(self.input_dim,), name="encoder_input_layer")
        
#         #Create the encoder layers
#         #The dimensions of the intermediate layers are stored in self.int_layer_dims
#         #The number of intermediate layers is stored in self.int_layers
#         if(self.int_layers>0):
#             layer1_dim_mid = self.int_layer_dims[0]
#             encoder_layer = layers.Dense(layer1_dim_mid, activation="relu",name='intermediate_layer_1')(encoder_input_layer)
#             if(self.int_layers>1):
#                 for int_layer in range(2,self.int_layers+1):
#                     thislayer_dim = self.int_layer_dims[int_layer-1]
#                     encoder_layer = layers.Dense(thislayer_dim, activation="relu",name='intermediate_layer_'+str(int_layer))(encoder_layer)

#         #z is the latent layer
#         z_mean = layers.Dense(self.latent_dim,activation=self.latent_activation,name='latent_mean')(encoder_layer)
#         z_log_var = layers.Dense(self.latent_dim,name='latent_log_var')(encoder_layer)
#         self.encoder = Model(inputs=encoder_input_layer, outputs=z_mean, name="encoder_vae")

#         # #Resample
#         def sampling(args):
#             z_mean, z_log_var = args
#             batch = tf.shape(z_mean)[0]
#             dim = tf.shape(z_mean)[1]
#             #epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
#                                         #stddev=epsilon_std)
#             epsilon = K.random_normal(shape=(batch, dim), mean=0.,
#                                     stddev=1)
#             return z_mean + K.exp(z_log_var / 2) * epsilon
      
#         latent_resampled = layers.Lambda(sampling, output_shape=(self.latent_dim,),name='resampled_latent_mean')([z_mean, z_log_var])
      

#         #latent_layer = layers.Dense(self.latent_dim, activation=self.latent_activation,name='latent_layer')(encoder_layer)
#         #self.encoder = Model(inputs=encoder_input_layer, outputs=latent_layer, name="encoder_ae")
    
#         # Define decoder model.
#         decoder_input_layer = keras.Input(shape=(self.latent_dim,), name="decoder_input_layer")
#         if(self.int_layers>0):
#             layer1_dim_mid = self.int_layer_dims[-1]
#             decoder_layer = layers.Dense(layer1_dim_mid, activation="relu",name='decoder_layer_1')(decoder_input_layer)
#             if(self.int_layers>1):
#                 for this_int_layer in range(2,self.int_layers+1):
#                     thislayer_dim_mid = self.int_layer_dims[-this_int_layer]
#                     decoder_layer = layers.Dense(round(thislayer_dim_mid), activation="relu",name='decoder_layer_'+str(this_int_layer))(decoder_layer)
         
#         decoder_output_layer = layers.Dense(self.input_dim, activation=self.decoder_output_activation,name='decoder_output_layer')(decoder_layer)
#         self.decoder = Model(inputs=decoder_input_layer, outputs=decoder_output_layer, name="decoder_vae")
            
#         outputs = self.decoder(latent_resampled)




      

  

      
      
#       self.vae = Model(inputs=encoder_input_layer, outputs=outputs, name="vae")
#       optimizer = optimizers.Adam(learning_rate=self.learning_rate)
#       print('I think the function VAE_n_layer is in test config and does not return a VAE??')






















        
        
        
#         def FAE_loss(y_true, y_pred):
#             the_latent_space = self.encoder(y_true)
#             #np_latent_space = the_latent_space.numpy()
            
#             column_max_min_tf = tf.reduce_min(tf.reduce_max(the_latent_space,axis=0))
            
#             #zeropen = 1 - np_latent_space.max(axis=0).min()
            
#             #zeropentf = 1 - tf.math.tanh(    )
            
#             zeropentf = 1# + tf.experimental.numpy.heaviside(column_max_min_tf,10) + tf.experimental.numpy.heaviside(-column_max_min_tf,10)
            
#             #print(zeropen)
#             #print(zeropentf)

            
#             latent_columns_decoded = tf.zeros(tf.shape(y_true))            
#             # latent_factors_decoded = tf.zeros([self.latent_dim,y_true.shape.as_list()[1]]).numpy()
            
#             for i in np.arange(self.latent_dim):
                
#             #TRY each column in latent space is a factor
#                 split1, split2, split3 = tf.split(the_latent_space,[i,1,(-1+self.latent_dim-i)],1)    
#                 zeros1 = tf.zeros(tf.shape(split1))
#                 zeros3 = tf.zeros(tf.shape(split3))
#                 onecol_withzeros = tf.concat([zeros1,split2,zeros3],axis=1)
            
#                 thiscol_decoded = self.decoder(onecol_withzeros)
#                 latent_columns_decoded = latent_columns_decoded + thiscol_decoded
#                 # thiscol_decoded_mean = tf.reduce_mean(thiscol_decoded,axis=0)
#                 # latent_factors_decoded[i,:] = thiscol_decoded_mean.numpy()

            
#             mse_cols = tf.reduce_mean(tf.metrics.mean_squared_error(y_true, latent_columns_decoded))
#             mse_total = tf.reduce_mean(tf.metrics.mean_squared_error(y_true, y_pred))
            
#             #Need to penalise it for having any columns that are all zero in the latent space
#             #zeropen = 1 - np_latent_space.max(axis=0).min()
#             #print(zeropen)
            
#             #cross_correlation between columns when decoded, 1 minus the r2
#             #pdb.set_trace()
            
#             return zeropentf*(mse_total + mse_cols)
#         #    return mse_cols 
        
        
        
#         self.ae = Model(inputs=encoder_input_layer, outputs=outputs, name="ae")
#         optimizer = optimizers.Adam(learning_rate=self.learning_rate)
    
#         #COMPILING
#         #self.ae.compile(optimizer, loss='mse')
#         self.ae.compile(optimizer, loss=FAE_loss,run_eagerly=True)
#         #self.ae.add_loss(tf.keras.losses.MeanSquaredError)
    
#     #Default batch size is the whole dataset
#     def fit_model(self, x_train,x_test='DEFAULT',batch_size=int(1e10),epochs=30,verbose='auto'):
#         if(str(x_test)=='DEFAULT'):
#             _history = self.ae.fit(x_train,x_train,
#                          shuffle=True,
#                          epochs=epochs,
#                          batch_size=batch_size,
#                          validation_data=(x_train,x_train),verbose=verbose)
#         else:
#             _history = self.ae.fit(x_train,x_train,
#                         shuffle=True,
#                         epochs=epochs,
#                         batch_size=batch_size,
#                         validation_data=(x_test,x_test),verbose=verbose)
            
#         return _history


#     def encode(self, data,batch_size=100):
#         return self.encoder.predict(data, batch_size=batch_size)
    
#     def decode(self, data,batch_size=100):
#         return self.decoder.predict(data, batch_size=batch_size)

    
# %%n-layer autoencoder class with nmf in the loss function (doesnt work well)
from sklearn.decomposition import NMF
class NMFAE_n_layer():
    def __init__(self,hp='DEFAULT',input_dim=964,latent_dim=3,int_layers=1,int_layer_dims='DEFAULT',latent_activation='relu',decoder_output_activation='relu'):
        if(hp=='DEFAULT'):#Use parameters from the list
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            self.int_layers = int_layers
            self.latent_activation=latent_activation
            self.decoder_output_activation = decoder_output_activation
            self.nmf = NMF(self.latent_dim)
            self.learning_rate = 1e-2
            
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
                
        
        else:   #Use kerastuner hyperparameters
            self.input_dim = hp.get('input_dim')
            self.latent_dim = hp.get('latent_dim')
            self.int_layers = hp.get('intermediate_layers')
            self.int_layer_dims = [val.value for key, val in hp._space.items() if 'intermediate_dim' in key]
            self.learning_rate = hp.get('learning_rate')
            self.decoder_output_activation = hp.get('decoder_output_activation')
            self.latent_activation = hp.get('latent_activation')
        
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

        latent_layer = layers.Dense(self.latent_dim, activation=self.latent_activation,name='latent_layer')(encoder_layer)
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
        
        outputs = self.decoder(latent_layer)#Original outputs line        
        
        
        def FAE_loss(y_true, y_pred):
            the_latent_space = self.encoder(y_true)
            np_latent_space = the_latent_space.numpy()

            model = self.nmf
            W = model.fit_transform(np_latent_space)
            H = model.components_

            Factor0_lat_mtx = np.outer(W.T[0], H[0])
            Factor1_lat_mtx = np.outer(W.T[1], H[1])
            Factor2_lat_mtx = np.outer(W.T[2], H[2])
            Factor0_mtx_decod = self.decoder(Factor0_lat_mtx)
            Factor1_mtx_decod = self.decoder(Factor1_lat_mtx)
            Factor2_mtx_decod = self.decoder(Factor2_lat_mtx)
            
            y_pred_factorsum = Factor0_mtx_decod + Factor1_mtx_decod + Factor2_mtx_decod
            
            mse_nmfsum = tf.reduce_mean(tf.metrics.mean_squared_error(y_true, y_pred_factorsum))
            mse_total = tf.reduce_mean(tf.metrics.mean_squared_error(y_true, y_pred))
            return mse_nmfsum + mse_total
        
        
        
        self.ae = Model(inputs=encoder_input_layer, outputs=outputs, name="ae")
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
    
        #COMPILING
        #self.ae.compile(optimizer, loss='mse')
        self.ae.compile(optimizer, loss=FAE_loss,run_eagerly=True)
        #self.ae.add_loss(tf.keras.losses.MeanSquaredError)
    
    #Default batch size is the whole dataset
    def fit_model(self, x_train,x_test='DEFAULT',batch_size=int(1e10),epochs=30,verbose='auto'):
        if(str(x_test)=='DEFAULT'):
            _history = self.ae.fit(x_train,x_train,
                         shuffle=True,
                         epochs=epochs,
                         batch_size=batch_size,
                         validation_data=(x_train,x_train),verbose=verbose)
        else:
            _history = self.ae.fit(x_train,x_train,
                        shuffle=True,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_test,x_test),verbose=verbose)
            
        return _history


    def encode(self, data,batch_size=100):
        return self.encoder.predict(data, batch_size=batch_size)
    
    def decode(self, data,batch_size=100):
        return self.decoder.predict(data, batch_size=batch_size)
    

#%%Kerastuner model builder for AE_n_layer
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


    
#######################
####RANDOM USEFUL FUNCTIONS#####
#######################
#%%Calculate the loss per sample for an autoencoder
#x and y must be numpy arrays
def AE_calc_loss_per_sample(ae_model,x,y):
    loss_per_sample = []
    for i in range(x.shape[0]):
        loss_i = ae_model.evaluate(x=x[i:i+1],
                                 y=y[i:i+1],
                                 batch_size=None,
                                 verbose=0,
                                 steps=1
                                 )
        loss_per_sample.append(loss_i)
    return loss_per_sample



#%%
###############################
#####DATA PREPROCESSING########
###############################

#Pipeline transformer, just multiplies the data by a factor
def multiply_transformer(factor):
    return FunctionTransformer(lambda x: np.multiply(x,factor))

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
    
    newdf = newdf * signoise
       


    #Efficient version
    #newdf = pd.DataFrame(np.repeat(df.values,num_copies,axis=0)) * np.random.normal(1, sig_noise_pct/100, [num_copies*num_rows,num_cols])
    return newdf




