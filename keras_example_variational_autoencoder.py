'''This script demonstrates how to build a variational autoencoder with Keras.
Original URL was https://raw.githubusercontent.com/keras-team/keras/keras-2/examples/variational_autoencoder.py
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# from keras.layers import Input, Dense, Lambda
# from keras.models import Model
# from keras import backend as K
# from keras import metrics
# from keras.datasets import mnist

#Set tensorflow logging to warnings only
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.datasets import mnist

import pandas as pd

import tensorflow as tf
import kerastuner as kt

from sklearn.preprocessing import RobustScaler, StandardScaler,FunctionTransformer,MinMaxScaler
from sklearn.pipeline import Pipeline


import os
os.chdir('C:/Work/Python/Github/Orbitrap_clustering')
from ae_functions import *

# %%Load data

path='C:/Users/mbcx5jt5/Google Drive/Shared_York_Man2/'
df_beijing_raw, df_beijing_filters, df_beijing_metadata = beijing_load(
    path + 'BJ_UnAmbNeg9.1.1_20210505-Times_Fixed.xlsx',path + 'BJ_UnAmbNeg9.1.1_20210505-Times_Fixed.xlsx',
    peaks_sheetname="Compounds",metadata_sheetname="massloading_Beijing")

# %%Scale data
pipe = MinMaxScaler()
#pipe = StandardScaler()
pipe.fit(df_beijing_filters.to_numpy())
df_beijing_scaled = pd.DataFrame(pipe.transform(df_beijing_filters.to_numpy()))
df_beijing_scaled = df_beijing_scaled.fillna(0)
ae_input=df_beijing_scaled.to_numpy()


# %%WORKING VAE BASIC EXAMPLE FOR MY DATASET


# batch_size = 100
# original_dim = ae_input.shape[1]
# latent_dim = 2
# intermediate_dim = 256
# epochs = 100
# epsilon_std = 1.0

# #x = Input(batch_shape=(batch_size, original_dim))
# x = Input(shape=(original_dim,))
# h = Dense(intermediate_dim, activation='relu')(x)
# z_mean = Dense(latent_dim)(h)
# z_log_var = Dense(latent_dim)(h)


# def sampling(args):
#     z_mean, z_log_var = args
#     batch = tf.shape(z_mean)[0]
#     dim = tf.shape(z_mean)[1]
#     #epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
#                               #stddev=epsilon_std)
#     epsilon = K.random_normal(shape=(batch, dim), mean=0.,
#                               stddev=epsilon_std)
#     return z_mean + K.exp(z_log_var / 2) * epsilon

# # note that "output_shape" isn't necessary with the TensorFlow backend
# z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# # we instantiate these layers separately so as to reuse them later
# decoder_h = Dense(intermediate_dim, activation='relu')
# decoder_mean = Dense(original_dim, activation='sigmoid')
# h_decoded = decoder_h(z)
# x_decoded_mean = decoder_mean(h_decoded)


# def vae_loss(x, x_decoded_mean):
#     mse_loss = original_dim * metrics.mse(x, x_decoded_mean)
#     #xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
#     kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
#     return mse_loss + kl_loss

# vae = Model(x, x_decoded_mean)
# vae.compile(optimizer='rmsprop', loss=vae_loss)




# %%Define VAE FUNCTINO

# batch_size = 100
# original_dim = ae_input.shape[1]
# latent_dim = 2
# intermediate_dim = 256
# intermediate_dim2 = 64
# epochs = 100
# epsilon_std = 1.0





# def model_builder(hp):
    
#     #Hyperparameters to tune
#     intermediate_dim1_hp_units = hp.Int('units1', min_value=400, max_value=800, step=20)
#     intermediate_dim2_hp_units = hp.Int('units2', min_value=200, max_value=400, step=20)
#     intermediate_dim3_hp_units = hp.Int('units3', min_value=100, max_value=200, step=10)
#     latent_dim_units = hp.Int('latent_units', min_value=10, max_value=100, step=5)
#     #Standard encoder layers
#     original_inputs = tf.keras.Input(shape=(original_dim,), name="encoder_input")
#     layer1_vae = layers.Dense(intermediate_dim1_hp_units, activation="relu")(original_inputs)
#     layer2_vae = layers.Dense(intermediate_dim2_hp_units, activation="relu")(layer1_vae)
#     layer3_vae = layers.Dense(intermediate_dim3_hp_units, activation="relu")(layer2_vae)
#     layer4_vae = layers.Dense(latent_dim_units, activation="relu")(layer3_vae)#Make this sigmoid to make latent space between 0 - 1
    
#     #Latent space meand and log variance
#     z_mean = Dense(latent_dim)(layer4_vae)
#     z_log_var = Dense(latent_dim)(layer4_vae)
    
#     def sampling(args):
#         z_mean, z_log_var = args
#         batch = tf.shape(z_mean)[0]
#         dim = tf.shape(z_mean)[1]
#         #epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
#                                   #stddev=epsilon_std)
#         epsilon = K.random_normal(shape=(batch, dim), mean=0.,
#                                   stddev=epsilon_std)
#         return z_mean + K.exp(z_log_var / 2) * epsilon

#     # note that "output_shape" isn't necessary with the TensorFlow backend
#     #z is the sampling layer
#     z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

#     # we instantiate these layers separately so as to reuse them later
#     decoder_vae_layer1 = Dense(intermediate_dim1_hp_units, activation='relu')
#     decoder_mean = Dense(original_dim, activation='sigmoid')
    
#     WHAT IS h? LOOK AT THAT PRESENTATION
#     https://hpc.nih.gov/training/handouts/DL_by_Example3_20210825.pdf
#     h_decoded = decoder_vae_layer1(z)
#     x_decoded_mean = decoder_mean(h_decoded)
    
#     def vae_loss(x, x_decoded_mean):
#         mse_loss = original_dim * metrics.mse(x, x_decoded_mean)
#         #xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
#         kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
#         return mse_loss + kl_loss


#     hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
#     optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
   
#     vae = Model(original_inputs, x_decoded_mean)    
#     vae.compile(optimizer=optimizer, loss=vae_loss)
    
#     return vae


 # %%Define VAE JT

batch_size = 100
original_dim = ae_input.shape[1]
#latent_dim = 2
intermediate_dim = 256
intermediate_dim2 = 64
epochs = 100
epsilon_std = 1.0


class MyHyperModel(kt.HyperModel):

    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        #pdb.set_trace()
        
    def build(self,hp): 
        latent_dim = self.latent_dim
        #Hyperparameters to tune
        intermediate_dim1_hp_units = hp.Int('units1', min_value=100, max_value=200, step=20)
        intermediate_dim2_hp_units = hp.Int('units2', min_value=50, max_value=100, step=10)
        intermediate_dim3_hp_units = hp.Int('units3', min_value=25, max_value=50, step=5)
        #latent_dim_units = hp.Int('latent_units', min_value=10, max_value=100, step=5)
        
        #Standard encoder layers
        original_inputs = tf.keras.Input(shape=(original_dim,), name="encoder_input")
        layer1_vae = Dense(intermediate_dim1_hp_units, activation="relu")(original_inputs)
        layer2_vae = Dense(intermediate_dim2_hp_units, activation="relu")(layer1_vae)
        layer3_vae = Dense(intermediate_dim3_hp_units, activation="relu")(layer2_vae)
        #layer4_vae = layers.Dense(latent_dim_units, activation="relu")(layer3_vae)#Make this sigmoid to make latent space between 0 - 1
        
        #Latent space mean and log variance
        z_mean = Dense(latent_dim)(layer3_vae)
        z_log_var = Dense(latent_dim)(layer3_vae)
        
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            #epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                      #stddev=epsilon_std)
            epsilon = K.random_normal(shape=(batch, dim), mean=0.,
                                      stddev=epsilon_std)
            return z_mean + K.exp(z_log_var / 2) * epsilon
      
        #z is the sampling layer
        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
        
        encoder = Model(original_inputs, [z_mean, z_log_var, z], name="encoder_vae")
        
    
    
        #BUILD THE DECODER
        latent_inputs = Input(shape=(latent_dim,))
        # we instantiate these layers separately so as to reuse them later
        decoder_vae_layer3 = Dense(intermediate_dim3_hp_units, activation='relu')(latent_inputs)
        decoder_vae_layer2 = Dense(intermediate_dim2_hp_units, activation='relu')(decoder_vae_layer3)
        decoder_vae_layer1 = Dense(intermediate_dim1_hp_units, activation='relu')(decoder_vae_layer2)
        #decoder_mean = Dense(original_dim, activation='sigmoid')
        decoder_outputs = Dense(original_dim, activation='sigmoid')(decoder_vae_layer1) #This should be a relu or tanh depending on the input
        decoder_ae = Model(inputs=latent_inputs, outputs=decoder_outputs, name="decoder_vae")
        AE_outputs = decoder_ae(z)#try this
        #AE_outputs = decoder_outputs(z)#doesnt work
        
        
        
        
        def vae_loss(inputs, outputs):
            mse_loss = original_dim * metrics.mse(inputs, outputs)
            #xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return mse_loss + kl_loss
    
    
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
       
        vae = Model(original_inputs, AE_outputs)    
        vae.compile(optimizer=optimizer, loss=vae_loss)
        
        return vae


#     decoder_h = Dense(intermediate_dim, activation='relu')
#     decoder_mean = Dense(original_dim, activation='sigmoid')
#     h_decoded = decoder_h(z)
#     x_decoded_mean = decoder_mean(h_decoded)


# %%Define VAE DEFINITELY WORKS WITH KERASTUNER

batch_size = 100
original_dim = ae_input.shape[1]
epochs = 100
epsilon_std = 1.0

def model_builder(hp):
    
    intermediate_dim = hp.Int('units1', min_value=50, max_value=250, step=50)
    latent_dim = hp.Int('units2', min_value=5, max_value=50, step=5)
    
    x = Input(shape=(original_dim,))
    h = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)


    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        #epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                  #stddev=epsilon_std)
        epsilon = K.random_normal(shape=(batch, dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')#I dont know why spyder thinks this is an error but it still works
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    def vae_loss(x, x_decoded_mean):
        mse_loss = metrics.mse(x, x_decoded_mean)
        #xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return mse_loss + kl_loss
       
    def mse_loss(x,x_decoded_mean):
        mse_loss = metrics.mse(x, x_decoded_mean)
        return mse_loss
    
    def kl_loss(x,x_decoded_mean):
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return kl_loss


    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
    
    vae = Model(x, x_decoded_mean)
    

    vae.compile(optimizer=optimizer, loss=vae_loss,metrics=[mse_loss,kl_loss])

    
    return vae

# # %%KERASTUNER BASIC VAE WITH ARGUMENT WORKS DO NOT CHANGE

# batch_size = 100
# original_dim = ae_input.shape[1]
# epochs = 100
# epsilon_std = 1.0


# class MyHyperModel(kt.HyperModel):

#     def __init__(self,latent_dim):
#         self.latent_dim = latent_dim
#         #pdb.set_trace()

#     def build(self,hp):
#         latent_dim = self.latent_dim
  
#         intermediate_dim1 = hp.Int('units1', min_value=250, max_value=500, step=50)
#         #intermediate_dim2 = hp.Int('units2', min_value=20, max_value=100, step=20)
        
#         # x = Input(shape=(original_dim,))
#         # h1 = Dense(intermediate_dim1, activation='relu')(x)
#         # h2 = Dense(intermediate_dim2, activation='relu')(h1)
#         # z_mean = Dense(latent_dim)(h2)
#         # z_log_var = Dense(latent_dim)(h2)
        
#         x = Input(shape=(original_dim,))
#         h = Dense(intermediate_dim1, activation='relu')(x)
#         z_mean = Dense(latent_dim)(h)
#         z_log_var = Dense(latent_dim)(h)
    
#         #pdb.set_trace()
    
#         def sampling(args):
#             z_mean, z_log_var = args
#             batch = tf.shape(z_mean)[0]
#             dim = tf.shape(z_mean)[1]
#             #epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
#                                       #stddev=epsilon_std)
#             epsilon = K.random_normal(shape=(batch, dim), mean=0.,
#                                       stddev=epsilon_std)
#             return z_mean + K.exp(z_log_var / 2) * epsilon
    
#         # note that "output_shape" isn't necessary with the TensorFlow backend
#         z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    
#         # we instantiate these layers separately so as to reuse them later
#         decoder_h = Dense(intermediate_dim1, activation='relu')
#         decoder_mean = Dense(original_dim, activation='sigmoid')
#         h_decoded = decoder_h(z)
#         x_decoded_mean = decoder_mean(h_decoded)
        
    
#         ###############
#         #Loss and compiling
    
#         #@tf.autograph.experimental.do_not_convert
#         def vae_loss(x, x_decoded_mean):
#             mse_loss = metrics.mse(x, x_decoded_mean)
#             #xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
#             kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
#             return mse_loss + kl_loss
        
#         #@tf.autograph.experimental.do_not_convert   
#         def mse_loss(x,x_decoded_mean):
#             mse_loss = metrics.mse(x, x_decoded_mean)
#             return mse_loss
        
#         #@tf.autograph.experimental.do_not_convert
#         def kl_loss(x,x_decoded_mean):
#             kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
#             return kl_loss
        
        
#         vae = Model(x, x_decoded_mean)

        
#         hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
#         optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
        
        
#         #vae.compile(optimizer=optimizer)
#         vae.compile(optimizer=optimizer, loss=vae_loss,metrics=[mse_loss,kl_loss])

        
        
#         return vae
# %%test
class VAE_meta:
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        #epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                  #stddev=epsilon_std)
        epsilon = K.random_normal(shape=(batch, dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon


    
    # %%KERASTUNER BASIC VAE WITH ARGUMENT WORKS DO NOT CHANGE

    batch_size = 100
    original_dim = ae_input.shape[1]
    epsilon_std = 1.0


    class MyHyperModel(kt.HyperModel):

        def __init__(self,original_dim,latent_dim):
            self.original_dim = original_dim
            self.latent_dim = latent_dim
            #pdb.set_trace()

        def build(self,hp):
            #latent_dim = self.latent_dim
      
            intermediate_dim1 = hp.Int('layer1_dim', min_value=250, max_value=500, step=50)
            intermediate_dim2 = hp.Int('layer2_dim', min_value=20, max_value=100, step=20)
            
            x = Input(shape=(self.original_dim,),name='encoder_input')
            h1 = Dense(intermediate_dim1, activation='relu',name='layer1')(x)
            h2 = Dense(intermediate_dim2, activation='relu',name='layer2')(h1)
            z_mean = Dense(self.latent_dim,name='latent_mean')(h2)
            z_log_var = Dense(self.latent_dim,name='latent_log_var')(h2)

            def sampling(args):
                z_mean, z_log_var = args
                batch = tf.shape(z_mean)[0]
                dim = tf.shape(z_mean)[1]
                #epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                          #stddev=epsilon_std)
                epsilon = K.random_normal(shape=(batch, dim), mean=0.,
                                          stddev=epsilon_std)
                return z_mean + K.exp(z_log_var / 2) * epsilon
            
            #Resample
            z = Lambda(sampling, output_shape=(self.latent_dim,),name='resampled_latent_mean')([z_mean, z_log_var])
        
            # we instantiate these layers separately so as to reuse them later
            #decoder_input_layer = Input(shape=(latent_dim,),name='decoder_input')
            decoder_h2 = Dense(intermediate_dim2, activation='relu',name='layer2_decoded')
            decoder_h1 = Dense(intermediate_dim1, activation='relu',name='layer1_decoded')
            decoder_mean = Dense(original_dim, activation='sigmoid',name='output_decoded')
            h2_decoded = decoder_h2(z)
            h1_decoded = decoder_h1(h2_decoded)
            x_decoded_mean = decoder_mean(h1_decoded)
            
            #self.decoder = tf.keras.Model(inputs=decoder_input_layer, outputs=x_decoded_mean, name="decoder")
            
            ###############
            #Loss and compiling
            #WHY DO THESE LINES NEED TO BE HERE IT
            #WONT RUN IF THEY ARE NOT HERE. THEY CAN BE BLANK ITS OK

            def vae_loss(x, x_decoded_mean):
                mse_loss = metrics.mse(x, x_decoded_mean)
                #xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
                kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
                return mse_loss + kl_loss
            
            # def mse_loss(x,x_decoded_mean):
            #     mse_loss = metrics.mse(x, x_decoded_mean)
            #     return mse_loss
            
            def kl_loss(x,x_decoded_mean):
                kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
                return kl_loss
            
            
            vae = Model(x, x_decoded_mean)

            
            hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
            optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
            
            vae.compile(optimizer=optimizer, loss=vae_loss,metrics=['mse',kl_loss])
            #pdb.set_trace()
            
            self.z_mean = z_mean
            vae.z_mean= z_mean
            self.encoder = tf.keras.Model(inputs=x, outputs=z_mean, name="encoder")
            
            
            return vae
            
        # This is here but uim not sure its working
        # well it isnt
        # can it work? Or do Ineed a totally separate non-kerastuner class? Probably
        #     def encode(self, data):
        #         encoded_data = self._session.run(self.encoder, feed_dict={self._input_layer:[data]})
        #         return encoded_data

# %%
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
#tf.autograph.experimental.do_not_convert

# %%Hypertune!
K.clear_session()
##############################
##TUNING HYPERPARAMETERS
##############################
latent_dim = 5
my_hyper_model = MyHyperModel(original_dim=ae_input.shape[1],latent_dim = latent_dim)

#tuner = kt.Hyperband(model_builder,
tuner = kt.Hyperband(my_hyper_model,
                     objective='val_loss',
                    max_epochs=3,
                    factor=3,
                    directory=os.path.normpath('C:/work/temp/keras'),
                    overwrite=True)

#This gives an error at the end on windows but don't worry about it
tuner.search(ae_input, ae_input, epochs=30, validation_data=(ae_input, ae_input))

# %% Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('layer1_dim')}, the second {best_hps.get('layer2_dim')}, and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# print(f"""
# The hyperparameter search is complete. The optimal number of units in the first densely-connected
# layer is {best_hps.get('units1')}, the second {best_hps.get('units2')}, third {best_hps.get('units3')}, latent {best_hps.get('latent_units')} and the optimal learning rate for the optimizer
# is {best_hps.get('learning_rate')}.
# """)

# %%Build the model with the optimal hyperparameters and train it on the data for 30 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(ae_input, ae_input, epochs=30, validation_data=(ae_input, ae_input))
val_acc_per_epoch = history.history['val_loss']
best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))


# %%define hypermodel
hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
history_new = hypermodel.fit(ae_input, ae_input, epochs=best_epoch, validation_data=(ae_input, ae_input))
# plot loss history
loss = history_new.history['loss']
val_loss = history_new.history['val_loss']
mse_loss = history_new.history['mse']
kl_loss = history_new.history['kl_loss']
epochs = range(len(loss))

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.plot(epochs, mse_loss, 'r', label='MSE loss')
plt.plot(epochs, kl_loss, 'k', label='KL loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# %%Variational autoencoder class

class VAE_2layer(object):
    
    def __init__(self,hyperparams):
        #pdb.set_trace()
        self.original_dim = hyperparams[0]
        self.latent_dim = hyperparams[1]
        self.intermediate_dim1 = hyperparams[2]
        self.intermediate_dim2 = hyperparams[3]
        self.learning_rate = hyperparams[4]
        
        self.x = None
        self.h1 = None
        self.h2 = None
        self.z_mean = None
        self.z_log_var = None
        self.decoder_h1 = None
        self.decoder_h2 = None
        self.decoder_mean = None
        
        self.batch_size = 50
        
        self.encoder = None
        self.vae = None
        
        K.clear_session()
        self.build_model()
     
    def sampling(self, args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0., stddev=1)
            return z_mean + K.exp(z_log_var / 2) * epsilon
    
    # def kl_loss2(self):
    #     #def kl_loss(x,x_decoded_mean):
    #     kl_loss2 = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
    #     return kl_loss2
        
    def build_model(self):
        #Define the encoder
        self.x = Input(shape=(self.original_dim,),name='encoder_input')
        self.h1 = Dense(self.intermediate_dim1, activation='relu',name='layer1')(self.x)
        self.h2 = Dense(self.intermediate_dim2, activation='relu',name='layer2')(self.h1)
        self.z_mean = Dense(self.latent_dim,name='latent_mean')(self.h2)
        self.z_log_var = Dense(self.latent_dim,name='latent_log_var')(self.h2)
        #self.encoder = tf.keras.Model(inputs=self.x, outputs=self.z_mean, name="encoder")
        
        #Sampling function and layer, for training only
        # def sampling(args):
        #     z_mean, z_log_var = args
        #     batch = tf.shape(z_mean)[0]
        #     dim = tf.shape(z_mean)[1]
        #     #epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
        #                               #stddev=epsilon_std)
        #     epsilon = K.random_normal(shape=(batch, dim), mean=0.,
        #                               stddev=1.)
        #     return z_mean + K.exp(z_log_var / 2) * epsilon
        
        print("F")
        # note that "output_shape" isn't necessary with the TensorFlow backend
        #z = Lambda(sampling, output_shape=(self.latent_dim,),name='resampled_latent_mean')([self.z_mean, self.z_log_var])
        z = Lambda(self.sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_var])
        
        
        #Define the decoder
        #decoder_input_layer = Input(shape=(self.latent_dim,),name='decoder_input')
        self.decoder_h2 = Dense(self.intermediate_dim2, activation='relu',name='layer2_decoded')
        self.decoder_h1 = Dense(self.intermediate_dim1, activation='relu',name='layer1_decoded')
        self.decoder_mean = Dense(self.original_dim, activation='sigmoid',name='output_decoded')
        #deco = decoder_input(z)
        h2_decoded = self.decoder_h2(z)
        h1_decoded = self.decoder_h1(h2_decoded)
        x_decoded_mean = self.decoder_mean(h1_decoded)
        print("L")
       # self._decoder = tf.keras.Model(inputs=decoder_input_layer, outputs=self.x_decoded_mean, name="decoder")
        print("M")
        #@tf.autograph.experimental.do_not_convert
        def vae_loss(x, x_decoded_mean):
            mse_loss = metrics.mse(x, x_decoded_mean)
            kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
            return kl_loss#mse_loss# + kl_loss
        
        def mse_loss(x,x_decoded_mean):
            mse_loss = metrics.mse(x, x_decoded_mean)
            return mse_loss
        
        
        #@tf.autograph.experimental.do_not_convert
        def kl_loss(x,x_decoded_mean):
            kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
            return kl_loss
        
        
        self.vae = Model(self.x, x_decoded_mean)
        
        # # Compute VAE loss
        # xent_loss = self.original_dim * metrics.binary_crossentropy(self.x, x_decoded_mean)
        # kl_loss3 = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        # vae_loss2 = K.mean(xent_loss + kl_loss3)
        # print("S")
        
        # self.vae.add_loss(vae_loss2)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)       
        print("T")
        #self.vae.add_loss(vae_loss2)
        print("U")
        #self.vae.compile(optimizer=optimizer, loss=vae_loss2,metrics=[mse_loss,kl_loss])
        #self.vae.compile(optimizer=optimizer, loss='mse',metrics=['mse',{"kl_loss":kl_loss3}])
        #self.vae.compile(optimizer=optimizer,loss=vae_loss,metrics=['mse',kl_loss])
        self.vae.compile(optimizer=optimizer,loss=vae_loss,metrics=['mse',kl_loss])
        #self.vae.compile(optimizer=optimizer)
        print("V")
        self.vae.summary()
        
        
    def train(self, input_train, input_test='DEFAULT', batch_size=100, epochs=30):    
        if(input_test == 'DEFAULT'):
            self.vae.fit(input_train, 
                            input_train,
                            epochs = epochs,
                            batch_size=batch_size,
                            shuffle=True)
        else:
            self.vae.fit(input_train, 
                        input_train,
                        epochs = epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(
                                input_test, 
                                input_test))
        
    def encode(self, data):
        # build a model to project inputs on the latent space
        self.encoder = Model(self.x, self.z_mean)
        return self.encoder.predict(data, batch_size=self.batch_size)

    def decode(self, data):
        # build a data generator that can sample from the learned distribution
        decoder_input = Input(shape=(self.latent_dim,))
        # _h_decoded = self.decoder_h(decoder_input)
        # _x_decoded_mean = self.decoder_mean(_h_decoded)
        _h2_decoded = self.decoder_h2(decoder_input)
        _h1_decoded = self.decoder_h1(_h2_decoded)
        _x_decoded_mean = self.decoder_mean(_h1_decoded)
        generator = Model(decoder_input, _x_decoded_mean)
        return generator.predict(data)



# %%
hyperparams = [original_dim,latent_dim,best_hps.get('layer1_dim'),best_hps.get('layer2_dim'),best_hps.get('learning_rate')]
vae = VAE_2layer(hyperparams=hyperparams)



# %%FIT!
vae.train(ae_input,
        epochs=best_epoch)#,
        #batch_size=batch_size)#,
        #validation_data=(x_test, x_test))

# %%
latent_space = vae.encode(test)

vae_output = vae.decode(latent_space)



#%% Test with the MNIST dataset
hyperparams2 = hyperparams
hyperparams2[0] = 784
vae2 = VAE_2layer(hyperparams=hyperparams2) 
vae2.train(x_train,epochs=3)
latent_space2 = vae2.encode(x_test)


# %%

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
