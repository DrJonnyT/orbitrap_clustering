'''
Code source: https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
from https://raw.githubusercontent.com/mattiacampana/Autoencoders/master/models/vae.py

 #Reference

 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
#from __future__ import print_function

from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.initializers import Zeros
import tensorflow as tf

#Set tensorflow logging to warnings only
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

#%%

class Vae_vanilla:

    def __init__(self, input_dim, batch_size=100, latent_dim=2, intermediate_dim=256, epochs=50, epsilon_std=1.0):

        self.original_dim = input_dim
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.epochs = epochs
        self.epsilon_std = epsilon_std
        self.vae = None
        self.encoder = None
        self.decoder = None
        self.x = None # Input layer
        self.z_mean = None
        self.z_log_var = None
        self.decoder_h = None
        self.decoder_mean = None

        K.clear_session()

        self.build_model()

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0., stddev=self.epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def build_model(self):

        self.x = Input(shape=(self.original_dim,))
        h = Dense(self.intermediate_dim, activation='relu')(self.x)
        self.z_mean = Dense(self.latent_dim)(h)
        self.z_log_var = Dense(self.latent_dim)(h)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self.sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_var])

        # we instantiate these layers separately so as to reuse them later
        self.decoder_h = Dense(self.intermediate_dim, activation='relu')
        self.decoder_mean = Dense(self.original_dim, activation='sigmoid')
        h_decoded = self.decoder_h(z)
        x_decoded_mean = self.decoder_mean(h_decoded)

        # instantiate VAE model
        self.vae = Model(self.x, x_decoded_mean)

        # Compute VAE loss
        xent_loss = self.original_dim * metrics.binary_crossentropy(self.x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)

        optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer=optimizer,loss='binary_crossentropy')#Added MSE loss
        self.vae.summary()

    def fit_model(self, x_train):

        self.vae.fit(x_train,
                     shuffle=True,
                     epochs=self.epochs,
                     batch_size=self.batch_size)

    def encode(self, data):
        # build a model to project inputs on the latent space
        self.encoder = Model(self.x, self.z_mean)

        return self.encoder.predict(data, batch_size=self.batch_size)

    def decode(self, data):
        # build a data generator that can sample from the learned distribution
        decoder_input = Input(shape=(self.latent_dim,))
        _h_decoded = self.decoder_h(decoder_input)
        _x_decoded_mean = self.decoder_mean(_h_decoded)
        generator = Model(decoder_input, _x_decoded_mean)

        return generator.predict(data)






class Vae:

    def __init__(self, input_dim, batch_size=100, latent_dim=2, intermediate_dim=256, epochs=50, epsilon_std=1.0):

        self.original_dim = input_dim
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.epochs = epochs
        self.epsilon_std = epsilon_std
        self.vae = None
        self.encoder = None
        self.decoder = None
        self.x = None # Input layer
        self.z_mean = None
        self.z_log_var = None
        self.decoder_h = None
        self.decoder_mean = None

        K.clear_session()

        self.build_model()

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0., stddev=self.epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def build_model(self):

        self.x = Input(shape=(self.original_dim,))
        h = Dense(self.intermediate_dim, activation='tanh')(self.x)
        self.z_mean = Dense(self.latent_dim)(h)
        self.z_log_var = Dense(self.latent_dim)(h)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self.sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_var])

        # we instantiate these layers separately so as to reuse them later
        self.decoder_h = Dense(self.intermediate_dim, activation='tanh')
        self.decoder_mean = Dense(self.original_dim, activation='tanh')
        h_decoded = self.decoder_h(self.z_mean)#changed z to z_mean, so no sampling layer
        x_decoded_mean = self.decoder_mean(h_decoded)

        # instantiate VAE model
        self.vae = Model(self.x, x_decoded_mean)

        # Compute VAE loss
        xent_loss = self.original_dim * metrics.binary_crossentropy(self.x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)
        
        def kl_loss2(x,x_decoded_mean):
            kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
            return kl_loss
        
        def vae_loss(x, x_decoded_mean):
            mse_loss = metrics.mse(x, x_decoded_mean)
            kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
            return mse_loss + kl_loss

        #optimizer = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        optimizer= optimizers.Adam(lr=0.01)

        #self.vae.add_loss(vae_loss)
        #self.vae.add_metric(kl_loss2,name='kl_loss')
        self.vae.compile(optimizer=optimizer,loss='mse',metrics=['mse',kl_loss2])
        self.vae.summary()

    def fit_model(self, x_train):

        self.vae.fit(x_train,
                     shuffle=True,
                     epochs=self.epochs,
                     batch_size=self.batch_size)

    def encode(self, data):
        # build a model to project inputs on the latent space
        self.encoder = Model(self.x, self.z_mean)

        return self.encoder.predict(data, batch_size=self.batch_size)

    def decode(self, data):
        # build a data generator that can sample from the learned distribution
        decoder_input = Input(shape=(self.latent_dim,))
        _h_decoded = self.decoder_h(decoder_input)
        _x_decoded_mean = self.decoder_mean(_h_decoded)
        generator = Model(decoder_input, _x_decoded_mean)

        return generator.predict(data)
    
    
    
    
# %%Really basic AE
class basic_AE():
    def __init__(self, input_dim, batch_size=100, latent_dim=2, intermediate_dim1=10, epochs=10):

        self.original_dim = input_dim
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.intermediate_dim1 = intermediate_dim1
        self.epochs = epochs
        self.ae = None
        self.encoder = None
        self.decoder = None
        self.input = None # Input layer
        self.encoder_layer1 = None
        #self.encoder_layer2 = None
        self.encoded = None
        self.decoder_layer1 = None
        #self.decoder_layer2 = None

        K.clear_session()

        self.build_model()
        
    def build_model(self):
        # self.input = Input(shape=(self.original_dim,),name='encoder_input')
        # #ENCODED is the latent layer
        # self.encoded = Dense(self.latent_dim, activation='linear',name='encoded_layer')(self.input)
        # #self.encoded = Dense(self.latent_dim, activation='linear',activity_regularizer=regularizers.l1(10e-5),name='encoded_layer')(self.input)
                
        # # "decoded" is the lossy reconstruction of the input
        # decoded = Dense(self.original_dim, activation='linear')(self.encoded)
        
        # # This model maps an input to its reconstruction
        # self.ae = Model(self.input, decoded)
        
        
        # #ENCODER MODEL
        # # This model maps an input to its encoded representation
        # self.encoder = Model(self.input, self.encoded)

        # #DECODER MODEL        ~
        # # This is our encoded (32-dimensional) input
        # encoded_input = Input(shape=(self.latent_dim,))
        # # Retrieve the last layer of the autoencoder model
        # decoder_layer = self.ae.layers[-1]
        # # Create the decoder model
        # self.decoder = Model(encoded_input, decoder_layer(encoded_input))
        
        
        
        # self.ae.compile(optimizer='adam', loss='mse')
        
        
        input_img = Input(shape=(self.original_dim,))
        # "encoded" is the encoded representation of the input
        self.encoded = Dense(8, activation='linear',activity_regularizer=regularizers.l1(10e-5))(input_img)
        self.encoded = Dense(4, activation='linear')(self.encoded)
        self.encoded = Dense(self.latent_dim, activation='linear')(self.encoded)

        # self.decoded = Dense(encoding_dim, activation='linear',activity_regularizer=regularizers.l1(10e-5))(self.encoded)
        # self.decoded = Dense(4, activation='linear')(self.decoded)
        # self.decoded = Dense(8, activation='linear')(self.decoded)
        # "decoded" is the lossy reconstruction of the input
        self.decoded = Dense(self.original_dim, activation='linear')(self.encoded)

        

        # This model maps an input to its encoded representation
        self.encoder = Model(input_img, self.encoded)

        # This is our encoded (32-dimensional) input
        encoded_input = Input(shape=(self.latent_dim,))
        print("B")
        # Retrieve the last layer of the autoencoder model
        decoder_layer = self.ae.layers[-1]
        print("C")
        # Create the decoder model
        self.decoder = Model(encoded_input, decoder_layer(encoded_input))
        print("A")
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        
        
        # This model maps an input to its reconstruction
        self.ae = Model(input_img, self.decoded)
        self.ae.compile(optimizer=opt, loss='mse')
        
    def fit_model(self, x_train):

       _history = self.ae.fit(x_train,
                    shuffle=True,
                    epochs=self.epochs,
                    batch_size=self.batch_size)
       return _history


    def encode(self, data):
        #self.encoder = Model(self.input, self.encoded)
        return self.encoder.predict(data, batch_size=self.batch_size)
    
    def decode(self, data):
        #encoded_input = Input(shape=(self.latent_dim,))
        #decoder_layer = self.ae.layers[-1]
        # Create the decoder model
        #self.decoder = Model(encoded_input, decoder_layer(encoded_input))
        return self.decoder.predict(data, batch_size=self.batch_size)    
        


# %%Load training data
# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

#%%Train and test basic AE
#x_train = np.outer(np.arange(1,1000),np.arange(10)) * 1e-4
x_train=np.outer(np.array([[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]),np.arange(1,4))
bae = basic_AE(x_train.shape[1],epochs=1000,latent_dim=1,intermediate_dim1=2)
history = bae.fit_model(x_train)
latent_space = bae.encode(x_train)
decoded_latent_space = bae.decode(latent_space)

bae_output = bae.ae.predict(x_train)

plt.figure()
plt.scatter(x_train,bae_output)
#plt.scatter(x_train,decoded_latent_space,c='r')

plt.show()

loss = history.history['loss']
epochs = range(len(loss))

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.title('Training loss')
plt.legend()
plt.show()


mse = tf.keras.metrics.mean_squared_error(x_train,bae_output)
print(mse)



#%%LITERALLY JUST THE TUTORIAL BASIC AE
# This is the size of our encoded representations
encoding_dim = 2

# This is our input image
input_img = Input(shape=(x_train.shape[1],))
# "encoded" is the encoded representation of the input
encoded = Dense(8, activation='linear',activity_regularizer=regularizers.l1(10e-5))(input_img)
encoded = Dense(4, activation='linear')(encoded)
encoded = Dense(encoding_dim, activation='linear')(encoded)

decoded = Dense(encoding_dim, activation='linear',activity_regularizer=regularizers.l1(10e-5))(encoded)
decoded = Dense(4, activation='linear')(decoded)
decoded = Dense(8, activation='linear')(decoded)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(x_train.shape[1], activation='linear')(encoded)

# This model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# This model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# This is our encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# Create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='mse')


history = autoencoder.fit(x_train, x_train,
                epochs=500,
                batch_size=256,
                shuffle=True,
                )


# Encode and decode some digits
# Note that we take them from the *test* set
encoded_imgs = encoder.predict(x_train)
decoded_imgs = decoder.predict(encoded_imgs)


plt.scatter(x_train,decoded_imgs)

loss = history.history['loss']
epochs = range(len(loss))

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.title('Training loss')
plt.legend()
plt.show()




# %%Make some data to train the model
#x_train = np.outer(np.arange(1,1000),np.arange(1,10)) * 1e-4
x_train = np.outer(np.arange(1,1000),np.ones(10)) * 1e-4
vae = Vae(x_train.shape[1],epochs=30,latent_dim=2,intermediate_dim=5)
vae.fit_model(x_train)
latent_space = vae.encode(x_train)
decoded_latent_space = vae.decode(latent_space)


# %%
vae_vanilla = Vae_vanilla(x_train.shape[1],epochs=100,latent_dim=2,intermediate_dim=5,epsilon_std=0.01)
vae_vanilla.fit_model(x_train)
latent_space_vanilla = vae_vanilla.encode(x_train)
decoded_latent_space_vanilla = vae_vanilla.decode(latent_space_vanilla)

# %%Make model normally, this works fine
vae = Vae(x_train.shape[1],epochs=3,latent_dim=10)
#vae.fit_model(x_train, validation_data=x_test)
vae.fit_model(x_train)

latent_space = vae.encode(x_test)# %%