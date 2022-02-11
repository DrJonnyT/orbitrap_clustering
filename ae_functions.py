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
import numpy as np
import pdb
import matplotlib.pyplot as plt

import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow.keras as keras
from tensorflow.keras import metrics
import kerastuner as kt
from sklearn.preprocessing import RobustScaler, StandardScaler,FunctionTransformer,MinMaxScaler
from sklearn.pipeline import Pipeline

import os

#%%

#######################
####FILE_LOADERS#######
#######################
def beijing_load(peaks_filepath,metadata_filepath,peaks_sheetname="DEFAULT",metadata_sheetname="DEFAULT",subtract_blank=True):
    if(metadata_sheetname=="DEFAULT"):
        df_beijing_metadata = pd.read_excel(metadata_filepath,engine='openpyxl',
                                           usecols='A:K',nrows=329,converters={'mid_datetime': str})
    else:
        df_beijing_metadata = pd.read_excel(metadata_filepath,engine='openpyxl',
                                       sheet_name=metadata_sheetname,usecols='A:K',nrows=329,converters={'mid_datetime': str})
    df_beijing_metadata['Sample.ID'] = df_beijing_metadata['Sample.ID'].astype(str)

    if(peaks_sheetname=="DEFAULT"):
        df_beijing_raw = pd.read_excel(peaks_filepath,engine='openpyxl')
    else:
        df_beijing_raw = pd.read_excel(peaks_filepath,engine='openpyxl',sheet_name=peaks_sheetname)
        
    #Filter out "bad" columns
    df_beijing_raw = orbitrap_filter(df_beijing_raw)

    #Cut some fluff columns out and make new df
    df_beijing_filters = df_beijing_raw.iloc[:,list(range(4,len(df_beijing_raw.columns)))].copy()
    index_backup = df_beijing_filters.index


    if(subtract_blank == True):
        #Extract blank
        beijing_blank = df_beijing_raw.iloc[:,320].copy()
        #Subtract blank
        df_beijing_filters = df_beijing_filters.subtract(beijing_blank.values,axis=0)
    
    
    #Set the index to sample ID for merging peaks with metadata
    sample_id = df_beijing_filters.columns.str.split('_|.raw').str[2]
    df_beijing_filters.columns = sample_id
    df_beijing_filters = df_beijing_filters.transpose()
    
    #Add on the metadata
    df_beijing_metadata.set_index(df_beijing_metadata["Sample.ID"].astype('str'),inplace=True)    
    df_beijing_filters = pd.concat([df_beijing_filters, df_beijing_metadata[['Volume_m3', 'Dilution_mL']]], axis=1, join="inner")

    #Divide the data columns by the sample volume and multiply by the dilution liquid volume (pinonic acid)
    df_beijing_filters = df_beijing_filters.div(df_beijing_filters['Volume_m3'], axis=0).mul(df_beijing_filters['Dilution_mL'], axis=0)
    df_beijing_filters.drop(columns=['Volume_m3','Dilution_mL'],inplace=True)
    df_beijing_filters['mid_datetime'] = pd.to_datetime(df_beijing_metadata['mid_datetime'],yearfirst=True)
    df_beijing_filters.set_index('mid_datetime',inplace=True)
    df_beijing_filters = df_beijing_filters.astype(float)   
    df_beijing_filters.columns = index_backup
    
    return df_beijing_raw, df_beijing_filters, df_beijing_metadata



def delhi_load(peaks_filepath,metadata_filepath,peaks_sheetname="DEFAULT",metadata_sheetname="DEFAULT",subtract_blank=True):
    if(metadata_sheetname=="DEFAULT"):
        df_delhi_metadata = pd.read_excel(metadata_filepath,engine='openpyxl',
                                           usecols='a:N',skiprows=0,nrows=108, converters={'mid_datetime': str})
    else:
        df_delhi_metadata = pd.read_excel(metadata_filepath,engine='openpyxl',
                                           sheet_name=metadata_sheetname,usecols='a:N',skiprows=0,nrows=108, converters={'mid_datetime': str})
    #df_delhi_metadata['Sample.ID'] = df_delhi_metadata['Sample.ID'].astype(str)

    if(peaks_sheetname=="DEFAULT"):
        df_delhi_raw = pd.read_excel(peaks_filepath,engine='openpyxl')
    else:
        df_delhi_raw = pd.read_excel(peaks_filepath,engine='openpyxl',sheet_name=peaks_sheetname)
    

    #Get rid of columns that are not needed
    df_delhi_raw.drop(df_delhi_raw.iloc[:,np.r_[0, 2:11, 14:18]],axis=1,inplace=True)
    
    
    
    #Fix column labels so they are consistent
    df_delhi_raw.columns = df_delhi_raw.columns.str.replace('DelhiS','Delhi_S')
    
    df_delhi_metadata.drop(labels="Filter ID.1",axis=1,inplace=True)
    df_delhi_metadata.set_index("Filter ID",inplace=True)
    #Get rid of bad filters, based on the notes
    df_delhi_metadata.drop(labels=["-","S25","S42","S51","S55","S68","S72"],axis=0,inplace=True)
     
    #Filter out "bad" columns
    df_delhi_raw = orbitrap_filter(df_delhi_raw)
    
    #Cut some fluff columns out and make new df
    df_delhi_filters = df_delhi_raw.iloc[:,list(range(4,len(df_delhi_raw.columns)))].copy()
    df_delhi_filters.columns = df_delhi_filters.columns.str.replace('DelhiS','Delhi_S')
    index_backup = df_delhi_filters.index
    
    if(subtract_blank == True):
        #Extract blanks
        df_delhi_raw_blanks = df_delhi_raw[df_delhi_raw.columns[df_delhi_raw.columns.str.contains('Blank')]] 
        #Subtract mean blank
        df_delhi_filters = df_delhi_filters.subtract(df_delhi_raw_blanks.transpose().mean().values,axis=0)
    
    
    sample_id = df_delhi_filters.columns.str.split('_|.raw').str[2]
    df_delhi_filters.columns = sample_id

    df_delhi_filters = df_delhi_filters.transpose()    
    
    #Add on the metadata    
    df_delhi_filters = pd.concat([df_delhi_filters, df_delhi_metadata[['Volume / m3', 'Dilution']]], axis=1, join="inner")

    #Divide the data columns by the sample volume and multiply by the dilution liquid volume (pinonic acid)
    df_delhi_filters = df_delhi_filters.div(df_delhi_filters['Volume / m3'], axis=0).mul(df_delhi_filters['Dilution'], axis=0)
    df_delhi_filters.drop(columns=['Volume / m3','Dilution'],inplace=True)
    
    #Some final QA
    df_delhi_filters['mid_datetime'] = pd.to_datetime(df_delhi_metadata['Mid-Point'],yearfirst=True)
    df_delhi_filters.set_index('mid_datetime',inplace=True)
    df_delhi_filters = df_delhi_filters.astype(float)
    df_delhi_filters.columns = index_backup

    return df_delhi_raw, df_delhi_filters, df_delhi_metadata


#######################
####PEAK FILTERING#####
#######################
# %%
#A function to filter a dataframe with all the peaks in
def orbitrap_filter(df_in):
    df_orbitrap_peaks = df_in.copy()
    #pdb.set_trace()
    #Manually remove dedecanesulfonic acid as it's a huge background signal
    df_orbitrap_peaks.drop(df_orbitrap_peaks[df_orbitrap_peaks["Formula"] == "C12 H26 O3 S"].index,inplace=True)

    #Filter out peaks with strange formula
    df_orbitrap_peaks = df_orbitrap_peaks[df_orbitrap_peaks["Formula"].apply(lambda x: filter_by_chemform(x))]
    #Merge compound peaks that have the same m/z and retention time
    #Round m/z to nearest integer and RT to nearest 2, as in 1/3/5/7/9 etc
    #Also remove anything with RT > 20min
        
    df_orbitrap_peaks.drop(df_orbitrap_peaks[df_orbitrap_peaks["RT [min]"] > 20].index, inplace=True)    
    #df.drop(df[df.score < 50].index, inplace=True)
    #Join the peaks with the same rounded m/z and RT    
    RT_round =  df_orbitrap_peaks["RT [min]"].apply(lambda x: round_odd(x))
    mz_round = df_orbitrap_peaks["m/z"].apply(lambda x: round(x, 2))
    
    
    #ORIGINAL
    #df_orbitrap_peaks = df_orbitrap_peaks.iloc[:,np.r_[0:4]].groupby([mz_round,RT_round]).aggregate("first").join(df_orbitrap_peaks.iloc[:,np.r_[4:len(df_orbitrap_peaks.columns)]].groupby([mz_round,RT_round]).aggregate("sum") )
    
    #MERGE SAME MOLECULE AND RT <10 OR >10
    RT_round10 =  df_orbitrap_peaks["RT [min]"].apply(lambda x: above_below_10(x))
    df_orbitrap_peaks = df_orbitrap_peaks.iloc[:,np.r_[0:4]].groupby([df_orbitrap_peaks["Formula"],RT_round10]).aggregate("first").join(df_orbitrap_peaks.iloc[:,np.r_[4:len(df_orbitrap_peaks.columns)]].groupby([df_orbitrap_peaks["Formula"],RT_round10]).aggregate("sum") )
    
    
    return df_orbitrap_peaks

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
    
    
#######################
####CHEMICAL ANALYSIS#######
#######################
# %%
#Function to extract the top n peaks from a cluster in terms of their chemical formula
def cluster_extract_peaks(cluster, df_peaks,num_peaks,chemform_namelist,printdf=False):
    #Check they are the same length
    if(cluster.shape[0] != df_peaks.shape[0]):
        print("cluster_extract_peaks returning null: cluster and peaks dataframe must have same number of peaks")
        return np.NaN
        quit()
    
    #First get top n peaks from the cluster
    #WAS WORKING
    # nlargest = cluster.nlargest(num_peaks)
    # nlargest_pct = nlargest / cluster.sum() * 100
    # output_df = pd.DataFrame(df_peaks["Formula"][nlargest.index])
    # output_df["peak_pct"] = nlargest_pct.round(1)
    #IVE JUST FIXED IT SO THE INDEX OF DF_FILTERS IS A MULTIINDEX
    #BUT NOW THIS IS KAKCING OUT, WANKWANKWANK
    #ITS ALMOST THERE I THINK
    
    nlargest = cluster.nlargest(num_peaks)
    nlargest_pct = nlargest / cluster.sum() * 100
    #pdb.set_trace()
    output_df = pd.DataFrame()#df_peaks["Formula"][nlargest.index])
    output_df["Formula"] = nlargest.index.get_level_values(0)
    output_df["peak_pct"] = nlargest_pct.round(1).values
    
    # #Get the chemical formula namelist
    # try:
    #     chemform_namelist
    # except NameError:
    #     #global chemform_namelist = pd.DataFrame()
    #     chemform_namelist = pd.read_excel(path + 'Beijing_Amb3.1_MZ_noblank.xlsx',engine='openpyxl')[["Formula","Name"]]
    #     chemform_namelist.set_index(chemform_namelist["Formula"],inplace=True)
    #pdb.set_trace()
    #output_df["Name"] = chemform_namelist[output_df.index]
    pdb.set_trace()
    output_df["Name"] = chemform_namelist.loc[output_df["Formula"]].values
    
    if(printdf == True):
        print(output_df)
        
    return output_df



#Load the mz file and generate a list of peak chemical names based off m/z
#This one does it just off the chemical formula
def load_chemform_namelist(peaks_filepath,peaks_sheetname="DEFAULT"):
    if(peaks_sheetname=="DEFAULT"):
        chemform_namelist = pd.read_excel(peaks_filepath,engine='openpyxl')[["Formula","Name"]]
    else:
        chemform_namelist = pd.read_excel(peaks_filepath,engine='openpyxl',sheetname=peaks_sheetname)[["Formula","Name"]]
    #chemform_namelist.fillna('-', inplace=True)
    chemform_namelist = chemform_namelist.groupby(chemform_namelist["Formula"]).agg(pd.Series.mode)
    
    #Do some filtering
    chemform_namelist['Name'] = chemform_namelist['Name'].astype('string')
    chemform_namelist['Name'] = np.where(chemform_namelist['Name'] == '[]',chemform_namelist.index,chemform_namelist['Name'])
    chemform_namelist = chemform_namelist.drop_duplicates()
    
    
    
    
    #chemform_namelist['Name'] = chemform_namelist['Name'].astype('str')
    #Remove blanks
    #chemform_namelist['Name'] = chemform_namelist['Name'].str.replace('-','',regex=False)
    return chemform_namelist


#################################################
#####AUTOENCODER GENERATORS###################
##############################################
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
    
    def fit_model(self, x_train,x_test='DEFAULT',batch_size=100,epochs=30,verbose='auto'):
        if(x_test=='DEFAULT'):
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
class VAE_n_layer():
    def __init__(self,hp='DEFAULT',input_dim=964,latent_dim=15,int_layers=2,int_layer_dims='DEFAULT',learning_rate=1e-3,latent_activation='linear'):
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
            self.latent_activation = latent_activation
            self.learning_rate = learning_rate
        
        else:   #Use kerastuner hyperparameters
            self.input_dim = hp.get('input_dim')
            self.latent_dim = hp.get('latent_dim')
            self.int_layers = hp.get('intermediate_layers')
            self.int_layer_dims = [val.value for key, val in hp._space.items() if 'intermediate_dim' in key]
            self.learning_rate = hp.get('learning_rate')
            self.decoder_output_activation = hp.get('decoder_output_activation')
            self.latent_activation = hp.get('decoder_output_activation')
        
        self.vae = None
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

        #z is the latent layer
        z_mean = layers.Dense(self.latent_dim,activation=self.latent_activation,name='latent_mean')(encoder_layer)
        z_log_var = layers.Dense(self.latent_dim,name='latent_log_var')(encoder_layer)
        self.encoder = Model(inputs=encoder_input_layer, outputs=z_mean, name="encoder_vae")
        
        # #Resample
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            #epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                      #stddev=epsilon_std)
            epsilon = K.random_normal(shape=(batch, dim), mean=0.,
                                      stddev=1)
            return z_mean + K.exp(z_log_var / 2) * epsilon
        
        latent_resampled = layers.Lambda(sampling, output_shape=(self.latent_dim,),name='resampled_latent_mean')([z_mean, z_log_var])
        
    
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
        
        outputs = self.decoder(latent_resampled)
        
        
        self.vae = Model(inputs=encoder_input_layer, outputs=outputs, name="vae")
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
    
    
        # def vae_loss(encoder_input_layer, outputs):
        #     mse_loss = self.input_dim * metrics.mse(encoder_input_layer, outputs)
        #     #xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        #     kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        #     return mse_loss + kl_loss
        # def mse_loss(encoder_input_layer,outputs):
        #     mse_loss = metrics.mse(encoder_input_layer, outputs)
        #     return mse_loss
        
        # def kl_loss(encoder_input_layer,outputs):
        #     kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        #     return kl_loss
        
        
        #self.vae.compile(optimizer=optimizer, loss=vae_loss,metrics=[mse_loss,kl_loss])
        #COMPILING
        self.vae.compile(optimizer, loss='mse')
    
    def fit_model(self, x_train,x_test='DEFAULT',batch_size=100,epochs=30,verbose='auto'):
        if(x_test=='DEFAULT'):
            _history = self.vae.fit(x_train,x_train,
                         shuffle=True,
                         epochs=epochs,
                         batch_size=batch_size,
                         validation_data=(x_train,x_train),verbose=verbose)
        else:
            _history = self.vae.fit(x_train,x_train,
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
# %%Random useful functions
def custom_round(x, base=5):
    return int(base * round(float(x)/base))

#round to nearest odd number
def round_odd(x):
    return (2*math.floor(x/2)+1)

#either above or below 10
def above_below_10(x):
    if(x > 20):
        return np.nan
    elif(x>10):
        return 15
    elif(x>0):
        return 5
    else:
        return np.nan
    
#Relabel cluster labels so the most frequent label is 0, second most is 1 etc
#labels must be an ndarray
def relabel_clusters_most_freq(labels):
    most_frequent_order = np.flip(np.argsort(np.bincount(labels))[-(np.unique(labels).size):])
    #return most_frequent_order
    labels_out = labels
    for lab in range(len(most_frequent_order)):
        labels_out = np.where(labels == most_frequent_order[lab],lab,labels_out)
    return labels_out

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
        print("using sig noise")
    
    newdf = newdf * signoise
       


    #Efficient version
    #newdf = pd.DataFrame(np.repeat(df.values,num_copies,axis=0)) * np.random.normal(1, sig_noise_pct/100, [num_copies*num_rows,num_cols])
    return newdf



#Make dataframe with top n% of data signal
#Extract the peaks from the real-space data
def extract_top_npercent(df,pct,plot=False):
    peaks_sum = df.sum()
    #set negative to zero
    peaks_sum = peaks_sum.clip(lower=0)
    peaks_sum_norm = peaks_sum/ peaks_sum.sum()
    peaks_sum_norm_sorted = peaks_sum_norm.sort_values(ascending=False)
    numpeaks_top70 = peaks_sum_norm_sorted.cumsum().searchsorted(0.7)
    peaks_sum_norm_sorted_cumsum = peaks_sum_norm_sorted.cumsum()

    if(plot==True):
        fig,ax = plt.subplots(1,figsize=(8,6))
        ax.plot(peaks_sum_norm_sorted_cumsum.values)
        ax.set_xlabel('Peak rank')
        ax.set_ylabel('Cumulative normalised sum')
        plt.show()
        
    #Now pick off the top 70% of peaks
    index_top70 = peaks_sum.nlargest(numpeaks_top70).index
    df_top70 = df[index_top70]
    return df_top70
