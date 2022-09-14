# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 17:00:03 2022

@author: mbcx5jt5
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.patches import Patch
import numpy as np
#import seaborn as sns
#import math
import time

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback, TerminateOnNaN


from sklearn.preprocessing import StandardScaler, FunctionTransformer, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score,silhouette_score
import sklearn.metrics as skmetrics

from math import pi


import os
os.chdir('C:/Work/Python/Github/Orbitrap_clustering')
from ae_functions import *
from orbitrap_functions import *





#%%Load data from HDF
filepath = r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\PMF data\ORBITRAP_Data_Pre_PMF.h5"
df_all_data, df_all_err, ds_all_mz = Load_pre_PMF_data(filepath)

#Load all time data, ie start/mid/end
df_all_times = pd.read_csv(r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\PMF data\Times_all.csv")
df_all_times['date_start'] = pd.to_datetime(df_all_times['date_start'],dayfirst=True)
df_all_times['date_mid'] = pd.to_datetime(df_all_times['date_mid'],dayfirst=True)
df_all_times['date_end'] = pd.to_datetime(df_all_times['date_end'],dayfirst=True)

df_all_times.set_index(df_all_times['date_mid'],inplace=True)
fuzzy_index = pd.merge_asof(pd.DataFrame(index=df_all_data.index),df_all_times,left_index=True,right_index=True,direction='nearest',tolerance=pd.Timedelta(hours=1.25))
df_all_times = df_all_times.loc[fuzzy_index['date_mid']]

dataset_cat = delhi_beijing_datetime_cat(df_all_data)
df_dataset_cat = pd.DataFrame(delhi_beijing_datetime_cat(df_all_data),columns=['dataset_cat'],index=df_all_data.index)
ds_dataset_cat = df_dataset_cat['dataset_cat']

time_cat = delhi_calc_time_cat(df_all_times)
df_time_cat = pd.DataFrame(delhi_calc_time_cat(df_all_times),columns=['time_cat'],index=df_all_times.index)
ds_time_cat = df_time_cat['time_cat']

#This is a list of peaks with Sari's description from her PMF
Sari_peaks_list = pd.read_csv(r'C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\Sari_Peaks_Sources.csv',index_col='Formula',na_filter=False)
Sari_peaks_list = Sari_peaks_list[~Sari_peaks_list.index.duplicated(keep='first')]


#%%Work out O:C, H:C, S:C, N:C ratios for all peaks
#df_element_ratios = df_all_data.columns.get_level_values(0).to_series().apply(lambda x: chemform_ratios(x)[0])
df_element_ratios = pd.DataFrame()
df_element_ratios['H/C'] = df_all_data.columns.get_level_values(0).to_series().apply(lambda x: chemform_ratios(x)[0])
df_element_ratios['O/C'] = df_all_data.columns.get_level_values(0).to_series().apply(lambda x: chemform_ratios(x)[1])
df_element_ratios['N/C'] = df_all_data.columns.get_level_values(0).to_series().apply(lambda x: chemform_ratios(x)[2])
df_element_ratios['S/C'] = df_all_data.columns.get_level_values(0).to_series().apply(lambda x: chemform_ratios(x)[3])


#%%Test data scaling


df_all_data_aug = augment_data_noise(df_all_data,50,1,0)

#Standard scaling
standardscaler_all = StandardScaler().fit(df_all_data_aug.to_numpy())
#df_all_data_standard = pd.DataFrame(standardscaler_all.fit_transform(df_all_data.to_numpy()),index=df_all_data.index,columns=df_all_data.columns)
df_all_data_aug_standard = pd.DataFrame(standardscaler_all.transform(df_all_data_aug.to_numpy()),columns=df_all_data.columns)
df_all_data_standard = pd.DataFrame(standardscaler_all.transform(df_all_data.to_numpy()),index=df_all_data.index,columns=df_all_data.columns)

ae_input = df_all_data_aug_standard.values
ae_input_val = df_all_data_standard.values
input_dim = ae_input.shape[1]



#standardscaler_all2 = StandardScaler(with_std=False)
#df_all_data_standard2 = pd.DataFrame(standardscaler_all2.fit_transform(df_all_data.to_numpy()),columns=df_all_data.columns)

# max_col = df_all_data.mean().idxmax()
# min_col = df_all_data.mean().idxmin()

# df_all_data[max_col].plot(kind='hist',title='No scaling, max column')
# df_all_data[min_col].plot(kind='hist',title='No scaling, min column')
# df_all_data_standard[max_col].plot(kind='hist',title='Standard scaling, max column')
# df_all_data_standard[min_col].plot(kind='hist',title='Standard scaling, min column')
# df_all_data_standard2[max_col].plot(kind='hist',title='Standard scaling (not normalised), max column')
# df_all_data_standard2[min_col].plot(kind='hist',title='Standard scaling (not normalised), min column')

#%%Normalise so the mean of the whole matrix is 1
orig_mean = df_all_data.mean().mean()
pipe_norm1_mtx = FunctionTransformer(lambda x: np.divide(x,orig_mean),inverse_func = lambda x: np.multiply(x,orig_mean))
pipe_norm1_mtx.fit(df_all_data.to_numpy())
df_all_data_norm1 = pd.DataFrame(pipe_norm1_mtx.transform(df_all_data),columns=df_all_data.columns)






#%%Clustering

#Do the clustering methodology thing from before
#%%Implementation of workflow for real space data
df_cluster_labels_mtx = cluster_n_times(df_all_data_standard,10,min_num_clusters=2)
df_cluster_counts_mtx = count_cluster_labels_from_mtx(df_cluster_labels_mtx)

df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx = calc_cluster_elemental_ratios(df_cluster_labels_mtx,df_all_data,df_element_ratios)
plot_cluster_elemental_ratios(df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,'Real-space data elemental ratios')

cluster_profiles_mtx, cluster_profiles_mtx_norm, num_clusters_index, cluster_index = average_cluster_profiles(df_cluster_labels_mtx,df_all_data)
df_cluster_corr_mtx, df_prevcluster_corr_mtx = correlate_cluster_profiles(cluster_profiles_mtx_norm, num_clusters_index, cluster_index)
plot_cluster_profile_corrs(df_cluster_corr_mtx, df_prevcluster_corr_mtx)

plot_all_cluster_tseries_BeijingDelhi(df_cluster_labels_mtx,ds_dataset_cat,title_prefix='Real space scaled, ')


plot_all_cluster_profiles(df_all_data,cluster_profiles_mtx_norm,num_clusters_index,ds_all_mz,df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,
                          df_cluster_corr_mtx,df_prevcluster_corr_mtx,df_cluster_counts_mtx,Sari_peaks_list,title_prefix='Real-space scaled, ')
                        
df_clust_cat_counts, df_cat_clust_counts, df_clust_time_cat_counts,df_time_cat_clust_counts = count_clusters_project_time(
    df_cluster_labels_mtx,ds_dataset_cat,ds_time_cat,title_prefix='Real space scaled, ',title_suffix='')





#%%

        


# %%#How many clusters should we have? Real space
min_clusters = 2
max_clusters = 10

num_clusters_index = range(min_clusters,(max_clusters+1),1)
ch_score = np.empty(len(num_clusters_index))
db_score = np.empty(len(num_clusters_index))
silhouette_scores = np.empty(len(num_clusters_index))

for num_clusters in num_clusters_index:
    agglom_native = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    #agglom_native = KMeans(n_clusters = num_clusters)
    clustering = agglom_native.fit(df_all_data_standard)
    ch_score[num_clusters-min_clusters] = calinski_harabasz_score(df_all_data_standard, clustering.labels_)
    db_score[num_clusters-min_clusters] = davies_bouldin_score(df_all_data_standard, clustering.labels_)
    silhouette_scores[num_clusters-min_clusters] = silhouette_score(df_all_data_standard, clustering.labels_)
fig,ax1 = plt.subplots(figsize=(10,6))
ax2=ax1.twinx()
ax1.plot(num_clusters_index,ch_score,label="CH score")
ax2.plot(num_clusters_index,db_score,c='red',label="DB score")
ax2.plot(num_clusters_index,silhouette_scores,c='black',label="Silhouette score")
ax1.set_xlabel("Num clusters")
ax1.set_ylabel("CH score")
ax2.set_ylabel("DB score")
plt.title("How many clusters? Standardised real-space data")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)
#ax1.xaxis.set_major_locator(plticker.MultipleLocator(base=5.0))
#ax1.xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()




















#%%Now compare loss for different latent dimensions
#This is NOT using kerastuner, and is using log-spaced intermediate layers
#WARNING THIS TAKES ABOUT HALF AN HOUR


latent_dims = []
AE1_MSE_best50epoch =[]
AE2_MSE_best50epoch =[]
AE3_MSE_best50epoch =[]
AE4_MSE_best50epoch =[]
best_hps_array = []


verbose = 0

start_time = time.time()

for latent_dim in range(1,30):
    print(latent_dim)
    K.clear_session()
    latent_dims.append(latent_dim)
    
    #Test for 1 intermediate layer
    ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=1)
    history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=50,verbose=verbose)
    val_acc_per_epoch = history.history['val_loss']
    AE1_MSE_best50epoch.append(min(history.history['val_loss']))
    
    #Test for 2 intermediate layers
    ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=2)
    history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=50,verbose=verbose)
    val_acc_per_epoch = history.history['val_loss']
    AE2_MSE_best50epoch.append(min(history.history['val_loss']))
    
    #Test for 3 intermediate layers
    ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=3)
    history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=50,verbose=verbose)
    val_acc_per_epoch = history.history['val_loss']
    AE3_MSE_best50epoch.append(min(history.history['val_loss']))
    
    #Test for 4 intermediate layers
    ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=4)
    history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=50,verbose=verbose)
    val_acc_per_epoch = history.history['val_loss']
    AE4_MSE_best50epoch.append(min(history.history['val_loss']))
    
print("--- %s seconds ---" % (time.time() - start_time))
    
#%%Plot the data for the different number of layers above
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


loc = plticker.MultipleLocator(base=10.0) # this locator puts ticks at regular intervals
ax[0].xaxis.set_major_locator(plticker.MultipleLocator(base=5.0))
ax[0].xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
ax[1].xaxis.set_major_locator(plticker.MultipleLocator(base=5.0))
ax[1].xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))

plt.show()



#%%STANDARD AUTOENCODER
#Based on the above, use an AE with 2 intermediate layers and latent dim of 20


latent_activation = 'linear'
if(type(latent_activation)==type('str')):
    latent_activation_name = latent_activation
else:
    latent_activation_name = latent_activation.__name__

ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=3,int_layers=3,latent_activation=latent_activation)
history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=100)
val_acc_per_epoch = history.history['val_loss']

latent_space = ae_obj.encoder(ae_input_val).numpy()
df_latent_space = pd.DataFrame(latent_space,index=df_all_data.index)
plt.scatter(latent_space[:,0],latent_space[:,1])


fig,ax = plt.subplots(1)
ax.set_xlabel('AE input')
ax.set_xlabel('AE output')
plt.scatter(ae_input_val,ae_obj.ae(ae_input_val))


#%%Plot latent space
cmap_EOS11 = Make_EOS11_cmap()

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
#scatter = ax.scatter(log_latent_space[:,0], log_latent_space[:,1],log_latent_space[:,2], c=ds_dataset_cat.cat.codes,cmap=cmap_EOS11)
scatter = ax.scatter(latent_space[:,0], latent_space[:,1],latent_space[:,2], c=ds_dataset_cat.cat.codes,cmap=cmap_EOS11)
ax.set_xlabel('Latent space dim 0')
ax.set_ylabel('Latent space dim 1')
ax.set_zlabel('Latent space dim 2')
#ax.set_box_aspect([np.ptp(i) for i in latent_space])  # equal aspect ratio

labels = ds_dataset_cat.drop_duplicates()#['type1', 'type2', 'type3', 'type4']
handles, _ = scatter.legend_elements(prop="colors", alpha=0.6) # use my own labels
legend1 = ax.legend(handles, labels, loc="upper right")
ax.add_artist(legend1)
plt.title('3 dimensional latent space, ' + latent_activation_name + ' latent activation')
plt.show()

#%%Scale latent space

from sklearn.preprocessing import MaxAbsScaler

standardscaler_latent = StandardScaler()
robustscaler_latent = RobustScaler()
maxabsscaler_latent = MaxAbsScaler()
df_latent_standard = pd.DataFrame(standardscaler_latent.fit_transform(latent_space),index=df_all_data.index)
df_latent_robust = pd.DataFrame(robustscaler_latent.fit_transform(latent_space),index=df_all_data.index)
df_latent_maxabs = pd.DataFrame(maxabsscaler_latent.fit_transform(latent_space),index=df_all_data.index)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
#scatter = ax.scatter(log_latent_space[:,0], log_latent_space[:,1],log_latent_space[:,2], c=ds_dataset_cat.cat.codes,cmap=cmap_EOS11)
scatter = ax.scatter(df_latent_standard.iloc[:,0], df_latent_standard.iloc[:,1],df_latent_standard.iloc[:,2], c=ds_dataset_cat.cat.codes,cmap=cmap_EOS11)
ax.set_xlabel('Latent space dim 0 (scaled)')
ax.set_ylabel('Latent space dim 1 (scaled)')
ax.set_zlabel('Latent space dim 2 (scaled)')
#ax.set_box_aspect([np.ptp(i) for i in latent_space])  # equal aspect ratio

labels = ds_dataset_cat.drop_duplicates()#['type1', 'type2', 'type3', 'type4']
handles, _ = scatter.legend_elements(prop="colors", alpha=0.6) # use my own labels
legend1 = ax.legend(handles, labels, loc="upper right")
ax.add_artist(legend1)
plt.title('3 dimensional latent space, StandardScaler')
plt.show()



#%%Run clustering on scaled latent space

#%%Implementation of workflow for latent space data
df_cluster_labels_mtx = cluster_n_times(df_latent_space,10,min_num_clusters=2)
df_cluster_counts_mtx = count_cluster_labels_from_mtx(df_cluster_labels_mtx)

df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx = calc_cluster_elemental_ratios(df_cluster_labels_mtx,df_all_data,df_element_ratios)
plot_cluster_elemental_ratios(df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,'Latent space data elemental ratios')

cluster_profiles_mtx, cluster_profiles_mtx_norm, num_clusters_index, cluster_index = average_cluster_profiles(df_cluster_labels_mtx,df_all_data)
df_cluster_corr_mtx, df_prevcluster_corr_mtx = correlate_cluster_profiles(cluster_profiles_mtx_norm, num_clusters_index, cluster_index)
plot_cluster_profile_corrs(df_cluster_corr_mtx, df_prevcluster_corr_mtx)

plot_all_cluster_tseries_BeijingDelhi(df_cluster_labels_mtx,ds_dataset_cat,title_prefix='Latent space scaled, ')


plot_all_cluster_profiles(df_all_data,cluster_profiles_mtx_norm,num_clusters_index,ds_all_mz,df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,
                          df_cluster_corr_mtx,df_prevcluster_corr_mtx,df_cluster_counts_mtx,Sari_peaks_list,title_prefix='Latent space scaled, ')
                        
df_clust_cat_counts, df_cat_clust_counts, df_clust_time_cat_counts,df_time_cat_clust_counts = count_clusters_project_time(
    df_cluster_labels_mtx,ds_dataset_cat,ds_time_cat,title_prefix='Latent space scaled, ',title_suffix='')




# %%#How many clusters should we have? Scaled Latent space
min_clusters = 2
max_clusters = 10

num_clusters_index = range(min_clusters,(max_clusters+1),1)
ch_score = np.empty(len(num_clusters_index))
db_score = np.empty(len(num_clusters_index))
silhouette_scores = np.empty(len(num_clusters_index))

for num_clusters in num_clusters_index:
    agglom_native = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    #agglom_native = KMeans(n_clusters = num_clusters)
    clustering = agglom_native.fit(df_latent_space)
    ch_score[num_clusters-min_clusters] = calinski_harabasz_score(df_latent_space, clustering.labels_)
    db_score[num_clusters-min_clusters] = davies_bouldin_score(df_latent_space, clustering.labels_)
    silhouette_scores[num_clusters-min_clusters] = silhouette_score(df_latent_space, clustering.labels_)
fig,ax1 = plt.subplots(figsize=(10,6))
ax2=ax1.twinx()
ax1.plot(num_clusters_index,ch_score,label="CH score")
ax2.plot(num_clusters_index,db_score,c='red',label="DB score")
ax2.plot(num_clusters_index,silhouette_scores,c='black',label="Silhouette score")
ax1.set_xlabel("Num clusters")
ax1.set_ylabel("CH score")
ax2.set_ylabel("DB score")
plt.title("How many clusters? Standardised latent-space data")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)
#ax1.xaxis.set_major_locator(plticker.MultipleLocator(base=5.0))
#ax1.xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()

#%%Plot latent space
cmap_EOS11 = Make_EOS11_cmap()

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
#scatter = ax.scatter(log_latent_space[:,0], log_latent_space[:,1],log_latent_space[:,2], c=ds_dataset_cat.cat.codes,cmap=cmap_EOS11)
scatter = ax.scatter(df_latent_standard.iloc[:,0], df_latent_standard.iloc[:,1],df_latent_standard.iloc[:,2], c=df_cluster_labels_mtx[6])
ax.set_xlabel('Latent space dim 0')
ax.set_ylabel('Latent space dim 1')
ax.set_zlabel('Latent space dim 2')
#ax.set_box_aspect([np.ptp(i) for i in latent_space])  # equal aspect ratio

labels = ds_dataset_cat.drop_duplicates()#['type1', 'type2', 'type3', 'type4']
handles, _ = scatter.legend_elements(prop="colors", alpha=0.6) # use my own labels
legend1 = ax.legend(handles, labels, loc="upper right")
ax.add_artist(legend1)
plt.title('3 dimensional latent space, ' + latent_activation_name + ' latent activation')
plt.show()



#%%Build and trainVAE
beta_schedule = np.array([1e-5,1e-4,1e-3,1e-2,1e-1,0.2,0.3,0.4,0.5,100])

# beta_schedule = np.concatenate([np.zeros(5),np.arange(0.1,1.1,0.1),np.ones(10,),np.arange(0.9,0.45,-0.05)])

# beta_schedule = np.abs(np.sin(np.arange(0,201)*0.45/pi))

#beta_schedule = 1 - np.cos(np.arange(0,50)*0.45/pi)

vae_obj = VAE_n_layer(input_dim=input_dim,latent_dim=5,int_layers=3,beta_schedule=beta_schedule)


#Callback for VAE to make beta change with training epoch
class vae_beta_scheduler(keras.callbacks.Callback):
    """Callback for VAE to make beta change with training epoch

  Arguments:
      schedule: this is a dummy and doesn't actually do anything.
      You need to tailor this to your vae_n_layer object, for example here it's called vae_obj'
  """

    def __init__(self,schedule):
        super(vae_beta_scheduler, self).__init__()
        self.schedule=schedule

    def on_epoch_begin(self,  epoch, logs=None):
        
        pdb.set_trace()
        
        beta_schedule=self.schedule
        if(epoch >= beta_schedule.shape[0]):
            new_beta = beta_schedule[-1]
        else:
            new_beta = beta_schedule[epoch]
        self.model.metrics[1] = (new_beta)
        #old_beta = self.model.metrics[1]
        #testfunc(self,new_beta)
        #self.model.add_metric(old_beta, name='beta3', aggregation='mean')
        #tf.keras.backend.set_value(self.model.metrics[1], new_beta)         
        print("\nEpoch %05d: beta is %6.4f." % (epoch, new_beta))
        
    # def testfunc(self,epoch,beta):
    #     self.model.add_metric(beta, name='beta3', aggregation='mean')

class MyCallback(keras.callbacks.Callback):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    # customize your behavior
    def on_epoch_begin(self, epoch, logs={}):
        if(epoch>=5):
            pdb.set_trace()
            self.alpha = self.alpha - 1
            self.beta = self.beta + 10
            self.model.compile(loss_weights=[1,self.alpha,self.beta])
            

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
#         pdb.set_trace()
#         beta_schedule=vae_obj.beta_schedule
#         if(epoch >= beta_schedule.shape[0]):
#             new_beta = beta_schedule[-1]
#         else:
#             new_beta = beta_schedule[epoch]
#         tf.keras.backend.set_value(vae_obj.beta, new_beta)         
#         print("\nEpoch %05d: beta is %6.4f." % (epoch, new_beta))


df_aug = augment_data_noise(df_all_data_norm1,50,1,0)
ae_input = df_aug.values
ae_input_val = df_all_data_norm1.values
input_dim = ae_input.shape[1]



history_vae = vae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=20,verbose=1,callbacks=[
    vae_beta_scheduler(vae_obj.beta_schedule),TerminateOnNaN()])


#history_vae = vae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=50,verbose=1)
latent_space = vae_obj.encoder(ae_input_val).numpy()
df_latent_space = pd.DataFrame(latent_space,index=df_all_data.index)
plt.scatter(latent_space[:,0],latent_space[:,1])


fig,ax = plt.subplots(1)
plt.scatter(ae_input_val,vae_obj.ae(ae_input_val))
plt.title('AE input vs output, ' + latent_activation_name + ' latent activation')
plt.xlabel('AE input')
plt.ylabel('AE output')
plt.show()

#%%
fig,ax=plt.subplots(2,1,figsize=(12,8))
ax[0].set_title('Finding optimum epochs- simple relu VAE')
ax[0].plot(history_vae.epoch,history_vae.history['val_loss'],c='k')
ax[0].plot(history_vae.epoch,history_vae.history['val_msemetric'],c='r')
ax[0].plot(history_vae.epoch,history_vae.history['val_kl'],c='b')
ax[0].set_xlabel('Number of epochs')
ax[0].set_ylabel('Loss')
ax[1].plot(history_vae.epoch,history_vae.history['val_loss'],label='Total loss',c='k')
ax[1].plot(history_vae.epoch,history_vae.history['val_msemetric'],label='MSE loss',c='r')
ax[1].plot(history_vae.epoch,history_vae.history['val_kl'], label='KL loss',c='b')
ax[1].set_xlabel('Number of epochs')
ax[1].set_ylabel('Loss')
ax[1].set_yscale('log')
loc = plticker.MultipleLocator(base=10.0) # this locator puts ticks at regular intervals
ax[0].xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
ax[0].xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
ax[1].xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
ax[1].xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
ax[1].legend()
plt.show()


#%%Plot latent space
cmap_EOS11 = Make_EOS11_cmap()

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
#scatter = ax.scatter(log_latent_space[:,0], log_latent_space[:,1],log_latent_space[:,2], c=ds_dataset_cat.cat.codes,cmap=cmap_EOS11)
scatter = ax.scatter(latent_space[:,0], latent_space[:,1],latent_space[:,2], c=ds_dataset_cat.cat.codes,cmap=cmap_EOS11)
ax.set_xlabel('Latent space dim 0')
ax.set_ylabel('Latent space dim 1')
ax.set_zlabel('Latent space dim 2')
#ax.set_box_aspect([np.ptp(i) for i in latent_space])  # equal aspect ratio

labels = ds_dataset_cat.drop_duplicates()#['type1', 'type2', 'type3', 'type4']
handles, _ = scatter.legend_elements(prop="colors", alpha=0.6) # use my own labels
legend1 = ax.legend(handles, labels, loc="upper right")
ax.add_artist(legend1)
plt.title('3 dimensional latent space, ' + latent_activation_name + ' latent activation')
plt.show()




# #%%Histogram of latent features
# fig,ax = plt.subplots(3,figsize=(6,8))
# fig.suptitle('3 dimensional latent space, ' + latent_activation_name + ' latent activation')
# df_latent_space.iloc[:,0].plot(ax=ax[0],kind='hist',title='Latent dimension 0')
# df_latent_space.iloc[:,1].plot(ax=ax[1],kind='hist',title='Latent dimension 1')
# df_latent_space.iloc[:,2].plot(ax=ax[2],kind='hist',title='Latent dimension 2')



#%%Now compare loss for different latent dimensions, VAE version
#This is NOT using kerastuner, and is using log-spaced intermediate layers
#WARNING THIS TAKES ABOUT HALF AN HOUR


# latent_dims = []

# VAE2_MSE_best50epoch =[]
# VAE3_MSE_best50epoch =[]
# VAE4_MSE_best50epoch =[]
# #best_hps_array = []


verbose = 1

start_time = time.time()

latent_dims = np.arange(1,2)
VAE1_MSE_best_loss = np.ones(len(latent_dims))*-1

for latent_dim in latent_dims:
    print(latent_dim)
    K.clear_session()
    
    #Test for 1 intermediate layer
    vae_obj2 = VAE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=1,beta_schedule=beta_schedule)
    
    #Callback for VAE to make beta change with training epoch
    class vae_beta_scheduler2(keras.callbacks.Callback):
        """Callback for VAE to make beta change with training epoch
    
      Arguments:
          schedule: this is a dummy and doesn't actually do anything.
          You need to tailor this to your vae_n_layer object, for example here it's called vae_obj'
      """
    
        def __init__(self,schedule):
            super(vae_beta_scheduler2, self).__init__()
            self.schedule=schedule
    
        def on_epoch_begin(self,  epoch, logs=None):
            #pdb.set_trace()
            beta_schedule=vae_obj2.beta_schedule
            if(epoch >= beta_schedule.shape[0]):
                new_beta = beta_schedule[-1]
            else:
                new_beta = beta_schedule[epoch]
            tf.keras.backend.set_value(vae_obj2.beta, new_beta)         
            print("\nEpoch %05d: beta is %6.4f." % (epoch, new_beta))
    
    
    
    
    
    history_vae = vae_obj2.fit_model(ae_input, x_test=ae_input_val,epochs=15,verbose=1,callbacks=[
       vae_beta_scheduler2(vae_obj2.beta_schedule),TerminateOnNaN()])

    
    
    
    #val_acc_per_epoch = history.history['val_loss']
    if(min(history_vae.history['val_loss']) == np.nan):
        latent_dim -= 1
    else:    
        VAE1_MSE_best_loss[latent_dim-1] = min(history_vae.history['val_loss'])
    
    
    # #Test for 2 intermediate layers
    # ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=2)
    # history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=50,verbose=verbose)
    # val_acc_per_epoch = history.history['val_loss']
    # AE2_MSE_best50epoch.append(min(history.history['val_loss']))
    
    # #Test for 3 intermediate layers
    # ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=3)
    # history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=50,verbose=verbose)
    # val_acc_per_epoch = history.history['val_loss']
    # AE3_MSE_best50epoch.append(min(history.history['val_loss']))
    
    # #Test for 4 intermediate layers
    # ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=4)
    # history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=50,verbose=verbose)
    # val_acc_per_epoch = history.history['val_loss']
    # AE4_MSE_best50epoch.append(min(history.history['val_loss']))
    
print("--- %s seconds ---" % (time.time() - start_time))









#%%Run clustering on VAE latent space

#%%Implementation of workflow for latent space data
df_cluster_labels_mtx = cluster_n_times(df_latent_space,10,min_num_clusters=2)
df_cluster_counts_mtx = count_cluster_labels_from_mtx(df_cluster_labels_mtx)

df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx = calc_cluster_elemental_ratios(df_cluster_labels_mtx,df_all_data,df_element_ratios)
plot_cluster_elemental_ratios(df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,'VAE latent space data elemental ratios')

cluster_profiles_mtx, cluster_profiles_mtx_norm, num_clusters_index, cluster_index = average_cluster_profiles(df_cluster_labels_mtx,df_all_data)
df_cluster_corr_mtx, df_prevcluster_corr_mtx = correlate_cluster_profiles(cluster_profiles_mtx_norm, num_clusters_index, cluster_index)
plot_cluster_profile_corrs(df_cluster_corr_mtx, df_prevcluster_corr_mtx)

plot_all_cluster_tseries_BeijingDelhi(df_cluster_labels_mtx,ds_dataset_cat,title_prefix='VAE latent space, ')


plot_all_cluster_profiles(df_all_data,cluster_profiles_mtx_norm,num_clusters_index,ds_all_mz,df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,
                          df_cluster_corr_mtx,df_prevcluster_corr_mtx,df_cluster_counts_mtx,Sari_peaks_list,title_prefix='VAE latent space, ')
                        
df_clust_cat_counts, df_cat_clust_counts, df_clust_time_cat_counts,df_time_cat_clust_counts = count_clusters_project_time(
    df_cluster_labels_mtx,ds_dataset_cat,ds_time_cat,title_prefix='VAE latent space scaled, ',title_suffix='')








# %%#How many clusters should we have? VAE Latent space
min_clusters = 2
max_clusters = 10

num_clusters_index = range(min_clusters,(max_clusters+1),1)
ch_score = np.empty(len(num_clusters_index))
db_score = np.empty(len(num_clusters_index))
silhouette_scores = np.empty(len(num_clusters_index))

for num_clusters in num_clusters_index:
    agglom_native = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    #agglom_native = KMeans(n_clusters = num_clusters)
    clustering = agglom_native.fit(df_latent_space)
    ch_score[num_clusters-min_clusters] = calinski_harabasz_score(df_latent_space, clustering.labels_)
    db_score[num_clusters-min_clusters] = davies_bouldin_score(df_latent_space, clustering.labels_)
    silhouette_scores[num_clusters-min_clusters] = silhouette_score(df_latent_space, clustering.labels_)
fig,ax1 = plt.subplots(figsize=(10,6))
ax2=ax1.twinx()
ax1.plot(num_clusters_index,ch_score,label="CH score")
ax2.plot(num_clusters_index,db_score,c='red',label="DB score")
ax2.plot(num_clusters_index,silhouette_scores,c='black',label="Silhouette score")
ax1.set_xlabel("Num clusters")
ax1.set_ylabel("CH score")
ax2.set_ylabel("DB score")
plt.title("How many clusters? VAE latent-space data")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)
#ax1.xaxis.set_major_locator(plticker.MultipleLocator(base=5.0))
#ax1.xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()










#%%AE top loss
ds_AE_loss_per_sample = pd.Series(AE_calc_loss_per_sample(ae_obj.ae,ae_input_val,ae_input_val), index=df_all_data.index)
plt.plot(ds_AE_loss_per_sample)
plt.title('AE loss per sample')
plt.show()

plot_orbitrap_top_ae_loss(df_all_data,mz_columns,ds_AE_loss_per_sample,num_top_losses=5,Sari_peaks_list=Sari_peaks_list)




