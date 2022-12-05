# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 17:00:03 2022

@author: mbcx5jt5
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.colors import ListedColormap
import numpy as np
import time

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback, TerminateOnNaN
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, FunctionTransformer, RobustScaler
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score,silhouette_score,adjusted_rand_score
from sklearn.manifold import TSNE

import math


import os
os.chdir('C:/Work/Python/Github/Orbitrap_clustering')
from ae_functions import *
from orbitrap_functions import *

from plotting.cmap_EOS11 import cmap_EOS11

from file_loaders.load_pre_PMF_data import load_pre_PMF_data



#%%Load data from HDF
filepath = r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\PMF data\ORBITRAP_Data_Pre_PMF.h5"
df_all_data, df_all_err, ds_all_mz = load_pre_PMF_data(filepath)

#Save data to CSV
df_all_data.to_csv(r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\PMF data\df_all_data.csv")
df_all_err.to_csv(r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\PMF data\df_all_err.csv")
ds_all_mz.to_csv(r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\PMF data\ds_all_mz.csv",index=False,header=False)

pd.DataFrame(df_all_data.columns.get_level_values(0),df_all_data.columns.get_level_values(1)).to_csv(r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\PMF data\RT_formula.csv",header=False)

#Load all time data, ie start/mid/end
df_all_times = pd.read_csv(r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\PMF data\Times_all.csv")
df_all_times['date_start'] = pd.to_datetime(df_all_times['date_start'],dayfirst=True)
df_all_times['date_mid'] = pd.to_datetime(df_all_times['date_mid'],dayfirst=True)
df_all_times['date_end'] = pd.to_datetime(df_all_times['date_end'],dayfirst=True)

df_all_times.set_index(df_all_times['date_mid'],inplace=True)
fuzzy_index = pd.merge_asof(pd.DataFrame(index=df_all_data.index),df_all_times,left_index=True,right_index=True,direction='nearest',tolerance=pd.Timedelta(hours=1.25))
df_all_times = df_all_times.loc[fuzzy_index['date_mid']]

ds_dataset_cat = delhi_beijing_datetime_cat(df_all_data.index)

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


#%%Data scaling


df_all_data_aug = augment_data_noise(df_all_data,50,1,0)

#Standard scaling
standardscaler_all = StandardScaler().fit(df_all_data_aug.to_numpy())
#df_all_data_standard = pd.DataFrame(standardscaler_all.fit_transform(df_all_data.to_numpy()),index=df_all_data.index,columns=df_all_data.columns)
df_all_data_aug_standard = pd.DataFrame(standardscaler_all.transform(df_all_data_aug.to_numpy()),columns=df_all_data.columns)
df_all_data_standard = pd.DataFrame(standardscaler_all.transform(df_all_data.to_numpy()),index=df_all_data.index,columns=df_all_data.columns)

ae_input = df_all_data_aug_standard.values
ae_input_val = df_all_data_standard.values
input_dim = ae_input.shape[1]


#%%Normalise so the mean of the whole matrix is 1
orig_mean = df_all_data.mean().mean()
pipe_norm1_mtx = FunctionTransformer(lambda x: np.divide(x,orig_mean),inverse_func = lambda x: np.multiply(x,orig_mean))
pipe_norm1_mtx.fit(df_all_data.to_numpy())
df_all_data_norm1 = pd.DataFrame(pipe_norm1_mtx.transform(df_all_data),columns=df_all_data.columns)


#%%Implementation of workflow for real space data, K-Medoids
df_cluster_labels_mtx = cluster_n_times(df_all_data_standard,10,min_num_clusters=2,cluster_type='kmedoids')
df_cluster_counts_mtx = count_cluster_labels_from_mtx(df_cluster_labels_mtx)

df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx = calc_cluster_elemental_ratios(df_cluster_labels_mtx,df_all_data,df_element_ratios)
plot_cluster_elemental_ratios(df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,'Real-space k-medoids elemental ratios')

cluster_profiles_mtx, cluster_profiles_mtx_norm, num_clusters_index, cluster_index = average_cluster_profiles(df_cluster_labels_mtx,df_all_data)
df_cluster_corr_mtx, df_prevcluster_corr_mtx = correlate_cluster_profiles(cluster_profiles_mtx_norm, num_clusters_index, cluster_index)
plot_cluster_profile_corrs(df_cluster_corr_mtx, df_prevcluster_corr_mtx,suptitle='Real space k-medoids')

plot_all_cluster_tseries_BeijingDelhi(df_cluster_labels_mtx,ds_dataset_cat,title_prefix='Real space k-medoids, ')

plot_all_cluster_profiles(df_all_data,cluster_profiles_mtx_norm,num_clusters_index,ds_all_mz,df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,
                          df_cluster_corr_mtx,df_prevcluster_corr_mtx,df_cluster_counts_mtx,Sari_peaks_list,title_prefix='Real-space k-medoids, ')
                        
df_clust_cat_counts, df_cat_clust_counts, df_clust_time_cat_counts,df_time_cat_clust_counts = count_clusters_project_time(
    df_cluster_labels_mtx,ds_dataset_cat,ds_time_cat,title_prefix='Real space k-medoids, ',title_suffix='')
    
compare_cluster_metrics(df_all_data_standard,2,10,'kmedoids','Real space ',' metrics')



#%%Implementation of workflow for real space data, k-means
df_cluster_labels_mtx = cluster_n_times(df_all_data_standard,10,min_num_clusters=2,cluster_type='kmeans')
df_cluster_counts_mtx = count_cluster_labels_from_mtx(df_cluster_labels_mtx)

df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx = calc_cluster_elemental_ratios(df_cluster_labels_mtx,df_all_data,df_element_ratios)
plot_cluster_elemental_ratios(df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,'Real-space k-means elemental ratios')

cluster_profiles_mtx, cluster_profiles_mtx_norm, num_clusters_index, cluster_index = average_cluster_profiles(df_cluster_labels_mtx,df_all_data)
df_cluster_corr_mtx, df_prevcluster_corr_mtx = correlate_cluster_profiles(cluster_profiles_mtx_norm, num_clusters_index, cluster_index)
plot_cluster_profile_corrs(df_cluster_corr_mtx, df_prevcluster_corr_mtx)

plot_all_cluster_tseries_BeijingDelhi(df_cluster_labels_mtx,ds_dataset_cat,title_prefix='Real space k-means, ')


plot_all_cluster_profiles(df_all_data,cluster_profiles_mtx_norm,num_clusters_index,ds_all_mz,df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,
                          df_cluster_corr_mtx,df_prevcluster_corr_mtx,df_cluster_counts_mtx,Sari_peaks_list,title_prefix='Real-space k-means, ')
                        
df_clust_cat_counts, df_cat_clust_counts, df_clust_time_cat_counts,df_time_cat_clust_counts = count_clusters_project_time(
    df_cluster_labels_mtx,ds_dataset_cat,ds_time_cat,title_prefix='Real space k-means, ',title_suffix='')

compare_cluster_metrics(df_all_data_standard,2,10,'kmeans','Real space ',' metrics')



#%%Implementation of workflow for real space data, hierarchical clustering
df_cluster_labels_mtx = cluster_n_times(df_all_data_standard,10,min_num_clusters=2,cluster_type='agglom')
df_cluster_counts_mtx = count_cluster_labels_from_mtx(df_cluster_labels_mtx)

df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx = calc_cluster_elemental_ratios(df_cluster_labels_mtx,df_all_data,df_element_ratios)
plot_cluster_elemental_ratios(df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,'Real-space hierarchical elemental ratios')

cluster_profiles_mtx, cluster_profiles_mtx_norm, num_clusters_index, cluster_index = average_cluster_profiles(df_cluster_labels_mtx,df_all_data)
df_cluster_corr_mtx, df_prevcluster_corr_mtx = correlate_cluster_profiles(cluster_profiles_mtx_norm, num_clusters_index, cluster_index)
plot_cluster_profile_corrs(df_cluster_corr_mtx, df_prevcluster_corr_mtx)

plot_all_cluster_tseries_BeijingDelhi(df_cluster_labels_mtx,ds_dataset_cat,title_prefix='Real space hierarchical, ')


plot_all_cluster_profiles(df_all_data,cluster_profiles_mtx_norm,num_clusters_index,ds_all_mz,df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,
                          df_cluster_corr_mtx,df_prevcluster_corr_mtx,df_cluster_counts_mtx,Sari_peaks_list,title_prefix='Real-space hierarchical, ')
                        
df_clust_cat_counts, df_cat_clust_counts, df_clust_time_cat_counts,df_time_cat_clust_counts = count_clusters_project_time(
    df_cluster_labels_mtx,ds_dataset_cat,ds_time_cat,title_prefix='Real space hierarchical, ',title_suffix='')

compare_cluster_metrics(df_all_data_standard,2,10,'agglom','Real space ',' metrics')


#%%
##Need to save top 30 peaks from each thingy
def save_cluster_profile_peaks(cluster_profiles_mtx_norm, num_clusters_index, num_clusters,Sari_peaks_list,filepath,header):
    
    #Make the initial file
    f = open(filepath,'w')
    f.write(header + '\n') #Give your csv text here.
    f.close()
    
    #Work out the indices of your clusters
    num_clusters_index = np.atleast_1d(num_clusters_index)
    
    if((num_clusters_index.shape[0])==1):    #Check if min number of clusters is 1
        if(num_clusters_index[0] == num_clusters):
            x_idx=0
        else:
            print("plot_cluster_profiles() error: nclusters is " + str(num_clusters) + " which is not in num_clusters_index")
    else:
        x_idx = np.searchsorted(num_clusters_index,num_clusters,side='left')
        if(x_idx == np.searchsorted(num_clusters_index,num_clusters,side='right')):
            print("plot_cluster_profiles() error: nclusters is " + str(num_clusters) + " which is not in num_clusters_index")
            return 0
    
    for y_idx in np.arange(num_clusters):
        this_cluster_profile = cluster_profiles_mtx_norm[x_idx,y_idx,:]
       
        ds_this_cluster_profile = pd.Series(this_cluster_profile,index=df_all_data.columns).T
        df_top_peaks = cluster_extract_peaks(ds_this_cluster_profile, df_all_data.T,50,dp=1,dropRT=False).drop('Name',axis=1)
        df_top_peaks.index = df_top_peaks.index.str.replace(' ', '')
        #pdb.set_trace()
        #Write mini header for this cluster
        f = open(filepath,'a')
        f.write('\nCluster ' + str(y_idx) + '\n') #Give your csv text here.
        f.close()
        #Append to CSV
        df_top_peaks.to_csv(filepath,mode='a',index=False)

csvpath = r'C:\Work\temp\test.csv'
save_cluster_profile_peaks(cluster_profiles_mtx_norm, num_clusters_index, 4,Sari_peaks_list,csvpath,header='Real-space data, k-medoids, 4 clusters')
# # %%#How many clusters should we have? Real space
# min_clusters = 2
# max_clusters = 10

# num_clusters_index = range(min_clusters,(max_clusters+1),1)
# ch_score = np.empty(len(num_clusters_index))
# db_score = np.empty(len(num_clusters_index))
# silhouette_scores = np.empty(len(num_clusters_index))

# for num_clusters in num_clusters_index:
#     agglom_native = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
#     clustering = agglom_native.fit(df_all_data_standard)
#     ch_score[num_clusters-min_clusters] = calinski_harabasz_score(df_all_data_standard, clustering.labels_)
#     db_score[num_clusters-min_clusters] = davies_bouldin_score(df_all_data_standard, clustering.labels_)
#     silhouette_scores[num_clusters-min_clusters] = silhouette_score(df_all_data_standard, clustering.labels_)
# fig,ax1 = plt.subplots(figsize=(10,6))
# ax2=ax1.twinx()
# ax1.plot(num_clusters_index,ch_score,label="CH score")
# ax2.plot(num_clusters_index,db_score,c='red',label="DB score")
# ax2.plot(num_clusters_index,silhouette_scores,c='black',label="Silhouette score")
# ax1.set_xlabel("Num clusters")
# ax1.set_ylabel("CH score")
# ax2.set_ylabel("DB score")
# plt.title("How many clusters? Real-space hierarchical")
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)
# #ax1.xaxis.set_major_locator(plticker.MultipleLocator(base=5.0))
# #ax1.xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
# ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
# plt.show()




#%%Based on the above, 4 clusters looks good with KMedoids
#Clustering metrics look like 4, particularly CH and DB scores
#Going up to 5 gives you 2 clusters where the mean mass spec look almost identical, R=0.96
real_ss_clustering = KMedoids(n_clusters = 4).fit(df_all_data_standard)
#real_ss_clustering = KMeans(n_clusters = 4).fit(df_all_data_standard)
ds_real_ss_cluster_labels = pd.Series(real_ss_clustering.labels_,index=df_all_data.index)
ds_real_ss_cluster_labels.to_csv(r'C:\Users\mbcx5jt5\Dropbox (The University of Manchester)\Complex-SOA\Clustering\all_data_SS_KMedoids4.csv',header=False,index=False)

real_ss_cluster_centers = real_ss_clustering.cluster_centers_
real_ss_cluster_centers_unscaled = standardscaler_all.inverse_transform(real_ss_cluster_centers)

df_element_ratios.to_csv(r'C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\PMF data\all_data_element_ratios.csv',index=False)

#%%Divvy up the dataset by cluster label

# a = df_all_data.loc[ds_real_ss_cluster_labels==0]

# df_all_err




#%%Visualise real standardscaled data with tsne
#Make an array with the data and cluster centers in







tsne = TSNE(n_components=2,random_state=0)
tsne_input = np.concatenate((real_ss_cluster_centers ,df_all_data_standard.to_numpy()),axis=0)
tsne_output = tsne.fit_transform(tsne_input)
tsne_centers, tsne_data = np.array_split(tsne_output,[4],axis=0)

tsne2 = TSNE(n_components=2,random_state=0)
tsne_input2 = np.concatenate((real_ss_cluster_centers_unscaled ,df_all_data.to_numpy()),axis=0)
tsne_output2 = tsne.fit_transform(tsne_input2)
tsne_centers2, tsne_data2 = np.array_split(tsne_output2,[4],axis=0)




colormap = ['gray','blue','red','yellow','purple','aqua','gold','k','orange']

cmap_EOS11 = cmap_EOS11()
#%%Plot Tsne. Top two are tsne of unscaled data, bottom two are for scaled data
fig,ax = plt.subplots(1,2,figsize=(10,5))
ax = ax.ravel()
scatter1 = ax[0].scatter(tsne_data[:, 0], tsne_data[:, 1],
            c=ds_real_ss_cluster_labels,
            cmap=ListedColormap(colormap[0:num_clusters]).reversed())
#            cmap=cmap_EOS11)
scatter2 = ax[0].scatter(tsne_centers[:,0], tsne_centers[:,1],c='k',marker='x',s=250)
ax[0].set_xlabel('T-SNE dimension 0')
ax[0].set_ylabel('T-SNE dimension 1')
ax[0].legend(handles=scatter1.legend_elements()[0], labels=['0','1','2','3'],title='Cluster')

scatter3 = ax[1].scatter(tsne_data[:, 0], tsne_data[:, 1],
            c=ds_dataset_cat.cat.codes.to_numpy(),
            cmap=cmap_EOS11)
ax[1].set_xlabel('T-SNE dimension 0')
ax[1].set_ylabel('T-SNE dimension 1')
ax[1].legend(handles=scatter3.legend_elements()[0],labels=ds_dataset_cat.cat.categories.to_list(),title='Dataset')

# scatter4 = ax[2].scatter(tsne_data2[:, 0], tsne_data2[:, 1],
#             c=ds_real_ss_cluster_labels,
#             cmap=ListedColormap(colormap[0:num_clusters]).reversed())
# #            cmap=cmap_EOS11)
# scatter5 = ax[2].scatter(tsne_centers2[:,0], tsne_centers2[:,1],c='k',marker='x',s=250)
# ax[2].set_xlabel('T-SNE dimension 0')
# ax[2].set_ylabel('T-SNE dimension 1')
# ax[2].legend(handles=scatter4.legend_elements()[0], labels=['0','1','2','3'],title='Cluster',loc='lower right')

# scatter6 = ax[3].scatter(tsne_data2[:, 0], tsne_data2[:, 1],
#             c=ds_dataset_cat.cat.codes.to_numpy(),
#             cmap=cmap_EOS11)
# ax[3].set_xlabel('T-SNE dimension 0')
# ax[3].set_ylabel('T-SNE dimension 1')
# ax[3].legend(handles=scatter6.legend_elements()[0],labels=ds_dataset_cat.cat.categories.to_list(),title='Dataset',loc='lower right')

# ax[0].set_title('TSNE, standardscaled data')
# ax[2].set_title('TSNE, unscaled data')


plt.tight_layout()
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



#%%Run standard AE and cluster many times and see how consistent it is
import gc
num_runs = 20
cluster_labels_runs = []
final_loss = []
for i in range(num_runs):
    tf.keras.backend.clear_session()
    gc.collect()
    ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=3,int_layers=3)
    history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=20)
    latent_space = ae_obj.encoder(ae_input_val).numpy()
    
    #Do clustering
    clustering = KMedoids(n_clusters = 5).fit(latent_space)
    cluster_labels_runs.append(relabel_clusters_most_freq(clustering.labels_))
    final_loss.append(history.history['val_loss'][-1])
    
    
    
cluster_labels_runs = np.asarray(cluster_labels_runs)

#Compare similarity to first run using adjusted rand score
arand_scores = []
arand_scores.append(np.nan)
for i in range(1,num_runs):
    arand_scores.append(adjusted_rand_score(cluster_labels_runs[0,:],cluster_labels_runs[i,:]))




#%%Run PCA with many different dimensions, and see how similar they all are when clustering
min_dims = 2
max_dims = 50
cluster_labels_pca = []
total_explained_variance_ratio_pca = []
arand_scores_pca = []
num_dims_pca = list(reversed(range(min_dims,max_dims)))

pca_input = df_all_data_standard


for n_components in num_dims_pca:
    pca = PCA(n_components = n_components,svd_solver = 'full').fit(pca_input)
    pca_space =pca.transform(pca_input)
    explained_variance_ratio_pca = pca.explained_variance_ratio_
    total_explained_variance_ratio_pca.append(np.sum(explained_variance_ratio_pca))
    #Scale data for clustering
    
    
    #Do clustering
    clustering = KMedoids(n_clusters = 5).fit(pca_space)
    cluster_labels_pca.append(clustering.labels_)
    if len(arand_scores_pca) == 0 :
        arand_scores_pca.append(np.nan)
        cluster_labels_run0 = clustering.labels_
    else:
        arand_scores_pca.append(adjusted_rand_score(cluster_labels_run0,clustering.labels_))
        
cluster_labels_pca = np.asarray(cluster_labels_pca)


plt.plot(num_dims_pca,total_explained_variance_ratio_pca)

# Do you run standard scaling before PCA?
# If you dont, you are cutting out the parts that are the smallest signals. But these are the ones you are
# maximising by doing standardscaler in the first place
# So I think you have to run standardscaler first?
# The justification for using any prescaling has to be fore clustering and PCA the same, ie you are interested in
# The composition not the total amounts

# using standardscaler befoer PCA, you need 36 components in pca to get 90% of total explained variance

#%%PCA with 36 components
pca = PCA(n_components = 36,svd_solver = 'full').fit(pca_input)
pca_space =pca.transform(pca_input)

# %%#How many clusters should we have? PCA36 space
min_clusters = 2
max_clusters = 10

num_clusters_index = range(min_clusters,(max_clusters+1),1)
ch_score = np.empty(len(num_clusters_index))
db_score = np.empty(len(num_clusters_index))
silhouette_scores = np.empty(len(num_clusters_index))

for num_clusters in num_clusters_index:
    clustering = KMeans(n_clusters = num_clusters).fit(pca_space)
    #clustering = KMedoids(n_clusters = num_clusters).fit(pca_space)
    ch_score[num_clusters-min_clusters] = calinski_harabasz_score(pca_space, clustering.labels_)
    db_score[num_clusters-min_clusters] = davies_bouldin_score(pca_space, clustering.labels_)
    silhouette_scores[num_clusters-min_clusters] = silhouette_score(pca_space, clustering.labels_)
fig,ax1 = plt.subplots(figsize=(10,6))
ax2=ax1.twinx()
ax1.plot(num_clusters_index,ch_score,label="CH score")
ax2.plot(num_clusters_index,db_score,c='red',label="DB score")
ax2.plot(num_clusters_index,silhouette_scores,c='black',label="Silhouette score")
ax1.set_xlabel("Num clusters")
ax1.set_ylabel("CH score")
ax2.set_ylabel("DB score")
plt.title("How many clusters? PCA-space k-means")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)
#ax1.xaxis.set_major_locator(plticker.MultipleLocator(base=5.0))
#ax1.xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()


#%%#test cluster PCA vs non-pca
cluster_labels_5_pca = relabel_clusters_most_freq(KMeans(n_clusters = 5).fit(df_all_data_standard).labels_)
cluster_labels_5 = relabel_clusters_most_freq(KMeans(n_clusters = 5).fit(pca_space).labels_)
print(adjusted_rand_score(cluster_labels_5_pca,cluster_labels_5))

print("They are just the same!!!")



#%%Plot latent space
cmap_EOS11 = cmap_EOS11()

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

data_to_cluster = df_latent_space

for num_clusters in num_clusters_index:
    agglom_native = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    #agglom_native = KMeans(n_clusters = num_clusters)
    clustering = agglom_native.fit(data_to_cluster)
    ch_score[num_clusters-min_clusters] = calinski_harabasz_score(data_to_cluster, clustering.labels_)
    db_score[num_clusters-min_clusters] = davies_bouldin_score(data_to_cluster, clustering.labels_)
    silhouette_scores[num_clusters-min_clusters] = silhouette_score(data_to_cluster, clustering.labels_)
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
cmap_EOS11 = cmap_EOS11()

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data_to_cluster.iloc[:,0], data_to_cluster.iloc[:,1],data_to_cluster.iloc[:,2], c=ds_dataset_cat.cat.codes,cmap=cmap_EOS11)
#scatter = ax.scatter(data_to_cluster.iloc[:,0], data_to_cluster.iloc[:,1],data_to_cluster.iloc[:,2], c=df_cluster_labels_mtx[4])
ax.set_xlabel('Latent space dim 0')
ax.set_ylabel('Latent space dim 1')
ax.set_zlabel('Latent space dim 2')
#ax.set_box_aspect([np.ptp(i) for i in latent_space])  # equal aspect ratio

labels = ds_dataset_cat.drop_duplicates()#['type1', 'type2', 'type3', 'type4']
handles, _ = scatter.legend_elements(prop="colors", alpha=0.6) # use my own labels
legend1 = ax.legend(handles, labels, loc="upper right")
ax.add_artist(legend1)
plt.title('3 dimensional latent space, VAE')
plt.show()



#%%Build and trainVAE
#beta_schedule = np.array([1e-5,1e-4,1e-3,1e-2,1e-1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
beta_schedule = np.array([1e-5,1e-4,1e-3,1e-2,1e-1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0,1.25])

# beta_schedule = np.concatenate([np.zeros(5),np.arange(0.1,1.1,0.1),np.ones(10,),np.arange(0.9,0.45,-0.05)])

# beta_schedule = np.abs(np.sin(np.arange(0,201)*0.45/math.pi))

#beta_schedule = 1 - np.cos(np.arange(0,50)*0.45/math.pi)

vae_obj = VAE_n_layer(input_dim=input_dim,latent_dim=3,int_layers=2,beta_schedule=beta_schedule)



df_aug = augment_data_noise(df_all_data_norm1,50,1,0)
ae_input = df_aug.values
ae_input_val = df_all_data_norm1.values
input_dim = ae_input.shape[1]



history_vae = vae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=20,verbose=1,callbacks=[TerminateOnNaN()])


#history_vae = vae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=50,verbose=1)
latent_space = vae_obj.encoder(ae_input_val).numpy()
df_latent_space = pd.DataFrame(latent_space,index=df_all_data.index)
plt.scatter(latent_space[:,0],latent_space[:,1])


fig,ax = plt.subplots(1)
plt.scatter(ae_input_val,vae_obj.ae(ae_input_val))
plt.title('AE input vs output')
plt.xlabel('AE input')
plt.ylabel('AE output')
plt.show()

#%%
fig,ax=plt.subplots(2,1,figsize=(12,8))
ax[0].set_title('Finding optimum epochs- simple relu VAE')
ax[0].plot(history_vae.epoch,history_vae.history['val_loss'],c='k')
ax[0].plot(history_vae.epoch,history_vae.history['val_mse'],c='r')
ax[0].plot(history_vae.epoch,history_vae.history['val_kl'],c='b')
ax[0].set_xlabel('Number of epochs')
ax[0].set_ylabel('Loss')
ax[1].plot(history_vae.epoch,history_vae.history['val_loss'],label='Total loss',c='k')
ax[1].plot(history_vae.epoch,history_vae.history['val_mse'],label='MSE loss',c='r')
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

plt.scatter(latent_space[:,0],latent_spsace[:,1])

#%%Plot latent space
cmap_EOS11 = cmap_EOS11()

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
plt.title('3 dimensional latent space')
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


from collections import deque

import gc

verbose = 1

start_time = time.time()

latent_dims = np.arange(1,10)
VAE1_MSE_20 = np.ones(len(latent_dims))*-1
VAE2_MSE_20 = np.ones(len(latent_dims))*-1
VAE3_MSE_20 = np.ones(len(latent_dims))*-1
VAE4_MSE_20 = np.ones(len(latent_dims))*-1
VAE1_KL_20 = np.ones(len(latent_dims))*-1
VAE2_KL_20 = np.ones(len(latent_dims))*-1
VAE3_KL_20 = np.ones(len(latent_dims))*-1
VAE4_KL_20 = np.ones(len(latent_dims))*-1


# init_op = tf.global_variables_initializer()
# # Try this to keep memory in check
# with tf.Session() as sess:
#     # Run the init operation. 
#     # This will make sure that memory is only allocated for the variable once.
#     sess.run(init_op)


for int_layers in range(1,5):

    #for latent_dim in latent_dims:
    queue = deque(latent_dims)
    while(queue):
        latent_dim=queue[0]
        print(latent_dim)
        tf.keras.backend.clear_session()
        gc.collect()
        
        vae_obj = VAE_n_layer(input_dim=input_dim,latent_dim=latent_dim,int_layers=int_layers,beta_schedule=beta_schedule,learning_rate=1e-3)
        

        #pdb.set_trace()
        history_vae = vae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=20,verbose=1,callbacks=[TerminateOnNaN()])
    
        if(math.isnan(history_vae.history['loss'][-1])==False and math.isinf(history_vae.history['loss'][-1])==False):
            #do nothing???
            #loopthing.rotate(-1)
        #else:    
            if(int_layers==1):
                VAE1_MSE_20[latent_dim-1] = history_vae.history['val_mse'][-1]
                VAE1_KL_20[latent_dim-1] = history_vae.history['val_kl'][-1]
            elif(int_layers==2):
                VAE2_MSE_20[latent_dim-1] = history_vae.history['val_mse'][-1]
                VAE2_KL_20[latent_dim-1] = history_vae.history['val_kl'][-1]
            elif(int_layers==3):
                VAE3_MSE_20[latent_dim-1] = history_vae.history['val_mse'][-1]
                VAE3_KL_20[latent_dim-1] = history_vae.history['val_kl'][-1]
            elif(int_layers==4):
                VAE4_MSE_20[latent_dim-1] = history_vae.history['val_mse'][-1]
                VAE4_KL_20[latent_dim-1] = history_vae.history['val_kl'][-1]
            queue.popleft()

    
print("--- %s seconds ---" % (time.time() - start_time))


#%%Plot the data for the different number of layers above
fig,ax=plt.subplots(2,1,figsize=(12,8))
ax[0].set_title('Finding optimum latent dims- relu VAE')
ax[0].plot(latent_dims,VAE1_MSE_20,c='black')
#ax[0].plot(latent_dims,VAE1_KL_20,c='b')
ax[0].plot(latent_dims,VAE2_MSE_20,c='red')
ax[0].plot(latent_dims,VAE3_MSE_20,c='gray')
ax[0].plot(latent_dims,VAE4_MSE_20,c='b')
ax[0].set_xlabel('Number of latent dims')
ax[0].set_ylabel('MSE after 20 epochs')
ax[1].plot(latent_dims,VAE1_MSE_20,c='black')
#ax[1].plot(latent_dims,VAE1_KL_20,c='b')
ax[1].plot(latent_dims,VAE2_MSE_20,c='red')
ax[1].plot(latent_dims,VAE3_MSE_20,c='gray')
ax[1].plot(latent_dims,VAE4_MSE_20,c='b')
ax[1].set_xlabel('Number of latent dims')
ax[1].set_ylabel('MSE after 20 epochs')
ax[1].set_yscale('log')
ax[1].legend([1,2,3,4],title='Intermediate layers')


loc = plticker.MultipleLocator(base=10.0) # this locator puts ticks at regular intervals
ax[0].xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
#ax[0].xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
ax[1].xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
#ax[1].xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))

plt.show()

fig,ax=plt.subplots(2,1,figsize=(12,8))
ax[0].set_title('Finding optimum latent dims- relu VAE')
ax[0].plot(latent_dims,VAE1_KL_20,c='black')
#ax[0].plot(latent_dims,VAE1_KL_20,c='b')
ax[0].plot(latent_dims,VAE2_KL_20,c='red')
ax[0].plot(latent_dims,VAE3_KL_20,c='gray')
ax[0].plot(latent_dims,VAE4_KL_20,c='b')
ax[0].set_xlabel('Number of latent dims')
ax[0].set_ylabel('KL after 20 epochs')
ax[1].plot(latent_dims,VAE1_KL_20,c='black')
#ax[1].plot(latent_dims,VAE1_KL_20,c='b')
ax[1].plot(latent_dims,VAE2_KL_20,c='red')
ax[1].plot(latent_dims,VAE3_KL_20,c='gray')
ax[1].plot(latent_dims,VAE4_KL_20,c='b')
ax[1].set_xlabel('Number of latent dims')
ax[1].set_ylabel('KL after 20 epochs')
ax[1].set_yscale('log')
ax[1].legend([1,2,3,4],title='Intermediate layers')


loc = plticker.MultipleLocator(base=10.0) # this locator puts ticks at regular intervals
ax[0].xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
#ax[0].xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
ax[1].xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
#ax[1].xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))

plt.show()


#%%Test peak CH index for multiple VAE trainings
df_aug = augment_data_noise(df_all_data_norm1,50,1,0)
ae_input = df_aug.values
ae_input_val = df_all_data_norm1.values
input_dim = ae_input.shape[1]

beta_schedule = np.array([1e-5,1e-4,1e-3,1e-2,1e-1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0,1.25])*2


#beta_schedule = np.tile(np.concatenate([np.arange(0,1.1,0.1),np.ones(9)]),4)

#beta_schedule = np.concatenate([np.arange(0,2,0.1),np.ones(9)*2])



num_runs = 10
min_clusters = 2
max_clusters = 10

peak_CH = []
peak_CH_clusters = []

#Set this for Kmeans or it complains
os.environ["OMP_NUM_THREADS"] = "2"

for i in range(num_runs):
    vae_obj = VAE_n_layer(input_dim=input_dim,latent_dim=3,int_layers=2,beta_schedule=beta_schedule)
    history_vae = vae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=50,verbose=0,callbacks=[TerminateOnNaN()])    
    latent_space = vae_obj.encoder(ae_input_val).numpy()
    
    
    num_clusters_index = range(min_clusters,(max_clusters+1),1)
    ch_score = np.empty(len(num_clusters_index))
    #db_score = np.empty(len(num_clusters_index))
    #silhouette_scores = np.empty(len(num_clusters_index))

    data_to_cluster = StandardScaler().fit_transform(latent_space)
    
    

    for num_clusters in num_clusters_index:
        #agglom_native = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
        agglom_native = KMedoids(n_clusters = num_clusters)
        clustering = agglom_native.fit(data_to_cluster)
        ch_score[num_clusters-min_clusters] = calinski_harabasz_score(data_to_cluster, clustering.labels_)
        #db_score[num_clusters-min_clusters] = davies_bouldin_score(data_to_cluster, clustering.labels_)
        #silhouette_scores[num_clusters-min_clusters] = silhouette_score(data_to_cluster, clustering.labels_)
    
    #pdb.set_trace()
    max_idx = ch_score.argmax()
    peak_CH_clusters.append(num_clusters_index[max_idx])
    peak_CH.append(ch_score[max_idx])






#%%Test cluster labels for multiple VAE trainings
df_aug = augment_data_noise(df_all_data_norm1,50,1,0)
ae_input = df_aug.values
ae_input_val = df_all_data_norm1.values
input_dim = ae_input.shape[1]

beta_schedule = np.array([1e-5,1e-4,1e-3,1e-2,1e-1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0,1.25])


#beta_schedule = np.tile(np.concatenate([np.arange(0,1.1,0.1),np.ones(9)]),4)

#beta_schedule = np.concatenate([np.arange(0,2,0.1),np.ones(9)*2])



num_runs = 10


cluster_labels = []

num_clusters = 6
ch_score_6 = []

#Set this for Kmeans or it complains
os.environ["OMP_NUM_THREADS"] = "2"

for i in range(num_runs):
    vae_obj = VAE_n_layer(input_dim=input_dim,latent_dim=3,int_layers=2,beta_schedule=beta_schedule)
    history_vae = vae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=20,verbose=0,callbacks=[TerminateOnNaN()])    
    latent_space = vae_obj.encoder(ae_input_val).numpy()
    
    
    num_clusters_index = range(min_clusters,(max_clusters+1),1)
    ch_score = np.empty(len(num_clusters_index))
    #db_score = np.empty(len(num_clusters_index))
    #silhouette_scores = np.empty(len(num_clusters_index))

    data_to_cluster = StandardScaler().fit_transform(latent_space)
    
    

      #agglom_native = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    agglom_native = KMedoids(n_clusters = num_clusters)
    clustering = agglom_native.fit(data_to_cluster)
        
    #pdb.set_trace()
    cluster_labels.append(relabel_clusters_most_freq(clustering.labels_))
    ch_score_6.append(calinski_harabasz_score(data_to_cluster, clustering.labels_))

cluster_labels = np.asarray(cluster_labels)
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

data_to_cluster = latent_space

for num_clusters in num_clusters_index:
    #agglom_native = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    agglom_native = KMeans(n_clusters = num_clusters)
    clustering = agglom_native.fit(data_to_cluster)
    ch_score[num_clusters-min_clusters] = calinski_harabasz_score(data_to_cluster, clustering.labels_)
    db_score[num_clusters-min_clusters] = davies_bouldin_score(data_to_cluster, clustering.labels_)
    silhouette_scores[num_clusters-min_clusters] = silhouette_score(data_to_cluster, clustering.labels_)
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




#%%Try PCA then clustering

#%%PCA transform the native dataset

pca3 = PCA(n_components = 3)
pca3_space = pca3.fit_transform(df_all_data)
pca7 = PCA(n_components = 7)
pca7_space = pca7.fit_transform(df_all_data)

#%%Plot pca3 space
cmap_EOS11 = cmap_EOS11()

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
#scatter = ax.scatter(log_latent_space[:,0], log_latent_space[:,1],log_latent_space[:,2], c=ds_dataset_cat.cat.codes,cmap=cmap_EOS11)
scatter = ax.scatter(pca3_space[:,0], pca3_space[:,1],pca3_space[:,2], c=ds_dataset_cat.cat.codes,cmap=cmap_EOS11)
ax.set_xlabel('Latent space dim 0')
ax.set_ylabel('Latent space dim 1')
ax.set_zlabel('Latent space dim 2')
#ax.set_box_aspect([np.ptp(i) for i in latent_space])  # equal aspect ratio

labels = ds_dataset_cat.drop_duplicates()#['type1', 'type2', 'type3', 'type4']
handles, _ = scatter.legend_elements(prop="colors", alpha=0.6) # use my own labels
legend1 = ax.legend(handles, labels, loc="upper right")
ax.add_artist(legend1)
plt.title('3 dimensional PCA space')
plt.show()

# %%#How many clusters should we have? PCA3 space
min_clusters = 2
max_clusters = 10

num_clusters_index = range(min_clusters,(max_clusters+1),1)
ch_score = np.empty(len(num_clusters_index))
db_score = np.empty(len(num_clusters_index))
silhouette_scores = np.empty(len(num_clusters_index))

#data_to_cluster = pca3_space

data_to_cluster = StandardScaler().fit_transform(pca3_space)

for num_clusters in num_clusters_index:
    #agglom_native = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    #agglom_native = KMeans(n_clusters = num_clusters)
    agglom_native = KMedoids(n_clusters = num_clusters)
    
    clustering = agglom_native.fit(data_to_cluster)
    ch_score[num_clusters-min_clusters] = calinski_harabasz_score(data_to_cluster, clustering.labels_)
    db_score[num_clusters-min_clusters] = davies_bouldin_score(data_to_cluster, clustering.labels_)
    silhouette_scores[num_clusters-min_clusters] = silhouette_score(data_to_cluster, clustering.labels_)
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