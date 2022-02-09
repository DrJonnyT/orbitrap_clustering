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


from sklearn.metrics.cluster import contingency_matrix

import scipy.cluster.hierarchy as sch
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score,silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as plticker

import skfuzzy as fuzz

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pickle

import os
os.chdir('C:/Work/Python/Github/Orbitrap_clustering')
from ae_functions import *

# %%Load data

path='C:/Users/mbcx5jt5/Google Drive/Shared_York_Man2/'
df_beijing_raw, df_beijing_filters, df_beijing_metadata = beijing_load(
    path + 'BJ_UnAmbNeg9.1.1_20210505-Times_Fixed.xlsx',path + 'BJ_UnAmbNeg9.1.1_20210505-Times_Fixed.xlsx',
    peaks_sheetname="Compounds",metadata_sheetname="massloading_Beijing")

df_delhi_raw, df_delhi_filters, df_delhi_metadata = delhi_load(
    path + 'Delhi_Amb3.1_MZ.xlsx',path + 'Delhi/Delhi_massloading_autumn_summer.xlsx')

df_beijing_winter = df_beijing_filters.iloc[0:124].copy()
df_beijing_summer = df_beijing_filters.iloc[124:].copy()

df_all_filters = df_beijing_filters.append(df_delhi_filters,sort=True).fillna(0)
df_all_raw = df_beijing_raw.transpose().append(df_delhi_raw.transpose(),sort=True).transpose()

# %%Check largest peaks

#IM STILLL WORKING OUT HOW TO JOIN THEM ALL TOGETHER BEARING IN MIND THAT NOT ALL MOLECULES ARE THE SAME IN THE NAMELISTS
#YOU END UP WITH SOME AS A LIST, SOME AS [], SOME AS THE RIGHT MOLECULE, ITS QUITE ANNOYING AND I WANT TO JUST USE A FOR LOOP
#LIKE A NORMAL PERSON

chemform_namelist_beijing = load_chemform_namelist(path + 'Beijing_Amb3.1_MZ.xlsx')
chemform_namelist_delhi = load_chemform_namelist(path + 'Delhi_Amb3.1_MZ.xlsx')
# chemform_namelist_all = chemform_namelist_beijing.append(chemform_namelist_delhi)#,sort=True)

# chemform_namelist_all = chemform_namelist_all.drop_duplicates()

# C = [','.join(z) for z in zip(chemform_namelist_beijing['Name'], chemform_namelist_delhi['Name'])]
# #pd.concat([chemform_namelist_beijing,chemform_namelist_delhi],axis=0)
# #test = 

# #chemform_namelist_beijing['Name'].apply(lambda x: '' if '-' in x else x)
# #chemform_namelist_beijing['Name'].apply(lambda x: '' if '-' in x else chemform_namelist_beijing.index[x])


# #chemform_namelist_beijing['Name'] = chemform_namelist_beijing['Name'].astype('string')
# #chemform_namelist_beijing['Name'] = np.where(chemform_namelist_beijing['Name'] == '[]',chemform_namelist_beijing.index,chemform_namelist_beijing['Name'])
# #chemform_namelist_beijing = chemform_namelist_beijing.drop_duplicates()


# print("BEIJING SUMMER")
# a =cluster_extract_peaks(df_beijing_summer.sum(), df_beijing_raw,10,chemform_namelist_beijing)
# print("BEIJING WINTER")
# a = cluster_extract_peaks(df_beijing_winter.sum(), df_beijing_raw,10,chemform_namelist_beijing)
# print("DELHI AUTUMN")
# a = cluster_extract_peaks(df_delhi_filters.sum(), df_delhi_raw,10,chemform_namelist_delhi)

# print("BEIJING BLANK")
# a = cluster_extract_peaks(df_beijing_raw.iloc[:,320].transpose(), df_beijing_raw,10,chemform_namelist_beijing)

# print("DELHI BLANKS MEAN")
# a = cluster_extract_peaks(df_delhi_raw_blanks.transpose().mean(), df_delhi_raw,10,chemform_namelist_delhi)

# print("COMBINED BEIJING/DELHI")
# a = cluster_extract_peaks(df_all_filters.sum(), df_all_raw,10,C)




#%%

#Augment the data
df_aug = augment_data_noise(df_beijing_filters,50,1,0)


#%%Scale data for input into AE
scalefactor = 1e6
pipe_1e6 = FunctionTransformer(lambda x: np.divide(x,scalefactor),inverse_func = lambda x: np.multiply(x,scalefactor))
pipe_1e6.fit(df_aug)
scaled_df = pd.DataFrame(pipe_1e6.transform(df_aug),columns=df_beijing_filters.columns)
ae_input=scaled_df.to_numpy()
scaled_df_val = pd.DataFrame(pipe_1e6.transform(df_beijing_filters), columns=df_beijing_filters.columns,index=df_beijing_filters.index)
ae_input_val = scaled_df_val.to_numpy()

minmax = MinMaxScaler()
beijing_winter_minmax = minmax.fit_transform(ae_input_val)


#%%Now compare loss for different latent dimensions
#This is NOT using kerastuner, and is using log-spaced intermediate layers
#WARNING THIS TAKES ABOUT HALF AN HOUR
latent_dims = []
AE1_MSE_best50epoch =[]
AE2_MSE_best50epoch =[]
AE3_MSE_best50epoch =[]
AE4_MSE_best50epoch =[]
best_hps_array = []

input_dim = ae_input.shape[1]
verbose = 0

for latent_dim in range(1,2):
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
ax[0].xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
ax[0].xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
ax[1].xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
ax[1].xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))

plt.show()


#%%Work out how many epochs to train for
#Based on the above, use an AE with 3 intermediate layers and latent dim of 4
ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=4,int_layers=3)
history = ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=300)
val_acc_per_epoch = history.history['val_loss']
best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
ae_obj = AE_n_layer(input_dim=input_dim,latent_dim=7,int_layers=2)
ae_obj.fit_model(ae_input, x_test=ae_input_val,epochs=best_epoch)
print('Best epoch: %d' % (best_epoch,))

#%%Plot loss vs epochs

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss)+1)
fig,ax=plt.subplots(2,1,figsize=(12,8))
ax[0].plot(epochs, loss, 'bo', label='Training loss')
ax[0].plot(epochs, val_loss, 'b', label='Validation loss')
ax[0].set_title('Training and validation loss- 3 intermediate layer AE')
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
latent_space = ae_obj.encoder(ae_input_val).numpy()
decoded_latent_space = ae_obj.decoder(latent_space)

fig,ax = plt.subplots(1)
plt.scatter(ae_input_val,ae_obj.ae(ae_input_val))
plt.title("AE input vs output")
plt.xlabel('AE input')
plt.ylabel('AE output')
plt.show()


#%%Now do some clustering on the latent space
############################################################################################
#CLUSTERING AND FACTOR ANALYSIS OF LATENT SPACE DATA
############################################################################################
#%%Latent space dendrogram
#Lets try a dendrogram to work out the optimal number of clusters
fig,axes = plt.subplots(1,1,figsize=(20,10))
plt.title('Latent space dendrogram')
dendrogram = sch.dendrogram(sch.linkage(latent_space, method='ward'))
plt.show()
#It comes out with 2 clusters for the 2 cluster solution so that's good

# %%#How many clusters should we have? Latent space
min_clusters = 2
max_clusters = 30

num_clusters_index = range(min_clusters,(max_clusters+1),1)
ch_score = np.empty(len(num_clusters_index))
db_score = np.empty(len(num_clusters_index))
silhouette_scores = np.empty(len(num_clusters_index))

for num_clusters in num_clusters_index:
    agglom_native = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    #agglom_native = KMeans(n_clusters = num_clusters)
    clustering = agglom_native.fit(latent_space)
    ch_score[num_clusters-min_clusters] = calinski_harabasz_score(latent_space, clustering.labels_)
    db_score[num_clusters-min_clusters] = davies_bouldin_score(latent_space, clustering.labels_)
    silhouette_scores[num_clusters-min_clusters] = silhouette_score(latent_space, clustering.labels_)
fig,ax1 = plt.subplots(figsize=(10,6))
ax2=ax1.twinx()
ax1.plot(num_clusters_index,ch_score,label="CH score")
ax2.plot(num_clusters_index,db_score,c='red',label="DB score")
ax2.plot(num_clusters_index,silhouette_scores,c='black',label="Silhouette score")
ax1.set_xlabel("Num clusters")
ax1.set_ylabel("CH score")
ax2.set_ylabel("DB score")
plt.title("How many clusters? AE Latent-space data")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)
ax1.xaxis.set_major_locator(plticker.MultipleLocator(base=5.0))
ax1.xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
plt.show()




#%%tSNE plots for latent space clusters
colormap = ['k','blue','red','yellow','gray','purple','aqua','gold']

for num_clusters in range(2,10):
    agglom = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    clustering = agglom.fit(latent_space)
    # plt.plot(clustering.labels_)
    # plt.scatter(df_beijing_filters.index,clustering.labels_,marker='.')
    
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(latent_space)
    plt.figure(figsize=(9,5))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                c=clustering.labels_,
                cmap=ListedColormap(colormap[0:num_clusters]))

    bounds = np.arange(0,num_clusters+1) -0.5
    ticks = np.arange(0,num_clusters)
    plt.colorbar(boundaries=bounds,ticks=ticks,label='Cluster')
    plt.title('tSNE latent space, ' + str(num_clusters) + ' clusters')
    plt.show()



# #%%
# #And what are the clusters?
# #The cluster labels can just be moved straight out the latent space
# cluster0_decoded = df_3clust[clustering.labels_==0].mean()
# cluster1_decoded = df_3clust[clustering.labels_==1].mean()
# cluster2_decoded = df_3clust[clustering.labels_==2].mean()

# #Latent space clusters
# cluster0_lat = df_latent_space[clustering.labels_==0].mean()
# cluster1_lat = df_latent_space[clustering.labels_==1].mean()
# cluster2_lat = df_latent_space[clustering.labels_==2].mean()


#####################################################################################################
#####################################################################################################
#####################################################################################################
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
    #pca_ae = PCA(n_components = num_components)
    prin_comp = pca.fit_transform(np.nan_to_num(ae_input_val))
    pca_variance_explained[num_components-min_components] = pca.explained_variance_ratio_.sum()
    #prin_comp = pca.fit_transform(ae_input_val)
    #pca_scaled_variance_explained[num_components-min_components] = pca.explained_variance_ratio_.sum()
    #prin_comp_ae = pca_ae.fit_transform(latent_space)
    #pca_ae_variance_explained[num_components-min_components] = pca_ae.explained_variance_ratio_.sum()
    
fig,ax = plt.subplots(figsize=(12,8))
ax.plot(num_components_index,pca_variance_explained,label="PCA on linearly scaled data")
#ax.plot(num_components_index,pca_scaled_variance_explained,label="PCA on scaled AE input")
#ax.plot(num_components_index,pca_ae_variance_explained,c='red',label="PCA on latent space")
ax.xaxis.set_major_locator(plticker.MultipleLocator(base=5.0))
ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
ax.set_xlabel("Num PCA components")
ax.set_ylabel("Fraction of variance explained")
plt.legend(loc="lower right")
plt.title("How many PCA components?")
plt.show()

#%%Now do some clustering on the PCA
############################################################################################
#CLUSTERING AND FACTOR ANALYSIS OF PCA DATA
############################################################################################
#First do the PCA with the ideal number of components
pca = PCA(n_components = 7)
PCA_space = pca.fit_transform(np.nan_to_num(ae_input_val))

pca2 = PCA(n_components = 2)
PCA2_space = pca2.fit_transform(np.nan_to_num(ae_input_val))
#%%PCA space dendrogram
#Lets try a dendrogram to work out the optimal number of clusters
fig,axes = plt.subplots(1,1,figsize=(20,10))
plt.title('PCA space dendrogram')
dendrogram = sch.dendrogram(sch.linkage(prin_comp, method='ward'))
plt.show()
#It comes out with 2 clusters for the 2 cluster solution so that's good

# %%#How many clusters should we have? Latent space
min_clusters = 2
max_clusters = 30

num_clusters_index = range(min_clusters,(max_clusters+1),1)
ch_score = np.empty(len(num_clusters_index))
db_score = np.empty(len(num_clusters_index))
silhouette_scores = np.empty(len(num_clusters_index))

for num_clusters in num_clusters_index:
    agglom_native = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    #agglom_native = KMeans(n_clusters = num_clusters)
    clustering = agglom_native.fit(PCA_space)
    ch_score[num_clusters-min_clusters] = calinski_harabasz_score(PCA_space, clustering.labels_)
    db_score[num_clusters-min_clusters] = davies_bouldin_score(PCA_space, clustering.labels_)
    silhouette_scores[num_clusters-min_clusters] = silhouette_score(PCA_space, clustering.labels_)
fig,ax1 = plt.subplots(figsize=(10,6))
ax2=ax1.twinx()
ax1.plot(num_clusters_index,ch_score,label="CH score")
ax2.plot(num_clusters_index,db_score,c='red',label="DB score")
ax2.plot(num_clusters_index,silhouette_scores,c='black',label="Silhouette score")
ax1.set_xlabel("Num clusters")
ax1.set_ylabel("CH score")
ax2.set_ylabel("DB score")
plt.title("How many clusters? PCA-space data")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)
ax1.xaxis.set_major_locator(plticker.MultipleLocator(base=5.0))
ax1.xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
plt.show()




#%%tSNE plots for latent space clusters
colormap = ['k','blue','red','yellow','gray','purple','aqua','gold','orange']

for num_clusters in range(2,10):
    agglom = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    clustering = agglom.fit(PCA_space)
    # plt.plot(clustering.labels_)
    # plt.scatter(df_beijing_filters.index,clustering.labels_,marker='.')
    
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(PCA_space)
    plt.figure(figsize=(9,5))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                c=clustering.labels_,
                cmap=ListedColormap(colormap[0:num_clusters]))

    bounds = np.arange(0,num_clusters+1) -0.5
    ticks = np.arange(0,num_clusters)
    plt.colorbar(boundaries=bounds,ticks=ticks,label='Cluster')
    plt.title('tSNE PCA space, ' + str(num_clusters) + ' clusters')
    plt.show()
    
    
    





#%%Now do some clustering on the real-space data
############################################################################################
#CLUSTERING AND FACTOR ANALYSIS OF real-space data DATA
############################################################################################
#Do it with the linearly scaled data?
#%%Real space dendrogram
#Lets try a dendrogram to work out the optimal number of clusters
fig,axes = plt.subplots(1,1,figsize=(20,10))
plt.title('Real space dendrogram')
dendrogram = sch.dendrogram(sch.linkage(ae_input_val, method='ward'))
plt.show()


# %%#How many clusters should we have? Real-space
min_clusters = 2
max_clusters = 30

num_clusters_index = range(min_clusters,(max_clusters+1),1)
ch_score = np.empty(len(num_clusters_index))
db_score = np.empty(len(num_clusters_index))
silhouette_scores = np.empty(len(num_clusters_index))

for num_clusters in num_clusters_index:
    agglom_native = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    #agglom_native = KMeans(n_clusters = num_clusters)
    clustering = agglom_native.fit(ae_input_val)
    ch_score[num_clusters-min_clusters] = calinski_harabasz_score(ae_input_val, clustering.labels_)
    db_score[num_clusters-min_clusters] = davies_bouldin_score(ae_input_val, clustering.labels_)
    silhouette_scores[num_clusters-min_clusters] = silhouette_score(ae_input_val, clustering.labels_)
fig,ax1 = plt.subplots(figsize=(10,6))
ax2=ax1.twinx()
ax1.plot(num_clusters_index,ch_score,label="CH score")
ax2.plot(num_clusters_index,db_score,c='red',label="DB score")
ax2.plot(num_clusters_index,silhouette_scores,c='black',label="Silhouette score")
ax1.set_xlabel("Num clusters")
ax1.set_ylabel("CH score")
ax2.set_ylabel("DB score")
plt.title("How many clusters? real-space data")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)
ax1.xaxis.set_major_locator(plticker.MultipleLocator(base=5.0))
ax1.xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
plt.show()




#%%tSNE plots for latent space clusters
colormap = ['k','blue','red','yellow','gray','purple','aqua','gold','orange']

for num_clusters in range(2,10):
    agglom = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    clustering = agglom.fit(ae_input_val)
    # plt.plot(clustering.labels_)
    # plt.scatter(df_beijing_filters.index,clustering.labels_,marker='.')
    
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(ae_input_val)
    plt.figure(figsize=(9,5))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                c=clustering.labels_,
                cmap=ListedColormap(colormap[0:num_clusters]))

    bounds = np.arange(0,num_clusters+1) -0.5
    ticks = np.arange(0,num_clusters)
    plt.colorbar(boundaries=bounds,ticks=ticks,label='Cluster')
    plt.title('tSNE real space, ' + str(num_clusters) + ' clusters')
    plt.show()
    
    
    
    
    
#%%Now do some clustering on the top 70% of real-space data
############################################################################################
#CLUSTERING AND FACTOR ANALYSIS OF top 70% of real-space data DATA
############################################################################################
#Extract the peaks from the real-space data
peaks_sum = df_beijing_filters.sum()
#set negative to zero
peaks_sum = peaks_sum.clip(lower=0)
peaks_sum_norm = peaks_sum/ peaks_sum.sum()
peaks_sum_norm_sorted = peaks_sum_norm.sort_values(ascending=False)
numpeaks_top70 = peaks_sum_norm_sorted.cumsum().searchsorted(0.7)
peaks_sum_norm_sorted_cumsum = peaks_sum_norm_sorted.cumsum()

fig,ax = plt.subplots(1,figsize=(8,6))
ax.plot(peaks_sum_norm_sorted_cumsum.values)
ax.set_xlabel('Peak rank')
ax.set_ylabel('Cumulative normalised sum')
plt.show()

#Now pick off the top 70% of peaks
index_top70 = peaks_sum.nlargest(numpeaks_top70).index
df_scaled_top70 = scaled_df_val[index_top70]
scaled_top70_np = df_scaled_top70.to_numpy()


#%%Real space top 70% dendrogram
#Lets try a dendrogram to work out the optimal number of clusters
fig,axes = plt.subplots(1,1,figsize=(20,10))
plt.title('Real space top 70% dendrogram')
dendrogram = sch.dendrogram(sch.linkage(scaled_top70_np, method='ward'))
plt.show()


# %%#How many clusters should we have? Latent space
min_clusters = 2
max_clusters = 30

num_clusters_index = range(min_clusters,(max_clusters+1),1)
ch_score = np.empty(len(num_clusters_index))
db_score = np.empty(len(num_clusters_index))
silhouette_scores = np.empty(len(num_clusters_index))

for num_clusters in num_clusters_index:
    agglom_native = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    #agglom_native = KMeans(n_clusters = num_clusters)
    clustering = agglom_native.fit(scaled_top70_np)
    ch_score[num_clusters-min_clusters] = calinski_harabasz_score(scaled_top70_np, clustering.labels_)
    db_score[num_clusters-min_clusters] = davies_bouldin_score(scaled_top70_np, clustering.labels_)
    silhouette_scores[num_clusters-min_clusters] = silhouette_score(scaled_top70_np, clustering.labels_)
fig,ax1 = plt.subplots(figsize=(10,6))
ax2=ax1.twinx()
ax1.plot(num_clusters_index,ch_score,label="CH score")
ax2.plot(num_clusters_index,db_score,c='red',label="DB score")
ax2.plot(num_clusters_index,silhouette_scores,c='black',label="Silhouette score")
ax1.set_xlabel("Num clusters")
ax1.set_ylabel("CH score")
ax2.set_ylabel("DB score")
plt.title("How many clusters? Real-space top 70% data")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)
ax1.xaxis.set_major_locator(plticker.MultipleLocator(base=5.0))
ax1.xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
plt.show()




#%%tSNE plots for latent space clusters
colormap = ['k','blue','red','yellow','gray','purple','aqua','gold','orange']

for num_clusters in range(2,10):
    agglom = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    clustering = agglom.fit(scaled_top70_np)
    # plt.plot(clustering.labels_)
    # plt.scatter(df_beijing_filters.index,clustering.labels_,marker='.')
    
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(ae_input_val)
    plt.figure(figsize=(9,5))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                c=clustering.labels_,
                cmap=ListedColormap(colormap[0:num_clusters]))

    bounds = np.arange(0,num_clusters+1) -0.5
    ticks = np.arange(0,num_clusters)
    plt.colorbar(boundaries=bounds,ticks=ticks,label='Cluster')
    plt.title('tSNE real space top 70%, ' + str(num_clusters) + ' clusters')
    plt.show()






#%%Now do some clustering on the minmax data
#"Give equal importance to all features"
############################################################################################
#CLUSTERING AND FACTOR ANALYSIS OF MINMAX SCALED DATA
############################################################################################
#First need to make negative values zero??
minmax = MinMaxScaler()
minmax.fit(df_beijing_filters.to_numpy())
minmax_scaled_df = pd.DataFrame(minmax.transform(df_beijing_filters.to_numpy()),columns=df_beijing_filters.columns)
minmax_scaled_np = minmax_scaled_df.to_numpy()
#%%Real space dendrogram
#Lets try a dendrogram to work out the optimal number of clusters
fig,axes = plt.subplots(1,1,figsize=(20,10))
plt.title('MinMax scaled dendrogram')
dendrogram = sch.dendrogram(sch.linkage(minmax_scaled_np, method='ward'))
plt.show()


# %%#How many clusters should we have? MinMax scaled data
min_clusters = 2
max_clusters = 30

num_clusters_index = range(min_clusters,(max_clusters+1),1)
ch_score = np.empty(len(num_clusters_index))
db_score = np.empty(len(num_clusters_index))
silhouette_scores = np.empty(len(num_clusters_index))

for num_clusters in num_clusters_index:
    agglom_native = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    #agglom_native = KMeans(n_clusters = num_clusters)
    clustering = agglom_native.fit(ae_input_val)
    ch_score[num_clusters-min_clusters] = calinski_harabasz_score(minmax_scaled_np, clustering.labels_)
    db_score[num_clusters-min_clusters] = davies_bouldin_score(minmax_scaled_np, clustering.labels_)
    silhouette_scores[num_clusters-min_clusters] = silhouette_score(minmax_scaled_np, clustering.labels_)
fig,ax1 = plt.subplots(figsize=(10,6))
ax2=ax1.twinx()
ax1.plot(num_clusters_index,ch_score,label="CH score")
ax2.plot(num_clusters_index,db_score,c='red',label="DB score")
ax2.plot(num_clusters_index,silhouette_scores,c='black',label="Silhouette score")
ax1.set_xlabel("Num clusters")
ax1.set_ylabel("CH score")
ax2.set_ylabel("DB score")
plt.title("How many clusters? MinMax scaled data")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)
ax1.xaxis.set_major_locator(plticker.MultipleLocator(base=5.0))
ax1.xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
plt.show()




#%%tSNE plots for minmax scaled clusters
colormap = ['k','blue','red','yellow','gray','purple','aqua','gold','orange']

for num_clusters in range(2,10):
    agglom = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    clustering = agglom.fit(minmax_scaled_np)
    # plt.plot(clustering.labels_)
    # plt.scatter(df_beijing_filters.index,clustering.labels_,marker='.')
    
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(minmax_scaled_np)
    plt.figure(figsize=(9,5))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                c=clustering.labels_,
                cmap=ListedColormap(colormap[0:num_clusters]))

    bounds = np.arange(0,num_clusters+1) -0.5
    ticks = np.arange(0,num_clusters)
    plt.colorbar(boundaries=bounds,ticks=ticks,label='Cluster')
    plt.title('tSNE MinMax scaled data, ' + str(num_clusters) + ' clusters')
    plt.show()
    

# %%Comparing clustering labels
################################################
####COMPARING DIFFERENT CLUSTER LABELS######
################################################
clusters = []
arand_real_top70 = []
arand_real_minmax = []
arand_real_pca = []
arand_real_ae = []

for num_clusters in range(2,10):
    agglom = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    clustering_real = agglom.fit(ae_input_val)
    agglom = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    clustering_top70 = agglom.fit(scaled_top70_np)
    agglom = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    clustering_minmax = agglom.fit(minmax_scaled_np)
    agglom = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    clustering_pca = agglom.fit(PCA_space)
    agglom = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    clustering_ae = agglom.fit(latent_space)
    
    clusters.append(num_clusters)
    arand_real_top70.append(adjusted_rand_score(clustering_real.labels_, clustering_top70.labels_))
    arand_real_minmax.append(adjusted_rand_score(clustering_real.labels_, clustering_minmax.labels_))
    arand_real_pca.append(adjusted_rand_score(clustering_real.labels_, clustering_pca.labels_))
    arand_real_ae.append(adjusted_rand_score(clustering_real.labels_, clustering_ae.labels_))
    
fig,ax = plt.subplots(1,figsize=(7,5))
ax.plot(clusters,arand_real_top70,label='Real-space vs top 70%',c='b')
ax.plot(clusters,arand_real_minmax,label='Real-space vs MinMax data',c='k')
ax.plot(clusters,arand_real_pca,label='Real-space vs PCA-space',c='r')
ax.plot(clusters,arand_real_ae,label='Real-space vs AE latent space',c='gray')
ax.set_title('Adjusted Rand score (how similar are the cluster labels)')
ax.legend()
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Similarity')
plt.tight_layout()
plt.show()

#%%Plot all the clustering labels

#%%
#Relabel cluster labels so the most frequent label is 0, second most is 1 etc
def relabel(labels):
    most_frequent_order = np.flip(np.argsort(np.bincount(labels))[-(np.unique(labels).size):])
    #return most_frequent_order
    labels_out = labels
    for lab in range(len(most_frequent_order)):
        labels_out = np.where(labels == most_frequent_order[lab],lab,labels_out)
    return labels_out
#%%Plot for 5 clusters
num_clusters = 5
agglom = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
clustering_real = agglom.fit(ae_input_val)
agglom = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
clustering_top70 = agglom.fit(scaled_top70_np)
agglom = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
clustering_minmax = agglom.fit(minmax_scaled_np)
agglom = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
clustering_pca = agglom.fit(PCA_space)
agglom = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
clustering_ae = agglom.fit(latent_space)

fig,ax = plt.subplots(5,1,figsize=(6,10))
ax[0].plot(relabel(clustering_real.labels_))
ax[0].set_title('Real-space cluster labels')
ax[1].plot(relabel(clustering_top70.labels_))
ax[1].set_title('Top 70% of peaks cluster labels')
ax[2].plot(relabel(clustering_minmax.labels_))
ax[2].set_title('MinMax data cluster labels')
ax[3].plot(relabel(clustering_pca.labels_))
ax[3].set_title('PCA-space cluster labels')
ax[4].plot(relabel(clustering_ae.labels_))
ax[4].set_title('AE latent-space cluster labels')
plt.tight_layout()
plt.show()

#%%adjusted rand scores
arand_real_top70 = adjusted_rand_score(clustering_real.labels_, clustering_top70.labels_)
arand_real_minmax = adjusted_rand_score(clustering_real.labels_, clustering_minmax.labels_)
arand_real_pca = adjusted_rand_score(clustering_real.labels_, clustering_pca.labels_)
arand_real_ae = adjusted_rand_score(clustering_real.labels_, clustering_ae.labels_)

#%% Contingency matrices
cont_mtx_real_top70 = contingency_matrix(clustering_real.labels_, clustering_top70.labels_)



#%%Testing rpy2 Nbclust
import rpy2
import os
#os.environ['R_HOME'] = 'C:\\R\\R-4.1.2'
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1) # select the first mirror in the list
# R package names
packnames = ('ggplot2', 'NbClust','parameters','factoextra','mclust')

# R vector of strings
from rpy2.robjects.vectors import StrVector

# Selectively install what needs to be install.
# We are fancy, just because we can.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))


NbClust = rpackages.importr('NbClust')
R_parameters = rpackages.importr('parameters')


#%%

import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()


clust_R = NbClust.NbClust(PCA_space, distance = "euclidean", min_nc=2, max_nc=8, 
            method = "complete", index = "all")
clust_py = dict(zip(clust_R.names, map(list,list(clust_R))))
best_num = clust_py['Best.nc'][0]

plt.plot(range(2,9),clust_py['All.index'])
plt.show()
plt.plot(relabel(clust_py['Best.partition']))
plt.show()


#%%
#R code
# library(parameters)

# n_clust <- n_clusters(Eurojobs,
#                       package = c("easystats", "NbClust", "mclust"),
#                       standardize = FALSE)
# n_clust


from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
df_Eurojobs = pd.read_csv('C:\Work\Downloads\Eurojobs.csv').drop('Country',axis=1)


with localconverter(ro.default_converter + pandas2ri.converter):
  r_from_pd_df = ro.conversion.py2rpy(pd.DataFrame(PCA_space))

r_from_pd_df






rv_Eurojobs = robjects.FloatVector(df_Eurojobs.values.ravel.tolist())
a = robjects.ListVector(df_Eurojobs.values.ravel())

a = robjects.DataFrame(df_Eurojobs.values.tolist())

v = robjects.FloatVector([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
m = robjects.r['matrix'](v, nrow = 2)

Ro_Eurojobs = robjects.r['matrix'](df_Eurojobs.values.ravel(), nrow = df_Eurojobs.shape[0])





#PRoduces 23 things
n_clust_R = R_parameters.n_clusters(df_Eurojobs.to_numpy(),standardize=False,NbClust_method='Ward.D2', package=['NbClust','mclust'])

n_clust_R = R_parameters.n_clusters(PCA_space,standardize=False,NbClust_method='Ward.D2',package=['NbClust'])#,package=['easystats','NbClust','mclust'])



with localconverter(ro.default_converter + pandas2ri.converter):
  pd_from_r_df = ro.conversion.rpy2py(n_clust_R)


#%%Do fuzzy clustering on the 2-component PCA space

#%%
# Set up the loop and plot
fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
#alldata = np.vstack((xpts, ypts))
fpcs = []

for ncenters, ax in enumerate(axes1.reshape(-1), 2):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        PCA2_space.transpose(), ncenters, 2, error=0.005, maxiter=1000, init=None)

    # Store fpc values for later
    fpcs.append(fpc)

    # Plot assigned clusters, for each data point in training set
    cluster_membership = np.argmax(u, axis=0)
    for j in range(ncenters):
        ax.plot(PCA2_space[:,0][cluster_membership == j],
                PCA2_space[:,1][cluster_membership == j], '.', color=colors[j])

    # Mark the center of each fuzzy cluster
    for pt in cntr:
        ax.plot(pt[0], pt[1], 'rs')

    ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
    ax.axis('off')

fig1.tight_layout()

#%%Calculate fuzzy partition coefficient
fig2, ax2 = plt.subplots()
ax2.plot(np.r_[2:11], fpcs)
ax2.set_xlabel("Number of centers")
ax2.set_ylabel("Fuzzy partition coefficient")
plt.show()

##I think based on that you need 5 clusters
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    PCA2_space.transpose(), 5, 2, error=0.005, maxiter=1000, init=None)


df_cluster_prob = pd.DataFrame(u)
fuzzy_labels = relabel(df_cluster_prob.idxmax())
plt.plot(fuzzy_labels)


fig, ax = plt.subplots(figsize=(12,6))
ax.stackplot(np.arange(316),df_cluster_prob.iloc[0],df_cluster_prob.iloc[1],df_cluster_prob.iloc[2],df_cluster_prob.iloc[3],df_cluster_prob.iloc[4])



#%%
#One way of comparing the clusters would be to change the numbering- the biggest cluser is cluster 1
#Then rank them by their correlation with that one? Or something. I'm not clear how the numbering works

Another thing to do is to do the thing where you normalise every filter to 1
So you don't get dominated by the periods with high sigan which might be caused by meteorology'





