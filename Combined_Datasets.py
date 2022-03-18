# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:34:32 2022

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
from sklearn.decomposition import PCA, NMF
from sklearn import metrics

from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as plticker

import skfuzzy as fuzz

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pickle

import os
os.chdir('C:/Work/Python/Github/Orbitrap_clustering')
from ae_functions import *

# %%Load data

path='C:/Users/mbcx5jt5/Google Drive/Shared_York_Man2/'
df_beijing_raw, df_beijing_filters, df_beijing_metadata = beijing_load(
    path + 'BJ_UnAmbNeg9.1.1_20210505-Times_Fixed.xlsx',path + 'BJ_UnAmbNeg9.1.1_20210505-Times_Fixed.xlsx',
    peaks_sheetname="Compounds",metadata_sheetname="massloading_Beijing")

df_delhi_raw, df_delhi_filters, df_delhi_metadata = delhi_load2(path + '/Delhi/Orbitrap/')

joined = pd.concat([df_beijing_filters, df_delhi_filters], axis=0, join="inner")


#%%Test plots
import matplotlib.colors as colors
plt.pcolor(joined,
                   norm=colors.LogNorm(),
                    shading='auto')



df_beijing_winter = df_beijing_filters.iloc[0:124].copy()
df_beijing_summer = df_beijing_filters.iloc[124:].copy()

df_all_filters = df_beijing_filters.append(df_delhi_filters,sort=True).fillna(0)
df_all_raw = df_beijing_raw.transpose().append(df_delhi_raw.transpose(),sort=True).transpose()
