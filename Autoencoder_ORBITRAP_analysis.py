# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(1337) # for reproducibility
import tensorflow as tf
tf.random.set_seed(69)
import keras
#from google.colab import drive
import pandas as pd
import glob
import pdb
from sklearn.preprocessing import RobustScaler, StandardScaler,FunctionTransformer,MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
import scipy.cluster.hierarchy as sch
import keras
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


#1 Load in filter metadata
df_beijing_metadata = pd.read_excel(path + 'BJ_UnAmbNeg9.1.1_20210505-Times_Fixed.xlsx',engine='openpyxl',
                                   sheet_name='massloading_Beijing',usecols='A:K',nrows=329,converters={'mid_datetime': str})
#mid_datetime = pd.read_excel('/content/gdrive/MyDrive/Data_YRK_MAN/BJ_UnAmbNeg9.1.1_20210505.xlsx',engine='openpyxl',
                                 #  sheet_name='massloading_Beijing',usecols='E',nrows=329,dtype='str')
#df_filter_metadata["DateTime"] =pd.to_datetime(df_filter_metadata["mid_datetime"])
#df_filter_metadata.set_index('DateTime',inplace=True)
#df_filter_metadata.set_index('Sample.ID',inplace=True)
#2 set index to time
#3 Set that index to the time of the filter data
#4 Normalise filter data by sample volume
#5 AE_input does not contain any column other than the peaks
df_beijing_metadata['Sample.ID'] = df_beijing_metadata['Sample.ID'].astype(str)
#df_filter_metadata

df_beijing_peaks = pd.read_excel(path + 'BJ_UnAmbNeg9.1.1_20210505-Times_Fixed.xlsx',engine='openpyxl',sheet_name='Compounds')


#Load Delhi data and remove columns that are not needed
df_delhi_peaks = pd.read_excel(path + 'Delhi_Amb3.1_MZ.xlsx',engine='openpyxl')
df_delhi_peaks.drop(df_delhi_peaks.iloc[:,np.r_[0, 2:11, 14:18]],axis=1,inplace=True)

df_delhi_metadata = pd.read_excel(path + 'Delhi_massloading.xlsx',engine='openpyxl',
                                   sheet_name='Sheet1',usecols='B:L',skiprows=1,nrows=108, converters={'mid_datetime': str})
df_delhi_metadata.drop(labels="Filter ID.1",axis=1,inplace=True)
df_delhi_metadata.set_index("Filter ID",inplace=True)
df_delhi_metadata.drop(labels=["-","S25","S42","S51","S55","S68","S72"],axis=0,inplace=True)




beijing_peaks_examples = pd.DataFrame([df_beijing_peaks["Formula"],
                                      df_beijing_peaks["Formula"].apply(lambda x: filter_by_chemform(x))]).T



# %% Manually remove dedecanesulfonic acid as it's a huge backgrouns signal
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
df_beijing_peaks.drop(df_beijing_peaks[df_beijing_peaks["RT [min]"] > 20].index, inplace=True)
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

#Manually dropping this peak as it's the biggest peak in everything
#df_beijing_peaks.loc[249.152,19]
#C12 H26 O3 S  9.259132       1-Dodecanesulfonic acid

#aggregation_functions = {'price': 'sum', 'amount': 'sum', 'name': 'first'}
#df_new = df.groupby(df['id']).aggregate(aggregation_functions)
beijing_mz_index = [i[0] for i in df_beijing_peaks.index]
# %%#


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


df_beijing_filters['mid_datetime'] = pd.to_datetime(df_beijing_metadata['mid_datetime'],yearfirst=True)
df_beijing_filters.set_index('mid_datetime',inplace=True)

df_beijing_filters = df_beijing_filters.astype(float)

# %% Subtract blank sample data and then remove from analysis
df_beijing_filters = df_beijing_filters - df_beijing_filters.iloc[-1]
df_beijing_filters.drop(df_beijing_filters.tail(1).index,inplace=True)
# %%

#JUST THE WINTER DATA
#df_beijing_winter = df_beijing_filters.iloc[0:128].copy()

#JUST THE SUMMER DATA
#df_beijing_summer = df_beijing_filters.iloc[128:].copy()


#Some basic checks like the time series of total concentration
beijing_rows_sum = df_beijing_summer.sum(axis=1)
fig, ax = plt.subplots()
ax.plot(df_beijing_summer.index,beijing_rows_sum)
plt.title("Total sum of all peaks")
fig.autofmt_xdate()
plt.show()


#How many data points?
num_filters = int(df_beijing_summer.shape[0])
num_cols = int(df_beijing_summer.shape[1])


fig = plt.plot(figsize=(30,20))
plt.pcolormesh(beijing_mz_index,df_beijing_summer.index,df_beijing_summer.astype(float))
plt.title("Beijing summer filters data, linear scale")
plt.xlabel("m/z")
plt.show()

fig = plt.plot(figsize=(30,20))
plt.pcolormesh(beijing_mz_index,df_beijing_summer.index,df_beijing_summer.astype(float),norm=matplotlib.colors.LogNorm())
plt.title("Beijing summer data, log scale")
plt.xlabel("m/z")
plt.show()

#
print("The index/column of max data point is " + str(df_beijing_summer.stack().idxmax()))

# %%
#What are the 10 biggest peaks in the whole thing?
beijing_summer_avg = df_beijing_summer.sum()


# %%
#Load the scaling pipeline that was used to train the autoencoder

pipe = load(r'C:\Work\Python\Github\Orbitrap_clustering\Models\ae_pipe\pipe.joblib')



df_beijing_summer_scaled = pd.DataFrame(pipe.transform(df_beijing_summer))
fig = plt.plot(figsize=(30,20))
plt.pcolormesh(beijing_mz_index,df_beijing_summer.index,df_beijing_summer_scaled.astype(float))
plt.title("Beijing summer data, logged and MinMaxScaled")
plt.xlabel("m/z")
plt.show()








# %%
#Scale the data for input into AE

ae_input=df_beijing_summer_scaled.to_numpy()

#Some basic checks like the time series of total concentration
scaled_df_sum = df_beijing_summer_scaled.sum(axis=1)
fig,ax = plt.subplots()
ax = plt.plot(df_beijing_summer.index,scaled_df_sum)
plt.title("Total sum of all scaled peaks")
fig.autofmt_xdate()
plt.show()


# %%
#load the autoencoder models

encoder_ae = tf.keras.models.load_model(r'C:\Work\Python\Github\Orbitrap_clustering\Models\encoder_ae')
decoder_ae = tf.keras.models.load_model(r'C:\Work\Python\Github\Orbitrap_clustering\Models\decoder_ae')
#Loading ae doesn't work for some reason
ae = tf.keras.models.load_model(r'C:\Work\Python\Github\Orbitrap_clustering\Models\ae')

# %%
#Encode the beijing summer data using the autoencoder
latent_space = encoder_ae.predict(ae_input)
df_latent_space = pd.DataFrame(latent_space)
df_latent_space = df_latent_space.set_index(df_beijing_summer.index)
latent_space_sum = latent_space.sum(axis=1)





# %%

#How good is our encode/decode?
#This is the really key plot. It's shit!! So need to improve it I think, otherwise it's not useful for getting the factors out of the latent space
#And if you don't know what the factors out, then you can't use them
#plt.scatter(ae_input_val,ae.predict(ae_input_val))
#plt.scatter(df_beijing_summer.values,pipe.inverse_transform(ae.predict(ae_input)))
plt.scatter(df_beijing_summer.values,pipe.inverse_transform(decoder_ae.predict(encoder_ae.predict(ae_input))))
plt.xlabel("Input data")
plt.ylabel("Reconstructed data")
plt.title("Point-by-point autoencoder performance")
plt.show()

# %%
#Is the sum total linear with input vs output?
#plt.scatter(df_beijing_summer.sum(axis=1),pipe.inverse_transform(ae.predict(ae_input)).sum(axis=1),c=df_beijing_summer.index)
plt.scatter(df_beijing_summer.sum(axis=1),pipe.inverse_transform(decoder_ae.predict(encoder_ae.predict(ae_input))).sum(axis=1),c=df_beijing_summer.index)
plt.xlabel("Beijing summer AE input")
plt.ylabel("Beijing summer AE output")

# %%
#Let's make a time series of that
reconstruction_sum_error = pipe.inverse_transform(decoder_ae.predict(encoder_ae.predict(ae_input))).sum(axis=1) / df_beijing_summer.sum(axis=1)
plt.plot(reconstruction_sum_error)

# %%
#What does the latent space look like?
plt.pcolormesh(df_latent_space)
plt.title("Beijing summer AE Latent space")
plt.show()


# %%
# #Principe component analysis
# pca = PCA(n_components = 5)
# prin_comp = pca.fit_transform(latent_space)
# plt.pcolormesh(prin_comp)
# plt.title


# %%
############################################################################################
#CLUSTERING AND FACTOR ANALYSIS OF REAL SPACE DATA
############################################################################################
#Lets try a dendrogram to work out the optimal number of clusters

fig,axes = plt.subplots(1,1,figsize=(8,6))
plt.title("dendrogram")
dendrogram = sch.dendrogram(sch.linkage(df_beijing_summer, method='ward'))
plt.title("Dendrogram of unencoded Beijing summer data")
plt.show()



# %%
#How many clusters should we have?
from sklearn.metrics import calinski_harabasz_score
min_clusters = 2
max_clusters = 10

num_clusters_index = range(min_clusters,(max_clusters+1),1)
ch_score = np.empty(len(num_clusters_index))
min_cluster_filters = np.empty(len(num_clusters_index))

for num_clusters in num_clusters_index:
    agglom_native = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    #agglom_native = KMeans(n_clusters = num_clusters)
    clustering = agglom_native.fit(df_beijing_summer.values)
    ch_score[num_clusters-2] = calinski_harabasz_score(df_beijing_summer.values, clustering.labels_)
    cluster_bins = range(0,num_clusters+1)
    min_cluster_filters[num_clusters-2] = np.histogram(clustering.labels_,bins=cluster_bins)[0].min()
    
fig,ax1 = plt.subplots()
ax1.plot(num_clusters_index,ch_score,label="CH score")
ax1.set_xlabel("Num clusters")
ax1.set_ylabel("Calinski-Harabasz score")
ax2 = ax1.twinx()
ax2.plot(num_clusters_index,min_cluster_filters,c='red',label="Minimum filters per cluster")
ax2.set_ylabel("Minimum filters per cluster")

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)


#plt.scatter(df_onerow_test.values,df_aug.values)
# %%
# Now we can perform agglomerative cluster analysis on this new ouput, assigning each row to a particular cluster. We have to define the number of clusters but thats ok for now

agglom_native = AgglomerativeClustering(n_clusters = 3, linkage = 'ward')
#clustering = agglom.fit(latent_space)
clustering = agglom_native.fit(df_beijing_summer.values)
fig, ax = plt.subplots()
ax.plot(df_beijing_summer.index,clustering.labels_)
plt.title("Agglom clustering labels, unencoded data")
fig.autofmt_xdate()
plt.show()

#And what are the clusters?
#The cluster labels can just be moved straight out the latent space
cluster0_uncoded = df_beijing_summer[clustering.labels_==0].mean()
cluster1_uncoded = df_beijing_summer[clustering.labels_==1].mean()
cluster2_uncoded = df_beijing_summer[clustering.labels_==2].mean()


# %%
############################################################################################
#CLUSTERING AND FACTOR ANALYSIS OF SCALEDDATA
############################################################################################
#fig,axes = plt.subplots(1,1,figsize=(20,10))

# %%
fig = plt.plot(figsize=(30,20))
#plt.pcolormesh(df_beijing_peaks.iloc[:,list(range(4,321))],norm=matplotlib.colors.LogNorm())
plt.pcolormesh(ae_input)
plt.title("Filter data, MinMaxScaled")
plt.show()


plt.title("dendrogram")
dendrogram = sch.dendrogram(sch.linkage(ae_input, method='ward'))
plt.title("Scaled space dendrogram")
plt.show()





# %%
# Now we can perform agglomerative cluster analysis on this new ouput, assigning each row to a particular cluster. We have to define the number of clusters but thats ok for now
n_clusters = 5
agglom = AgglomerativeClustering(n_clusters = n_clusters, linkage = 'ward')
#clustering = agglom.fit(latent_space)
clustering = agglom.fit(ae_input)
fig, ax1 = plt.subplots()
ax1.plot(df_beijing_summer.index,clustering.labels_,marker=".")
plt.title("Scaled space cluster labels, " + str(n_clusters) + " clusters")
fig.autofmt_xdate()
plt.plot()
#plt.plot(df_beijing_filters.index,clustering.labels_)

# %%
############################################################################################
#CLUSTERING AND FACTOR ANALYSIS OF LATENT SPACE DATA
############################################################################################

#Lets try a dendrogram to work out the optimal number of clusters


fig,axes = plt.subplots(figsize=(20,10))
dendrogram = sch.dendrogram(sch.linkage(latent_space, method='ward'))
plt.title("Latent space dendrogram of Beijing summer data")
plt.show()

# %%
min_clusters = 2
max_clusters = 10

num_clusters_index = range(min_clusters,(max_clusters+1),1)
ch_score = np.empty(len(num_clusters_index))
min_cluster_filters = np.empty(len(num_clusters_index))

for num_clusters in num_clusters_index:
    agglom_native = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
    #agglom_native = KMeans(n_clusters = num_clusters)
    clustering = agglom_native.fit(latent_space)
    ch_score[num_clusters-2] = calinski_harabasz_score(latent_space, clustering.labels_)
    cluster_bins = range(0,num_clusters+1)
    min_cluster_filters[num_clusters-2] = np.histogram(clustering.labels_,bins=cluster_bins)[0].min()
    
fig,ax1 = plt.subplots()
ax1.plot(num_clusters_index,ch_score)
ax1.set_xlabel("Num clusters")
ax1.set_ylabel("CH score")
#fig.set_title("Calinski-Harabasz score for encoded Beijing summer")

#ax2 = ax1.twinx()
#ax2.plot(num_clusters_index,min_cluster_filters)
#ax2.set_ylabel("Minimum filters per cluster")

# %%
# Now we can perform agglomerative cluster analysis on this new ouput, assigning each row to a particular cluster. We have to define the number of clusters but thats ok for now
n_clusters = 4
agglom = AgglomerativeClustering(n_clusters = n_clusters, linkage = 'ward')
#clustering = agglom.fit(latent_space)
clustering = agglom.fit(latent_space)
fig,ax = plt.subplots()
plt.plot(df_beijing_summer.index,clustering.labels_,marker=".")
plt.title("Latent space cluster labels, " + str(n_clusters) + " clusters")
fig.autofmt_xdate()
plt.show()
#plt.plot(df_beijing_filters.index,clustering.labels_)
# %%
#And what are the clusters?
#The cluster labels can just be moved straight out the latent space
cluster0_decoded = df_beijing_summer[clustering.labels_==0].mean()
cluster1_decoded = df_beijing_summer[clustering.labels_==1].mean()
cluster2_decoded = df_beijing_summer[clustering.labels_==2].mean()
cluster3_decoded = df_beijing_summer[clustering.labels_==3].mean()

#Latent space clusters
cluster0_lat = df_latent_space[clustering.labels_==0].mean()
cluster1_lat = df_latent_space[clustering.labels_==1].mean()
cluster2_lat = df_latent_space[clustering.labels_==2].mean()


# %%
#Convert chemical formula into a molecule name

#This version should be good but isn't great because things don't always come out at the right RT or something
chemform_namelist = pd.read_excel(path + 'Beijing_Amb3.1_MZ_noblank.xlsx',engine='openpyxl')[["m/z","RT [min]","Name"]]
RT_round =  chemform_namelist["RT [min]"].apply(lambda x: round_odd(x))
mz_round = chemform_namelist["m/z"].apply(lambda x: round(x, 3))
chemform_namelist = chemform_namelist.groupby([mz_round,RT_round]).aggregate("first")
chemform_namelist = chemform_namelist["Name"]

#This one does it just off the chemical formula
chemform_namelist = pd.read_excel(path + 'Beijing_Amb3.1_MZ_noblank.xlsx',engine='openpyxl')[["Formula","Name"]]
chemform_namelist.fillna('-', inplace=True)
chemform_namelist = chemform_namelist.groupby(chemform_namelist["Formula"]).agg(pd.Series.mode)

#chemform_namelist.set_index(chemform_namelist["Formula"])




# %%
#Some tests on 249.152 19 C12 H26 O3 S 15.48259 1-Dodecanesulfonic acid
df_dodec = pd.DataFrame([df_beijing_filters.loc[:][249.152,19],df_beijing_filters.loc[:][249.153,19]]).transpose()
#df_dodec.plot()
#dodec_153 = df_beijing_filters.loc[:][249.153,19]

fig,ax = plt.subplots()
ax1 = plt.plot(df_dodec - df_dodec.iloc[315])
fig.autofmt_xdate()
plt.show()

# %%
#What are the 10 biggest molecules in the background filter?
THIS IS WHWAT I WAS WORKING ON
IM NOT SURE ITS ALIGNED JUST RIGHT, NEED TO DOUBLE CHECK BECAUSE THE FILTERS DF
IS 1 COLUMN LESS THAN THE PEAKS DF


background_top10 = cluster_extract_peaks(df_beijing_filters.iloc[-1], df_beijing_peaks,10)
ambient_top10 = cluster_extract_peaks(df_beijing_filters.iloc[0:315].mean(), df_beijing_peaks,10)
    
    




# %%
#Function to extract the top n peaks from a cluster in terms of their chemical formula
def cluster_extract_peaks(cluster, df_peaks,num_peaks,printdf=False):
    #Check they are the same length
    if(cluster.shape[0] != df_peaks.shape[0]):
        print("cluster_extract_peaks returning null: cluster and peaks dataframe must have same number of peaks")
        return np.NaN
        quit()
    
    #First get top n peaks from the cluster
    #cluster = pd.Series(cluster)
    nlargest = cluster.nlargest(num_peaks)
    nlargest_pct = nlargest / cluster.sum() * 100
    
    output_df = pd.DataFrame(df_beijing_peaks["Formula"][nlargest.index])
    output_df["peak_pct"] = nlargest_pct
    
    # #Get the chemical formula namelist
    # try:
    #     chemform_namelist
    # except NameError:
    #     #global chemform_namelist = pd.DataFrame()
    #     chemform_namelist = pd.read_excel(path + 'Beijing_Amb3.1_MZ_noblank.xlsx',engine='openpyxl')[["Formula","Name"]]
    #     chemform_namelist.set_index(chemform_namelist["Formula"],inplace=True)
    
    #output_df["Name"] = chemform_namelist[output_df.index]
    output_df["Name"] = chemform_namelist.loc[output_df["Formula"]].values
    
    if(printdf == True):
        print(output_df)
        
    return output_df


# %%
def round_time(dt=None, date_delta=datetime.timedelta(minutes=1), to='average'):
    """
    Round a datetime object to a multiple of a timedelta
    dt : datetime.datetime object, default now.
    dateDelta : timedelta object, we round to a multiple of this, default 1 minute.
    from:  http://stackoverflow.com/questions/3463930/how-to-round-the-minute-of-a-datetime-object-python
    """
    round_to = date_delta.total_seconds()
    if dt is None:
        dt = datetime.now()
    seconds = (dt - dt.min).seconds

    if seconds % round_to == 0 and dt.microsecond == 0:
        rounding = (seconds + round_to / 2) // round_to * round_to
    else:
        if to == 'up':
            # // is a floor division, not a comment on following line (like in javascript):
            rounding = (seconds + dt.microsecond/1000000 + round_to) // round_to * round_to
        elif to == 'down':
            rounding = seconds // round_to * round_to
        else:
            rounding = (seconds + round_to / 2) // round_to * round_to

    return dt + datetime.timedelta(0, rounding - seconds, - dt.microsecond)


def roundTime(dt=None, dateDelta=datetime.timedelta(minutes=1)):
    """Round a datetime object to a multiple of a timedelta
    dt : datetime.datetime object, default now.
    dateDelta : timedelta object, we round to a multiple of this, default 1 minute.
    Author: Thierry Husson 2012 - Use it as you want but don't blame me.
            Stijn Nevens 2014 - Changed to use only datetime objects as variables
    """
    roundTo = dateDelta.total_seconds()

    if dt == None : dt = datetime.datetime.now()
    seconds = (dt - dt.min).seconds
    # // is a floor division, not a comment on following line:
    rounding = (seconds+roundTo/2) // roundTo * roundTo
    return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)

# %%
#Lets get some stats on air quality data for the different clusters
df_aq_filtime = pd.read_csv(path+'aphh_summer_filter_aggregate_merge.csv')
df_aq_filtime["DateTime"] =pd.to_datetime(df_aq_filtime["date_mid"])
df_aq_filtime.set_index('DateTime',inplace=True)

#merge aq to the time of the filters
df_aq_filtime = pd.merge_asof(df_beijing_summer.iloc[:,0].sort_index(),df_aq_filtime.sort_index(),left_index=True,right_index=True,direction="nearest",tolerance=pd.Timedelta('5 minute'))


df_aq_filtime["o3_ppbv"][clustering.labels_==0].mean()


# %%
import scipy

def cluster_stats(df,clustering,cluster_num):
    print("Time hours " + str(scipy.stats.circmean(df.index.hour.values[clustering.labels_==cluster_num], high=24,nan_policy='omit').round(1)) + "+/- " + str(scipy.stats.circstd(df.index.hour.values[clustering.labels_==cluster_num], high=24,nan_policy='omit').round(1)))
    print("O3 " + str(df["o3_ppbv"][clustering.labels_==cluster_num].mean().round(0)) + "+/- " + str(df["o3_ppbv"][clustering.labels_==cluster_num].std().round(0)))
    print("CO " + str(df["co_ppbv"][clustering.labels_==cluster_num].mean().round(-1)) + "+/- " + str(df["co_ppbv"][clustering.labels_==cluster_num].std().round(-1)))
    print("NO2 " + str(df["no2_ppbv"][clustering.labels_==cluster_num].mean().round(0)) + "+/- " + str(df["no2_ppbv"][clustering.labels_==cluster_num].std().round(0)))
    print("RH " + str(df["rh_8m"][clustering.labels_==cluster_num].mean().round(0)) + "+/- " + str(df["rh_8m"][clustering.labels_==cluster_num].std().round(0)))   
    print("Org AMS " + str(df["Org_ams"][clustering.labels_==cluster_num].mean().round(0)) + "+/- " + str(df["Org_ams"][clustering.labels_==cluster_num].std().round(0)))
    print("NO3 AMS " + str(df["NO3_ams"][clustering.labels_==cluster_num].mean().round(0)) + "+/- " + str(df["NO3_ams"][clustering.labels_==cluster_num].std().round(0)))
    print("SO4 AMS " + str(df["SO4_ams"][clustering.labels_==cluster_num].mean().round(0)) + "+/- " + str(df["SO4_ams"][clustering.labels_==cluster_num].std().round(0)))

    print("OOA1 AMS " + str(df["OOA1_ams"][clustering.labels_==cluster_num].mean().round(1)) + "+/- " + str(df["OOA1_ams"][clustering.labels_==cluster_num].std().round(1)))
    print("OOA2 AMS " + str(df["OOA2_ams"][clustering.labels_==cluster_num].mean().round(1)) + "+/- " + str(df["OOA2_ams"][clustering.labels_==cluster_num].std().round(1)))
    print("OOA3 AMS " + str(df["OOA3_ams"][clustering.labels_==cluster_num].mean().round(1)) + "+/- " + str(df["OOA3_ams"][clustering.labels_==cluster_num].std().round(1)))
    print("HOA AMS " + str(df["HOA_ams"][clustering.labels_==cluster_num].mean().round(1)) + "+/- " + str(df["HOA_ams"][clustering.labels_==cluster_num].std().round(1)))
    print("COA AMS " + str(df["COA_ams"][clustering.labels_==cluster_num].mean().round(1)) + "+/- " + str(df["COA_ams"][clustering.labels_==cluster_num].std().round(1)))
    
    print("Wind dir " + str(scipy.stats.circmean(df["wd_8m"][clustering.labels_==cluster_num], high=360,nan_policy='omit').round(0)) + "+/- " + str(scipy.stats.circstd(df["wd_8m"][clustering.labels_==cluster_num], high=24,nan_policy='omit').round(0)))
    print("Wind speed " + str(df["ws_8m"][clustering.labels_==cluster_num].mean().round(1)) + "+/- " + str(df["ws_8m"][clustering.labels_==cluster_num].std().round(1)))
    
    
# %%

#Now lets decode the clusters from latent space and see if it still works
#Cluster0_decod = transformer.inverse_transform(decoder_ae.predict(np.expand_dims(cluster0_lat, axis=0)))
#Cluster1_decod = transformer.inverse_transform(decoder_ae.predict(np.expand_dims(cluster1_lat, axis=0)))
#Cluster2_decod = transformer.inverse_transform(decoder_ae.predict(np.expand_dims(cluster2_lat, axis=0)))

Cluster0_decod = pipe.inverse_transform((decoder_ae.predict(np.expand_dims(cluster0_lat, axis=0))))
Cluster1_decod = pipe.inverse_transform((decoder_ae.predict(np.expand_dims(cluster1_lat, axis=0))))
Cluster2_decod = pipe.inverse_transform((decoder_ae.predict(np.expand_dims(cluster2_lat, axis=0))))

#THESE SHOULD ALWAYS ALWAYS BE A STRAIGHT LINE OTHERWISE SOMEHTING IT NOT WORKING RIGHT IN THE AE OR TRANSFORMER
plt.scatter(cluster0_decoded,Cluster0_decod)

plt.scatter(cluster1_decoded,Cluster1_decod)

plt.scatter(cluster2_decoded,Cluster2_decod)

# %%
#Now do some NMF clustering
from sklearn.decomposition import NMF
num_nmf_factors = 5
nmf_model = NMF(n_components=num_nmf_factors)
#model.fit(latent_space)

#Get the different components
#nmf_features = model.transform(latent_space)
#print(model.components_)

W = nmf_model.fit_transform(latent_space)
#W = model.fit_transform(df_3clust.values)
H = nmf_model.components_

# %%
prefix = "Beijing_summer_"
factor_sum_filename = prefix + str(range)

df_latent_factors_mtx = pd.DataFrame()
df_factor_profiles = pd.DataFrame(columns=df_beijing_filters.columns)
df_factorsum_tseries = pd.DataFrame()

#Extract and save all the components
#Factor names like for 5 factors, it would go factor5_0, factor5_1...factor5_4
for factor in range(num_nmf_factors):
    factor_name = ("factor_"+str(num_nmf_factors))+"_"+str(factor)
    Factor_lat = H[factor]
    Factor_lat_mtx = np.outer(W.T[factor], H[factor])
    
    #Factor profile
    Factor_decod = pipe.inverse_transform(decoder_ae.predict(np.expand_dims(Factor_lat, axis=0)))

    #Time series of factor as a matrix
    Factor_mtx_decod = pipe.inverse_transform(decoder_ae.predict(Factor_lat_mtx))
    Factor_decod_sum = Factor_mtx_decod.sum(axis=1)
    
    df_latent_factors_mtx = df_latent_factors_mtx.append(pd.DataFrame(Factor_lat_mtx),index=[factor_name])
    df_factor_profiles = df_factor_profiles.append(pd.DataFrame(Factor_decod,columns=df_beijing_filters.columns,index=[factor_name]))
    #factor_sums_tseries.append(Factor_decod_sum,axis=1)
    df_factorsum_tseries[factor_name] = Factor_decod_sum
    
df_factorsum_tseries.index = df_beijing_filters.index


#What is the total residual error in the latent data?

#What is the residual error in the real data?

nmf_total_sum =  df_factorsum_tseries.sum(axis=1)
nmf_residual = beijing_rows_sum - nmf_total_sum
nmf_residual_pct = (nmf_residual / beijing_rows_sum)*100

plt.plot(df_factorsum_tseries)
    
# %%
#1 What is the time series of the 2 factors? Need each factor as a t series
Factor0_lat = H[0]
Factor1_lat = H[1]
Factor2_lat = H[2]
Factor3_lat = H[3]
Factor4_lat = H[4]


Factor0_lat_mtx = np.outer(W.T[0], H[0])
Factor1_lat_mtx = np.outer(W.T[1], H[1])
Factor2_lat_mtx = np.outer(W.T[2], H[2])
Factor3_lat_mtx = np.outer(W.T[3], H[3])
Factor4_lat_mtx = np.outer(W.T[4], H[4])
#Now need to decode these matrices to get the time series matrix of each factor

Factor0_decod = pipe.inverse_transform(decoder_ae.predict(np.expand_dims(Factor0_lat, axis=0)))
Factor1_decod = pipe.inverse_transform(decoder_ae.predict(np.expand_dims(Factor1_lat, axis=0)))
Factor2_decod = pipe.inverse_transform(decoder_ae.predict(np.expand_dims(Factor2_lat, axis=0)))
Factor3_decod = pipe.inverse_transform(decoder_ae.predict(np.expand_dims(Factor3_lat, axis=0)))
Factor4_decod = pipe.inverse_transform(decoder_ae.predict(np.expand_dims(Factor4_lat, axis=0)))

Factor0_mtx_decod = pipe.inverse_transform(decoder_ae.predict(Factor0_lat_mtx))
Factor1_mtx_decod = pipe.inverse_transform(decoder_ae.predict(Factor1_lat_mtx))
Factor2_mtx_decod = pipe.inverse_transform(decoder_ae.predict(Factor2_lat_mtx))
Factor3_mtx_decod = pipe.inverse_transform(decoder_ae.predict(Factor3_lat_mtx))
Factor4_mtx_decod = pipe.inverse_transform(decoder_ae.predict(Factor4_lat_mtx))

Factor0_decod_sum = Factor0_mtx_decod.sum(axis=1)
Factor1_decod_sum = Factor1_mtx_decod.sum(axis=1)
Factor2_decod_sum = Factor2_mtx_decod.sum(axis=1)
Factor3_decod_sum = Factor3_mtx_decod.sum(axis=1)
Factor4_decod_sum = Factor4_mtx_decod.sum(axis=1)

plt.plot(Factor0_decod_sum)
plt.plot(Factor1_decod_sum)
plt.plot(Factor2_decod_sum)
plt.plot(Factor3_decod_sum)
plt.plot(Factor4_decod_sum)
plt.ylim(bottom=0)
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
plt.scatter(Cluster_A,Factor1_decod)
#plt.scatter(min_filter,Factor1_decod)
plt.xlabel("Input peak height")
plt.ylabel("Output peak height")

#This one does not either
plt.scatter(Cluster_B,Factor0_decod)
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
