# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:30:09 2021

@author: mbcx5jt5
"""

#from google.colab import drive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from datetime import date
import calendar


# %%
#Colab
#path='/content/gdrive/My Drive/Shared_York_Man2/'
#Laptop
path='C:/Users/mbcx5jt5/Google Drive/Shared_York_Man2/'


#Load the met data
df_merge_1min = pd.read_excel(path+'all api and met data 1 min.xls')
df_merge_1min["DateTime"] =pd.to_datetime(df_merge_1min["TheTime"])
df_merge_1min.set_index('DateTime',inplace=True)
df_merge_filtime = pd.read_csv(path+'aphh_summer_filter_aggregate_merge.csv')
df_merge_filtime["DateTime"] =pd.to_datetime(df_merge_filtime["date_mid"])
df_merge_filtime.set_index('DateTime',inplace=True)
df_merge_filtime['date_start'] = pd.to_datetime(df_merge_filtime['date_start'])
df_merge_filtime['date_end'] = pd.to_datetime(df_merge_filtime['date_end'])
df_merge_filtime['sample_time_h'] = (df_merge_filtime['date_end'] - df_merge_filtime['date_start']) / np.timedelta64(1, 'h')

#%%Gantt chart of filters
sample_num = pd.Series(range(1, df_merge_filtime.shape[0]+1))
sample_num.index = df_merge_filtime.index
df_merge_filtime['sample_num'] = sample_num
fig, ax = plt.subplots(1, figsize=(16,6))
ax.barh(sample_num, df_merge_filtime['sample_time_h'], left=df_merge_filtime['date_start'])
plt.show()


# %%
df_merge_1min["NOx_age"] = -np.log(df_merge_1min['NOx / ppbv'] / df_merge_1min['Noy / ppbv'])

#Photochemical age. Original calculation from Parrish (1992)
k_toluene = 5.63e-12
k_benzene = 1.22e-12
OH_conc = 1.5e6

#This this is less noisy but has one big spike in it
df_merge_filtime["toluene_over_benzene_syft"]=  df_merge_filtime["toluene_ppb_syft"] / df_merge_filtime["benzene._ppb_syft"]
df_merge_filtime["toluene_over_benzene_syft"].values[df_merge_filtime["toluene_over_benzene_syft"] > 5] = np.nan#Remove one big spike

#This seems quite noisy
#toluene_over_benzene_ptrtof = df_merge_filtime["toluene_ppb_ptrtof"] / df_merge_filtime["benzene_ppb_ptrtof"]

#benzene/tolluene emission ratio
benzene_tolluene_ER = df_merge_filtime["toluene_over_benzene_syft"].quantile(0.99)

df_merge_filtime["Photochem_age_h"] = 1/(3600*OH_conc*(k_toluene-k_benzene))  * (np.log(benzene_tolluene_ER) - np.log(df_merge_filtime["toluene_over_benzene_syft"]))
# %%
#Find the data points that are from the 5th, 50th and 95th percentile of photochemical age
photochem_q05_time = df_merge_filtime[df_merge_filtime["Photochem_age_h"]== df_merge_filtime["Photochem_age_h"].quantile(0.05,interpolation='nearest')].index[0]
photochem_q50_time = df_merge_filtime[df_merge_filtime["Photochem_age_h"]== df_merge_filtime["Photochem_age_h"].quantile(0.50,interpolation='nearest')].index[0]
photochem_q95_time = df_merge_filtime[df_merge_filtime["Photochem_age_h"]== df_merge_filtime["Photochem_age_h"].quantile(0.95,interpolation='nearest')].index[0]

# %%
df_merge_filtime["nox_over_noy"] = df_merge_filtime["nox_ppbv"] / df_merge_filtime["noy_ppbv"]

# %%
#Basic gas phase plot from the 1min data
#Time series of O3, NOx, CO
#And some diurnals

fig, ax = plt.subplots(3,1,figsize=(25,15),sharex=True)
fig.suptitle('1-min gas phase data')

tmin = df_merge_1min.index.min()
tmax = df_merge_1min.index.max()

# bigger plot elements suitable for giving talks
sns.set_context("talk")
sns.lineplot(x="TheTime",y="CO / ppbv", data=df_merge_1min, ci=None,ax=ax[0])
sns.lineplot(x="TheTime",y="NOx / ppbv", data=df_merge_1min, ci=None,ax=ax[1])
sns.lineplot(x="TheTime",y="O3 / ppbv", data=df_merge_1min, ci=None,ax=ax[2])




# axis labels
plt.xlabel("", size=14)
#ax[0].ylabel("CO (ppbv)", size=14)
#ax[1].ylabel("NOx (ppbv)", size=14)
#ax[2].ylabel("O3 (ppbv)", size=14)
# save image as PNG file
#plt.savefig("Time_Series_Plot_with_Seaborn.png",
#                   format='png',
#                                      dpi=150)


# %%
#Basic gas phase plot from the 1min data
#Time series of O3, NOx, CO
#And some diurnals

fig, ax = plt.subplots(3,1,figsize=(25,15),sharex=True)
fig.suptitle('Filter-time gas phase data')

tmin = df_merge_1min.index.min()
tmax = df_merge_1min.index.max()

# bigger plot elements suitable for giving talks
sns.set_context("talk")
sns.lineplot(x="DateTime",y="co_ppbv", data=df_merge_filtime, ci=None,ax=ax[0])
sns.lineplot(x="DateTime",y="nox_ppbv", data=df_merge_filtime, ci=None,ax=ax[1])
sns.lineplot(x="DateTime",y="o3_ppbv", data=df_merge_filtime, ci=None,ax=ax[2])
# axis labels
plt.xlabel("", size=14)

# %%
#Basic gas phase plot from the 1min data
#Time series of O3, NOx, CO
#And some diurnals

fig, ax = plt.subplots(4,1,figsize=(25,15),sharex=True)
fig.suptitle('Filter-time AMS data')

tmin = df_merge_1min.index.min()
tmax = df_merge_1min.index.max()

# bigger plot elements suitable for giving talks
sns.set_context("talk")
sns.lineplot(x="DateTime",y="Org_ams", data=df_merge_filtime, ci=None,ax=ax[0])
sns.lineplot(x="DateTime",y="NO3_ams", data=df_merge_filtime, ci=None,ax=ax[1])
sns.lineplot(x="DateTime",y="SO4_ams", data=df_merge_filtime, ci=None,ax=ax[2])
sns.lineplot(x="DateTime",y="NH4_ams", data=df_merge_filtime, ci=None,ax=ax[3])
# axis labels
plt.xlabel("", size=14)

# %%
#Basic gas phase plot from the 1min data
#Time series of O3, NOx, CO
#And some diurnals

fig, ax = plt.subplots(5,1,figsize=(25,15),sharex=True)
fig.suptitle('Filter-time AMS PMF data')

tmin = df_merge_1min.index.min()
tmax = df_merge_1min.index.max()

# bigger plot elements suitable for giving talks
sns.set_context("talk")
sns.lineplot(x="DateTime",y="HOA_ams", data=df_merge_filtime, ci=None,ax=ax[0])
sns.lineplot(x="DateTime",y="OOA1_ams", data=df_merge_filtime, ci=None,ax=ax[1])
sns.lineplot(x="DateTime",y="OOA2_ams", data=df_merge_filtime, ci=None,ax=ax[2])
sns.lineplot(x="DateTime",y="OOA3_ams", data=df_merge_filtime, ci=None,ax=ax[3])
sns.lineplot(x="DateTime",y="COA_ams", data=df_merge_filtime, ci=None,ax=ax[4])
# axis labels
plt.xlabel("", size=14)


# %%
#Basic gas phase plot from the 1min data
#Time series of O3, NOx, CO
#And some diurnals

fig, ax = plt.subplots(4,1,figsize=(25,15),sharex=True)
fig.suptitle('Photochemical age data')

tmin = df_merge_1min.index.min()
tmax = df_merge_1min.index.max()

# bigger plot elements suitable for giving talks
sns.set_context("talk")
sns.lineplot(x="DateTime",y="nox_over_noy", data=df_merge_filtime, ci=None,ax=ax[0])
sns.lineplot(x="DateTime",y="Photochem_age_h", data=df_merge_filtime, ci=None,ax=ax[1])
#sns.lineplot(x="DateTime",y="SO4_ams", data=df_merge_filtime, ci=None,ax=ax[2])
#sns.lineplot(x="DateTime",y="NH4_ams", data=df_merge_filtime, ci=None,ax=ax[3])
# axis labels
plt.xlabel("", size=14)

# %%
#Weekday trend
fig, ax = plt.subplots(3,1,figsize=(15,10),sharex=True)
fig.suptitle('gas phase weekday')
sns.boxplot(data=df_merge_1min, x=df_merge_1min.index.day_name(), y="CO / ppbv",order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],ax=ax[0])
sns.boxplot(data=df_merge_1min, x=df_merge_1min.index.day_name(), y="NOx / ppbv",order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],ax=ax[1])
sns.boxplot(data=df_merge_1min, x=df_merge_1min.index.day_name(), y="O3 / ppbv",order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],ax=ax[2])
ax[0].set(ylim=(0, 1500))
ax[1].set(ylim=(0, 100))
ax[0].set(xlabel="")
ax[1].set(xlabel="")
ax[2].set(xlabel="")




# %%Munge 1min data for diurnal analysis
df_merge_1min_diurnal = df_merge_1min.groupby(df_merge_1min.index.time).describe()
df_merge_1min_diurnal.index = pd.to_datetime(df_merge_1min_diurnal.index.astype(str))


#%%
fig,ax = plt.subplots(2,1)
ax1=ax[0]
ax2=ax[1]

ax1.plot(df_merge_1min_diurnal.index, df_merge_1min_diurnal['NO3_ams']['mean'], linewidth=2.0,c='b',label='NO3')
ax1.plot(df_merge_1min_diurnal.index, df_merge_1min_diurnal['Org_ams']['mean'], linewidth=2.0,c='g',label='Org')
ax1.plot(df_merge_1min_diurnal.index, df_merge_1min_diurnal['SO4_ams']['mean'], linewidth=2.0,c='r',label='SO4')
ax1.plot(df_merge_1min_diurnal.index, df_merge_1min_diurnal['NH4_ams']['mean'], linewidth=2.0,c='o',label='NH4')
ax1.plot(df_merge_1min_diurnal.index, df_merge_1min_diurnal['Chl_ams']['mean'], linewidth=2.0,c='pink',label='Chl')


