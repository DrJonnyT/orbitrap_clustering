# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 17:00:03 2022

@author: mbcx5jt5
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
#import math

import os
os.chdir('C:/Work/Python/Github/Orbitrap_clustering')
from ae_functions import *



#%%Load Sari's pre-PMF filtered calibrated data from August 2022
import h5py
##E.g. round to nearest multiple of 2 would be rounded to 0, 2, 4, etc
def round_to_nearest_x_even(num,x):
    return np.round(num / x) * x
##E.g. round to nearest multiple of 2 would be rounded to 0, 2, 4, etc
def round_to_nearest_x_odd(num,x):
    return np.floor((num) / x) * x +x/2

#Merge peaks with same mz and RT
def sqrt_sum_squares(x):
    x = np.array(x)
    return np.sqrt(np.sum(np.multiply(x,x)))

#def Load_pre_PMF_data(filepath):

filepath = r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\PMF data\ORBITRAP_Data_Pre_PMF.h5"

with h5py.File(filepath, 'r') as hf:
    Beijing_winter_mz = np.array(hf['Beijing_Winter']['noNaNs_mz']).astype(float).round(3)
    Beijing_winter_formula = pd.Series(hf['Beijing_Winter']['noNaNs_Formula']).astype(str).str.replace(r'\'', '',regex=True).str.replace(r'b', '',regex=True)
    Beijing_winter_RT = pd.Series(hf['Beijing_Winter']['noNaNs_RT_min']).apply(lambda x: round_to_nearest_x_odd(x,1))
    Beijing_winter_data = np.array(hf['Beijing_Winter']['noNaNs_bjgdata3_wtr'])
    Beijing_winter_err = np.array(hf['Beijing_Winter']['noNaNs_bjgerr3_wtrPropWk'])
    Beijing_winter_time = np.array(hf['Beijing_Winter']['noNaNs_midtime_wtr'])
    df_Beijing_winter_data = pd.DataFrame(Beijing_winter_data,index=Beijing_winter_time)
    df_Beijing_winter_err = pd.DataFrame(Beijing_winter_data,index=Beijing_winter_time)
    
    Beijing_summer_mz = np.array(hf['Beijing_Summer']['noNaNs_mz']).astype(float).round(3)
    Beijing_summer_formula = pd.Series(hf['Beijing_Summer']['noNaNs_Formula']).astype(str).str.replace(r'\'', '',regex=True).str.replace(r'b', '',regex=True)
    Beijing_summer_RT = pd.Series(hf['Beijing_Summer']['noNaNs_RT_min']).apply(lambda x: round_to_nearest_x_odd(x,1))
    Beijing_summer_data = np.array(hf['Beijing_Summer']['noNaNs_bjgdata3_smr'])
    Beijing_summer_err = np.array(hf['Beijing_Summer']['noNaNs_bjgerr3_smrPropWk'])
    Beijing_summer_time = np.array(hf['Beijing_Summer']['noNaNs_midtime_smr'])
    df_Beijing_summer_data = pd.DataFrame(Beijing_summer_data,index=Beijing_summer_time)
    df_Beijing_summer_err = pd.DataFrame(Beijing_summer_data,index=Beijing_summer_time)
    
    Delhi_summer_mz = np.array(hf['Delhi_Summer']['noNaNs_mz']).astype(float).round(3)
    Delhi_summer_formula = pd.Series(hf['Delhi_Summer']['noNaNs_Formula']).astype(str).str.replace(r'\'', '',regex=True).str.replace(r'b', '',regex=True)
    Delhi_summer_RT = pd.Series(hf['Delhi_Summer']['noNaNs_RT_min']).apply(lambda x: round_to_nearest_x_odd(x,1))
    Delhi_summer_data = np.array(hf['Delhi_Summer']['noNaNs_dlhdata1_smr'])
    Delhi_summer_err = np.array(hf['Delhi_Summer']['noNaNs_dlherr1_smrPropWk'])
    Delhi_summer_time = np.array(hf['Delhi_Summer']['noNaNs_midtime_smr'])
    df_Delhi_summer_data = pd.DataFrame(Delhi_summer_data,index=Delhi_summer_time)
    df_Delhi_summer_err = pd.DataFrame(Delhi_summer_data,index=Delhi_summer_time)
    
    Delhi_autumn_mz = np.array(hf['Delhi_Autumn']['noNaNs_mz']).astype(float).round(3)
    Delhi_autumn_formula = pd.Series(hf['Delhi_Autumn']['noNaNs_Formula']).astype(str).str.replace(r'\'', '',regex=True).str.replace(r'b', '',regex=True)
    Delhi_autumn_RT = pd.Series(hf['Delhi_Autumn']['noNaNs_RT_min']).apply(lambda x: round_to_nearest_x_odd(x,1))
    Delhi_autumn_data = np.array(hf['Delhi_Autumn']['noNaNs_dlhdata1_aut'])
    Delhi_autumn_err = np.array(hf['Delhi_Autumn']['noNaNs_dlherr1_autPropWk'])
    Delhi_autumn_time = np.array(hf['Delhi_Autumn']['noNaNs_midtime_aut'])
    df_Delhi_autumn_data = pd.DataFrame(Delhi_autumn_data,index=Delhi_autumn_time)
    df_Delhi_autumn_err = pd.DataFrame(Delhi_autumn_data,index=Delhi_autumn_time)
    
    df_Beijing_winter_data = df_Beijing_winter_data.groupby([Beijing_winter_formula,Beijing_winter_RT],axis=1).aggregate("sum")
    df_Beijing_winter_err = df_Beijing_winter_err.groupby([Beijing_winter_formula,Beijing_winter_RT],axis=1).aggregate(sqrt_sum_squares)
    df_Beijing_summer_data = df_Beijing_summer_data.groupby([Beijing_summer_formula,Beijing_summer_RT],axis=1).aggregate("sum")
    df_Beijing_summer_err = df_Beijing_summer_err.groupby([Beijing_summer_formula,Beijing_summer_RT],axis=1).aggregate(sqrt_sum_squares)
    df_Delhi_summer_data = df_Delhi_summer_data.groupby([Delhi_summer_formula,Delhi_summer_RT],axis=1).aggregate("sum")
    df_Delhi_summer_err = df_Delhi_summer_err.groupby([Delhi_summer_formula,Delhi_summer_RT],axis=1).aggregate(sqrt_sum_squares)
    df_Delhi_autumn_data = df_Delhi_autumn_data.groupby([Delhi_autumn_formula,Delhi_autumn_RT],axis=1).aggregate("sum")
    df_Delhi_autumn_err = df_Delhi_autumn_err.groupby([Delhi_autumn_formula,Delhi_autumn_RT],axis=1).aggregate(sqrt_sum_squares)






#%%Merge the data frames
df_all_data = pd.concat([df_Beijing_winter_data,df_Beijing_summer_data,df_Delhi_summer_data,df_Delhi_autumn_data],join='inner')
df_all_err = pd.concat([df_Beijing_winter_err,df_Beijing_summer_err,df_Delhi_summer_err,df_Delhi_autumn_err],join='inner')

plt.figure(figsize=(8,8))
plt.scatter(df_all_err,df_all_data)

#Make mz lookup
df_Beijing_winter_mz = pd.DataFrame(Beijing_winter_mz).T.groupby([Beijing_winter_formula,Beijing_winter_RT],axis=1).aggregate("first")
df_Beijing_summer_mz = pd.DataFrame(Beijing_summer_mz).T.groupby([Beijing_summer_formula,Beijing_summer_RT],axis=1).aggregate("first")
df_Delhi_summer_mz = pd.DataFrame(Delhi_summer_mz).T.groupby([Delhi_summer_formula,Delhi_summer_RT],axis=1).aggregate("first")
df_Delhi_autumn_mz = pd.DataFrame(Delhi_autumn_mz).T.groupby([Delhi_autumn_formula,Delhi_autumn_RT],axis=1).aggregate("first")
ds_all_mz = pd.concat([df_Beijing_winter_mz,df_Beijing_summer_mz,df_Delhi_summer_mz,df_Delhi_autumn_mz],join='inner').iloc[0]

#Sort 
df_all_data.columns = ds_all_mz
df_all_err.columns = ds_all_mz
df_all_data.sort_index(axis=1,inplace=True)
df_all_err.sort_index(axis=1,inplace=True)
ds_all_mz.sort_values(inplace=True)
df_all_data.columns = ds_all_mz.index
df_all_err.columns = ds_all_mz.index

plt.figure(figsize=(8,8))
plt.scatter(df_all_err,df_all_data)

#%%Make molecule lookup
# ds_all_formula = pd.concat([Beijing_summer_formula,Beijing_winter_formula,Delhi_summer_formula,Delhi_autumn_formula])
# ds_all_formula.index = np.concatenate([Beijing_summer_mz,Beijing_winter_mz,Delhi_summer_mz,Delhi_autumn_mz])
# ds_all_formula = ds_all_formula[df_all_data.columns.get_level_values(0)]

# df_all_formula = pd.DataFrame()
# df_all_formula['formula'] = ds_all_formula.values
# df_all_formula['mz'] = ds_all_formula.index
# df_all_formula.drop_duplicates(inplace=True)


# #ds_all_formula.drop_duplicates(inplace=True)
# ds_all_formula = ds_all_formula.loc[ds_all_formula.index.drop_duplicates()]
