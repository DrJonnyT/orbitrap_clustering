# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import h5py
from functions.math import round_to_nearest_x_odd, sqrt_sum_squares
from functions.igor_time_to_unix import igor_time_to_unix
import pdb

def load_pre_PMF_data(filepath,join='inner',justBeijing=False,justDelhi=False):
    
    with h5py.File(filepath, 'r') as hf:
        Beijing_winter_mz = np.array(hf['Beijing_Winter']['noNaNs_mz']).astype(float).round(3)
        Beijing_winter_mw = np.array(hf['Beijing_Winter']['noNaNs_MW']).astype(float).round(3)
        Beijing_winter_formula = pd.Series(hf['Beijing_Winter']['noNaNs_Formula']).astype(str).str.replace(r'\'', '',regex=True).str.replace(r'b', '',regex=True)
        Beijing_winter_RT = pd.Series(hf['Beijing_Winter']['noNaNs_RT_min']).apply(lambda x: round_to_nearest_x_odd(x,1))
        Beijing_winter_data = np.array(hf['Beijing_Winter']['noNaNs_bjgdata3_wtr'])
        Beijing_winter_err = np.array(hf['Beijing_Winter']['noNaNs_bjgerr3_wtrPropWk'])
        Beijing_winter_time = pd.to_datetime(igor_time_to_unix(pd.Series(hf['Beijing_Winter']['noNaNs_midtime_wtr'])),unit='s')
        df_Beijing_winter_data = pd.DataFrame(Beijing_winter_data,index=Beijing_winter_time)
        df_Beijing_winter_err = pd.DataFrame(Beijing_winter_err,index=Beijing_winter_time)
        
        Beijing_summer_mz = np.array(hf['Beijing_Summer']['noNaNs_mz']).astype(float).round(3)
        Beijing_summer_mw = np.array(hf['Beijing_Summer']['noNaNs_MW']).astype(float).round(3)
        Beijing_summer_formula = pd.Series(hf['Beijing_Summer']['noNaNs_Formula']).astype(str).str.replace(r'\'', '',regex=True).str.replace(r'b', '',regex=True)
        Beijing_summer_RT = pd.Series(hf['Beijing_Summer']['noNaNs_RT_min']).apply(lambda x: round_to_nearest_x_odd(x,1))
        Beijing_summer_data = np.array(hf['Beijing_Summer']['noNaNs_bjgdata3_smr'])
        Beijing_summer_err = np.array(hf['Beijing_Summer']['noNaNs_bjgerr3_smrPropWk'])
        Beijing_summer_time = pd.to_datetime(igor_time_to_unix(pd.Series(hf['Beijing_Summer']['noNaNs_midtime_smr'])),unit='s')
        df_Beijing_summer_data = pd.DataFrame(Beijing_summer_data,index=Beijing_summer_time)
        df_Beijing_summer_err = pd.DataFrame(Beijing_summer_err,index=Beijing_summer_time)
        
        Delhi_summer_mz = np.array(hf['Delhi_Summer']['noNaNs_mz']).astype(float).round(3)
        Delhi_summer_mw = np.array(hf['Delhi_Summer']['noNaNs_MW']).astype(float).round(3)
        Delhi_summer_formula = pd.Series(hf['Delhi_Summer']['noNaNs_Formula']).astype(str).str.replace(r'\'', '',regex=True).str.replace(r'b', '',regex=True)
        Delhi_summer_RT = pd.Series(hf['Delhi_Summer']['noNaNs_RT_min']).apply(lambda x: round_to_nearest_x_odd(x,1))
        Delhi_summer_data = np.array(hf['Delhi_Summer']['noNaNs_dlhdata1_smr'])
        Delhi_summer_err = np.array(hf['Delhi_Summer']['noNaNs_dlherr1_smrPropWk'])
        Delhi_summer_time = pd.to_datetime(igor_time_to_unix(pd.Series(hf['Delhi_Summer']['noNaNs_midtime_smr'])),unit='s')
        df_Delhi_summer_data = pd.DataFrame(Delhi_summer_data,index=Delhi_summer_time)
        df_Delhi_summer_err = pd.DataFrame(Delhi_summer_err,index=Delhi_summer_time)
        
        Delhi_autumn_mz = np.array(hf['Delhi_Autumn']['noNaNs_mz']).astype(float).round(3)
        Delhi_autumn_mw = np.array(hf['Delhi_Autumn']['noNaNs_MW']).astype(float).round(3)
        Delhi_autumn_formula = pd.Series(hf['Delhi_Autumn']['noNaNs_Formula']).astype(str).str.replace(r'\'', '',regex=True).str.replace(r'b', '',regex=True)
        Delhi_autumn_RT = pd.Series(hf['Delhi_Autumn']['noNaNs_RT_min']).apply(lambda x: round_to_nearest_x_odd(x,1))
        Delhi_autumn_time = pd.to_datetime(igor_time_to_unix(pd.Series(hf['Delhi_Autumn']['noNaNs_midtime_aut'])),unit='s')
        df_Delhi_autumn_data = pd.DataFrame(hf['Delhi_Autumn']['noNaNs_dlhdata1_aut'],index=Delhi_autumn_time)
        df_Delhi_autumn_err = pd.DataFrame(hf['Delhi_Autumn']['noNaNs_dlherr1_autPropWk'],index=Delhi_autumn_time)

        #Merge peaks of same molecule and rounded RT
        df_Beijing_winter_data = df_Beijing_winter_data.groupby([Beijing_winter_formula,Beijing_winter_RT],axis=1).aggregate("sum")
        df_Beijing_winter_err = df_Beijing_winter_err.groupby([Beijing_winter_formula,Beijing_winter_RT],axis=1).aggregate(sqrt_sum_squares)
        df_Beijing_summer_data = df_Beijing_summer_data.groupby([Beijing_summer_formula,Beijing_summer_RT],axis=1).aggregate("sum")
        df_Beijing_summer_err = df_Beijing_summer_err.groupby([Beijing_summer_formula,Beijing_summer_RT],axis=1).aggregate(sqrt_sum_squares)
        df_Delhi_summer_data = df_Delhi_summer_data.groupby([Delhi_summer_formula,Delhi_summer_RT],axis=1).aggregate("sum")
        df_Delhi_summer_err = df_Delhi_summer_err.groupby([Delhi_summer_formula,Delhi_summer_RT],axis=1).aggregate(sqrt_sum_squares)
        df_Delhi_autumn_data = df_Delhi_autumn_data.groupby([Delhi_autumn_formula,Delhi_autumn_RT],axis=1).aggregate("sum")
        df_Delhi_autumn_err = df_Delhi_autumn_err.groupby([Delhi_autumn_formula,Delhi_autumn_RT],axis=1).aggregate(sqrt_sum_squares)
        
    
    #Make mz lookup
    df_Beijing_winter_mz = pd.DataFrame(Beijing_winter_mz).T.groupby([Beijing_winter_formula,Beijing_winter_RT],axis=1).aggregate("first")
    df_Beijing_summer_mz = pd.DataFrame(Beijing_summer_mz).T.groupby([Beijing_summer_formula,Beijing_summer_RT],axis=1).aggregate("first")
    df_Delhi_summer_mz = pd.DataFrame(Delhi_summer_mz).T.groupby([Delhi_summer_formula,Delhi_summer_RT],axis=1).aggregate("first")
    df_Delhi_autumn_mz = pd.DataFrame(Delhi_autumn_mz).T.groupby([Delhi_autumn_formula,Delhi_autumn_RT],axis=1).aggregate("first")
    
    #Make MW lookup
    df_Beijing_winter_mw = pd.DataFrame(Beijing_winter_mw).T.groupby([Beijing_winter_formula,Beijing_winter_RT],axis=1).aggregate("first")
    df_Beijing_summer_mw = pd.DataFrame(Beijing_summer_mw).T.groupby([Beijing_summer_formula,Beijing_summer_RT],axis=1).aggregate("first")
    df_Delhi_summer_mw = pd.DataFrame(Delhi_summer_mw).T.groupby([Delhi_summer_formula,Delhi_summer_RT],axis=1).aggregate("first")
    df_Delhi_autumn_mw = pd.DataFrame(Delhi_autumn_mw).T.groupby([Delhi_autumn_formula,Delhi_autumn_RT],axis=1).aggregate("first")
    
    #Merge the datasets
    if justBeijing:
        df_all_data = pd.concat([df_Beijing_winter_data,df_Beijing_summer_data],join=join).fillna(0)
        df_all_err = pd.concat([df_Beijing_winter_err,df_Beijing_summer_err],join=join).fillna(0)  
        ds_all_mz = pd.concat([df_Beijing_winter_mz,df_Beijing_summer_mz],join=join).mean()
        ds_all_mw = pd.concat([df_Beijing_winter_mw,df_Beijing_summer_mw],join=join).mean()
    elif justDelhi:
        df_all_data = pd.concat([df_Delhi_summer_data,df_Delhi_autumn_data],join=join).fillna(0)
        df_all_err = pd.concat([df_Delhi_summer_err,df_Delhi_autumn_err],join=join).fillna(0)  
        ds_all_mz = pd.concat([df_Delhi_summer_mz,df_Delhi_autumn_mz],join=join).mean()
        ds_all_mw = pd.concat([df_Delhi_summer_mw,df_Delhi_autumn_mw],join=join).mean()
    else:
        df_all_data = pd.concat([df_Beijing_winter_data,df_Beijing_summer_data,df_Delhi_summer_data,df_Delhi_autumn_data],join=join).fillna(0)
        df_all_err = pd.concat([df_Beijing_winter_err,df_Beijing_summer_err,df_Delhi_summer_err,df_Delhi_autumn_err],join=join).fillna(0)  
        ds_all_mz = pd.concat([df_Beijing_winter_mz,df_Beijing_summer_mz,df_Delhi_summer_mz,df_Delhi_autumn_mz],join=join).mean()
        ds_all_mw = pd.concat([df_Beijing_winter_mw,df_Beijing_summer_mw,df_Delhi_summer_mw,df_Delhi_autumn_mw],join=join).mean()

    
    #Sort by mz
    df_all_data.columns = ds_all_mz
    df_all_err.columns = ds_all_mz
    ds_all_mw.index = ds_all_mz
    df_all_data.sort_index(axis=1,inplace=True)
    df_all_err.sort_index(axis=1,inplace=True)
    ds_all_mw.sort_index(inplace=True)
    ds_all_mz.sort_values(inplace=True)
    df_all_data.columns = ds_all_mz.index
    df_all_err.columns = ds_all_mz.index
    ds_all_mw.index = ds_all_mz.index
    
    #Remove mystery sample that wasn't in my previous data
    df_all_data.drop(pd.to_datetime('2017/06/02 23:19:28'),inplace=True)
    df_all_err.drop(pd.to_datetime('2017/06/02 23:19:28'),inplace=True)
    
    return df_all_data, df_all_err, ds_all_mz, ds_all_mw