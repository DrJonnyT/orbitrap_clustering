# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 14:02:30 2022

@author: mbcx5jt5
"""

import pandas as pd
import math
import datetime as dt
import numpy as np
import pdb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import re
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score,silhouette_score
from scipy.stats import pearsonr

from chem.chemform import ChemForm
from plotting.cmap_EOS11 import cmap_EOS11


#######################
####FILE_LOADERS#######
#######################
def beijing_load(peaks_filepath,metadata_filepath,peaks_sheetname="DEFAULT",metadata_sheetname="DEFAULT",subtract_blank=True):
    #Define constants
    CalPA = 28138.3956527531 #Pinonic acid calibration
    FF = 0.1115 #Fraction of the filter used for analysis
    EF = 0.85 #Extraction efficiency of pinonic acid
    RIE = 4.2 #Relative ionization efficiency of pinonic acid
    
    LOD = 8.6 ##Limit of detection in ppb, so raw signal/CalPA
    u_RIE = 3.9 # Uncertainty in RIE
    u_analytical = 0.063 # 6.3%
    
    
    #Load metadata
    if(metadata_sheetname=="DEFAULT"):
        df_beijing_metadata = pd.read_excel(metadata_filepath,engine='openpyxl',
                                           usecols='A:K',nrows=329,converters={'mid_datetime': str})
    else:
        df_beijing_metadata = pd.read_excel(metadata_filepath,engine='openpyxl',
                                       sheet_name=metadata_sheetname,usecols='A:K',nrows=329,converters={'mid_datetime': str})
    df_beijing_metadata['Sample.ID'] = df_beijing_metadata['Sample.ID'].astype(str)

    #Load peaks
    if(peaks_sheetname=="DEFAULT"):
        df_beijing_raw = pd.read_excel(peaks_filepath,engine='openpyxl')
    else:
        df_beijing_raw = pd.read_excel(peaks_filepath,engine='openpyxl',sheet_name=peaks_sheetname)
        
    #Filter out "bad" columns
    df_beijing_raw = orbitrap_filter(df_beijing_raw)

    #Cut some fluff columns out and make new df
    df_beijing_data = df_beijing_raw.iloc[:,list(range(4,len(df_beijing_raw.columns)))].copy()
    index_backup = df_beijing_data.index

    #Apply calibrations
    #Step 1. Apply CalPA, Pinonic acid calibration to turn area into ppb (ug/ml)
    df_beijing_data = df_beijing_data * (1/CalPA)
    
    #Step 2. Calculate error in ppb
    df_beijing_err = df_beijing_data.copy()
    #pdb.set_trace()
    df_beijing_err[df_beijing_data.abs() <= LOD] = 5/6 * LOD
    df_beijing_err[df_beijing_data.abs() > LOD] = np.sqrt( (df_beijing_data[df_beijing_data.abs() > LOD] * u_analytical)**2  +  (df_beijing_data[df_beijing_data.abs() > LOD] * 3.9/4.2)**2  +  (0.5 * LOD)**2  )
    
    #Step 1. Subtract blanks
    if(subtract_blank == True):
        #Extract blank
        beijing_blank = df_beijing_data.iloc[:,316].copy()
        beijing_blank_err = df_beijing_err.iloc[:,316].copy()
        #Subtract blank
        df_beijing_data = df_beijing_data.subtract(beijing_blank.values,axis=0)
        df_beijing_err = ((df_beijing_err**2).add(np.square(beijing_blank_err.values),axis=0))**(1/2)
    
    
    #Step 4. Apply FF, EF and RIE calibrations
    df_beijing_data = df_beijing_data * (1/FF) * (1/EF) * RIE
    df_beijing_err = df_beijing_err * (1/FF) * (1/EF) * RIE
    
    
    #Set the index to sample ID for merging peaks with metadata
    sample_id = df_beijing_data.columns.str.split('_|.raw').str[2]
    df_beijing_data.columns = sample_id
    df_beijing_err.columns = sample_id
    df_beijing_data = df_beijing_data.transpose()
    df_beijing_err = df_beijing_err.transpose()
    
    #Add on the metadata
    df_beijing_metadata.set_index(df_beijing_metadata["Sample.ID"].astype('str'),inplace=True)    
    df_beijing_data = pd.concat([df_beijing_data, df_beijing_metadata[['Volume_m3', 'Dilution_mL']]], axis=1, join="inner")
    df_beijing_err = pd.concat([df_beijing_err, df_beijing_metadata[['Volume_m3', 'Dilution_mL']]], axis=1, join="inner")

    #Step 5. Divide the data columns by the sample volume and multiply by the dilution liquid volume (pinonic acid)
    df_beijing_data = df_beijing_data.div(df_beijing_data['Volume_m3'], axis=0).mul(df_beijing_data['Dilution_mL'], axis=0) / 1000  #The 1000 comes from 1000ml division in the dilution
    df_beijing_data.drop(columns=['Volume_m3','Dilution_mL'],inplace=True)
    df_beijing_data['mid_datetime'] = pd.to_datetime(df_beijing_metadata['mid_datetime'],yearfirst=True)
    df_beijing_data.set_index('mid_datetime',inplace=True)
    df_beijing_data = df_beijing_data.astype(float)   
    df_beijing_data.columns = index_backup
    
    df_beijing_err = df_beijing_err.div(df_beijing_err['Volume_m3'], axis=0).mul(df_beijing_err['Dilution_mL'], axis=0) / 1000  #The 1000 comes from 1000ml division in the dilution
    df_beijing_err.drop(columns=['Volume_m3','Dilution_mL'],inplace=True)
    df_beijing_err['mid_datetime'] = pd.to_datetime(df_beijing_metadata['mid_datetime'],yearfirst=True)
    df_beijing_err.set_index('mid_datetime',inplace=True)
    df_beijing_err = df_beijing_err.astype(float)   
    df_beijing_err.columns = index_backup
    
    return df_beijing_data, df_beijing_err, df_beijing_metadata, df_beijing_raw



# def delhi_load(peaks_filepath,metadata_filepath,peaks_sheetname="DEFAULT",metadata_sheetname="DEFAULT",subtract_blank=True):
#     if(metadata_sheetname=="DEFAULT"):
#         df_delhi_metadata = pd.read_excel(metadata_filepath,engine='openpyxl',
#                                            usecols='a:N',skiprows=0,nrows=108, converters={'mid_datetime': str})
#     else:
#         df_delhi_metadata = pd.read_excel(metadata_filepath,engine='openpyxl',
#                                            sheet_name=metadata_sheetname,usecols='a:N',skiprows=0,nrows=108, converters={'mid_datetime': str})
#     #df_delhi_metadata['Sample.ID'] = df_delhi_metadata['Sample.ID'].astype(str)

#     if(peaks_sheetname=="DEFAULT"):
#         df_delhi_raw = pd.read_excel(peaks_filepath,engine='openpyxl')
#     else:
#         df_delhi_raw = pd.read_excel(peaks_filepath,engine='openpyxl',sheet_name=peaks_sheetname)
    

#     #Get rid of columns that are not needed
#     df_delhi_raw.drop(df_delhi_raw.iloc[:,np.r_[0, 2:11, 14:18]],axis=1,inplace=True)
    
    
    
#     #Fix column labels so they are consistent
#     df_delhi_raw.columns = df_delhi_raw.columns.str.replace('DelhiS','Delhi_S')
    
#     df_delhi_metadata.drop(labels="Filter ID.1",axis=1,inplace=True)
#     df_delhi_metadata.set_index("Filter ID",inplace=True)
#     #Get rid of bad filters, based on the notes
#     df_delhi_metadata.drop(labels=["-","S25","S42","S51","S55","S68","S72"],axis=0,inplace=True)
     
#     #Filter out "bad" columns
#     df_delhi_raw = orbitrap_filter(df_delhi_raw)
    
#     #Cut some fluff columns out and make new df
#     df_delhi_data = df_delhi_raw.iloc[:,list(range(4,len(df_delhi_raw.columns)))].copy()
#     df_delhi_data.columns = df_delhi_data.columns.str.replace('DelhiS','Delhi_S')
#     index_backup = df_delhi_data.index
    
#     if(subtract_blank == True):
#         #Extract blanks
#         df_delhi_raw_blanks = df_delhi_raw[df_delhi_raw.columns[df_delhi_raw.columns.str.contains('Blank')]] 
#         #Subtract mean blank
#         df_delhi_data = df_delhi_data.subtract(df_delhi_raw_blanks.transpose().mean().values,axis=0)
    
    
#     sample_id = df_delhi_data.columns.str.split('_|.raw').str[2]
#     df_delhi_data.columns = sample_id

#     df_delhi_data = df_delhi_data.transpose()    
    
#     #Add on the metadata    
#     df_delhi_data = pd.concat([df_delhi_data, df_delhi_metadata[['Volume / m3', 'Dilution']]], axis=1, join="inner")

#     #Divide the data columns by the sample volume and multiply by the dilution liquid volume (pinonic acid)
#     df_delhi_data = df_delhi_data.div(df_delhi_data['Volume / m3'], axis=0).mul(df_delhi_data['Dilution'], axis=0)
#     df_delhi_data.drop(columns=['Volume / m3','Dilution'],inplace=True)
    
#     #Some final QA
#     df_delhi_data['mid_datetime'] = pd.to_datetime(df_delhi_metadata['Mid-Point'],yearfirst=True)
#     df_delhi_data.set_index('mid_datetime',inplace=True)
#     df_delhi_data = df_delhi_data.astype(float)
#     df_delhi_data.columns = index_backup

#     return df_delhi_raw, df_delhi_data, df_delhi_metadata


#Load the Delhi data, from files 
#Raw peaks: DH_UnAmbNeg9.0_20210409.xlsx
#Metadata- Delhi_massloading_autumn_summer.xlsx
def delhi_load2(path,subtract_blank=True,output="DEFAULT"):
    CalPA = 28138.3956527531 #Pinonic acid calibration
    FF = 0.1115 #Fraction of the filter used for analysis
    EF = 0.85 #Extraction efficiency of pinonic acid
    RIE = 4.2 #Relative ionization efficiency of pinonic acid
    
    LOD = 8.6 ##Limit of detection in ppb, so raw signal/CalPA
    u_RIE = 3.9 # Uncertainty in RIE
    u_analytical = 0.063 # 6.3%
    
    
    peaks_filepath = path + 'DH_UnAmbNeg9.0_20210409.xlsx'
    metadata_filepath = path + 'Delhi_massloading_autumn_summer.xlsx'
    df_metadata_autumn = pd.read_excel(metadata_filepath,engine='openpyxl',
                                           sheet_name='autumn',usecols='a:L',skiprows=0,nrows=108, converters={'mid_datetime': str})
    df_metadata_summer = pd.read_excel(metadata_filepath,engine='openpyxl',
                                           sheet_name='summer_premonsoon',usecols='a:N',skiprows=0,nrows=108, converters={'mid_datetime': str})
      
    df_metadata_summer['sample_ID'] = df_metadata_summer['sample_ID'].astype(str)
    
    df_delhi_raw = pd.read_excel(peaks_filepath,engine='openpyxl')
    #df_delhi_data = 1
    
    #Remove columns that are not needed
    df_delhi_raw.drop(df_delhi_raw.iloc[:,np.r_[0:7, 10:14]],axis=1,inplace=True)
    
    #This column is not in the spreadsheet, so create one using the average difference from other data 
    df_delhi_raw.insert(loc=2, column='m/z', value=(df_delhi_raw['Molecular Weight'] - 1.0073))
    
    #Filter out "bad" columns
    df_delhi_raw = orbitrap_filter(df_delhi_raw)
    #pdb.set_trace()
    df_delhi_raw_chem = df_delhi_raw.iloc[:,np.r_[0:4]]
    df_delhi_raw_autumn = df_delhi_raw.iloc[:,np.r_[37:138]]
    df_delhi_raw_summer = df_delhi_raw.iloc[:,np.r_[4:37]]
    df_delhi_raw_blanks = df_delhi_raw.iloc[:,np.r_[138:142]]
    
   
    df_metadata_autumn.set_index("sample_ID",inplace=True)
    #Get rid of bad filters, based on the notes
    df_metadata_autumn.drop(labels=["-","S25","S42","S51","S55","S68","S72"],axis=0,inplace=True)
    
    df_metadata_summer.set_index("sample_ID",inplace=True)
    
    df_metadata_summer['start_datetime'] = pd.to_datetime(df_metadata_summer['start_datetime'],yearfirst=True)
    df_metadata_summer['mid_datetime'] = pd.to_datetime(df_metadata_summer['mid_datetime'],yearfirst=True)
    df_metadata_summer['end_datetime'] = pd.to_datetime(df_metadata_summer['end_datetime'],yearfirst=True)
    
    df_metadata_autumn['start_datetime'] = pd.to_datetime(df_metadata_autumn['start_datetime'],yearfirst=True)
    df_metadata_autumn['mid_datetime'] = pd.to_datetime(df_metadata_autumn['mid_datetime'],yearfirst=True)
    df_metadata_autumn['end_datetime'] = pd.to_datetime(df_metadata_autumn['end_datetime'],yearfirst=True)   
    
    
    #Cut some fluff columns out and make new df
    df_delhi_data_autumn = df_delhi_raw_autumn.copy()
    df_delhi_data_summer = df_delhi_raw_summer.copy()
    index_backup_autumn = df_delhi_data_autumn.index
    index_backup_summer = df_delhi_data_summer.index
    
    df_delhi_data_blanks = df_delhi_raw_blanks.copy()
        
    #Apply calibrations      
    #Step 1. Apply CalPA, Pinonic acid calibration to turn area into ppb (ug/ml)
    df_delhi_data_autumn = df_delhi_data_autumn * (1/CalPA)
    df_delhi_data_summer = df_delhi_data_summer * (1/CalPA)
    df_delhi_data_blanks = df_delhi_data_blanks * (1/CalPA)
    
    #Step 2. Calculate error in ppb
    df_delhi_err_autumn = df_delhi_data_autumn.copy()
    df_delhi_err_summer = df_delhi_data_summer.copy()
    df_delhi_err_blanks = df_delhi_data_blanks.copy()
    df_delhi_err_autumn[df_delhi_data_autumn.abs() < LOD] = 5/6 * LOD
    df_delhi_err_summer[df_delhi_data_summer.abs() < LOD] = 5/6 * LOD
    df_delhi_err_blanks[df_delhi_data_blanks.abs() < LOD] = 5/6 * LOD
    df_delhi_err_autumn[df_delhi_data_autumn.abs() >= LOD] = np.sqrt( (df_delhi_data_autumn[df_delhi_data_autumn.abs() >= LOD] * u_analytical)**2  +  (df_delhi_data_autumn[df_delhi_data_autumn.abs() >= LOD] * 3.9/4.2)**2  +  (0.5 * LOD)**2  )
    df_delhi_err_summer[df_delhi_data_summer.abs() >= LOD] = np.sqrt( (df_delhi_data_summer[df_delhi_data_summer.abs() >= LOD] * u_analytical)**2  +  (df_delhi_data_summer[df_delhi_data_summer.abs() >= LOD] * 3.9/4.2)**2  +  (0.5 * LOD)**2  )
    df_delhi_err_blanks[df_delhi_data_blanks.abs() >= LOD] = np.sqrt( (df_delhi_data_blanks[df_delhi_data_blanks.abs() >= LOD] * u_analytical)**2  +  (df_delhi_data_blanks[df_delhi_data_blanks.abs() >= LOD] * 3.9/4.2)**2  +  (0.5 * LOD)**2  )
    
    #Step 1. Subtract blanks
    if(subtract_blank == True):
        #Subtract mean blank        
        delhi_blank_mean = df_delhi_data_blanks.mean(axis=1)
        delhi_blank_err = df_delhi_err_blanks.mean(axis=1)
        
        df_delhi_data_autumn = df_delhi_data_autumn.subtract(delhi_blank_mean.values,axis=0)
        df_delhi_data_summer = df_delhi_data_summer.subtract(delhi_blank_mean.values,axis=0)
        
        df_delhi_err_autumn = ((df_delhi_err_autumn**2).add(np.square(delhi_blank_err.values),axis=0))**(1/2)
        df_delhi_err_summer = ((df_delhi_err_summer**2).add(np.square(delhi_blank_err.values),axis=0))**(1/2)   

    
    #Step 4. Apply FF, EF and RIE calibrations
    df_delhi_data_autumn = df_delhi_data_autumn * (1/FF) * (1/EF) * RIE
    df_delhi_err_autumn = df_delhi_err_autumn * (1/FF) * (1/EF) * RIE
    df_delhi_data_summer = df_delhi_data_summer * (1/FF) * (1/EF) * RIE
    df_delhi_err_summer = df_delhi_err_summer * (1/FF) * (1/EF) * RIE
    
    #Sort out indexing
    sample_id_autumn = df_delhi_data_autumn.columns.str.split('_|.raw').str[2]
    sample_id_summer = df_delhi_data_summer.columns.str.split('_|.raw').str[2]
    df_delhi_data_autumn.columns = sample_id_autumn
    df_delhi_err_autumn.columns = sample_id_autumn
    df_delhi_data_summer.columns = sample_id_summer
    df_delhi_err_summer.columns = sample_id_summer

    df_delhi_data_autumn = df_delhi_data_autumn.transpose()
    df_delhi_err_autumn = df_delhi_err_autumn.transpose()
    df_delhi_data_summer = df_delhi_data_summer.transpose() 
    df_delhi_err_summer = df_delhi_err_summer.transpose()
    
    #pdb.set_trace()
    #Add on the metadata    
    df_delhi_data_autumn = pd.concat([df_delhi_data_autumn, df_metadata_autumn[['volume_m3', 'dilution']]], axis=1, join="inner")
    df_delhi_err_autumn = pd.concat([df_delhi_err_autumn, df_metadata_autumn[['volume_m3', 'dilution']]], axis=1, join="inner")
    df_delhi_data_summer = pd.concat([df_delhi_data_summer, df_metadata_summer[['volume_m3', 'dilution']]], axis=1, join="inner")
    df_delhi_err_summer = pd.concat([df_delhi_err_summer, df_metadata_summer[['volume_m3', 'dilution']]], axis=1, join="inner")

    #Step 5. #Divide the data columns by the sample volume and multiply by the dilution liquid volume (pinonic acid)
    df_delhi_err_autumn = df_delhi_err_autumn.div(df_delhi_err_autumn['volume_m3'], axis=0).mul(df_delhi_err_autumn['dilution'], axis=0) / 1000
    df_delhi_data_autumn = df_delhi_data_autumn.div(df_delhi_data_autumn['volume_m3'], axis=0).mul(df_delhi_data_autumn['dilution'], axis=0) / 1000
    df_delhi_err_summer = df_delhi_err_summer.div(df_delhi_err_summer['volume_m3'], axis=0).mul(df_delhi_err_summer['dilution'], axis=0) / 1000
    df_delhi_data_summer = df_delhi_data_summer.div(df_delhi_data_summer['volume_m3'], axis=0).mul(df_delhi_data_summer['dilution'], axis=0) / 1000
    df_delhi_data_autumn.drop(columns=['volume_m3','dilution'],inplace=True)
    df_delhi_err_autumn.drop(columns=['volume_m3','dilution'],inplace=True)
    df_delhi_data_summer.drop(columns=['volume_m3','dilution'],inplace=True)
    df_delhi_err_summer.drop(columns=['volume_m3','dilution'],inplace=True)
    
    #Some final QA
    df_delhi_data_autumn['mid_datetime'] = pd.to_datetime(df_metadata_autumn['mid_datetime'],yearfirst=True)
    df_delhi_err_autumn['mid_datetime'] = pd.to_datetime(df_metadata_autumn['mid_datetime'],yearfirst=True)
    df_delhi_data_summer['mid_datetime'] = pd.to_datetime(df_metadata_summer['mid_datetime'],yearfirst=True)
    df_delhi_err_summer['mid_datetime'] = pd.to_datetime(df_metadata_summer['mid_datetime'],yearfirst=True)
    df_delhi_data_autumn.set_index('mid_datetime',inplace=True)
    df_delhi_err_autumn.set_index('mid_datetime',inplace=True)
    df_delhi_data_summer.set_index('mid_datetime',inplace=True)
    df_delhi_err_summer.set_index('mid_datetime',inplace=True)
    
    
    df_delhi_data_autumn = df_delhi_data_autumn.astype(float)
    df_delhi_err_autumn = df_delhi_err_autumn.astype(float)
    df_delhi_data_summer = df_delhi_data_summer.astype(float)
    df_delhi_err_summer = df_delhi_err_summer.astype(float)
    df_delhi_data_autumn.columns = index_backup_autumn
    df_delhi_err_autumn.columns = index_backup_autumn
    df_delhi_data_summer.columns = index_backup_summer
    df_delhi_err_summer.columns = index_backup_summer
    #pdb.set_trace()
    
    #Make the final output data
    if(output=="DEFAULT" or output=="all"):
        df_delhi_metadata = pd.concat([df_metadata_summer,df_metadata_autumn])
        df_delhi_data = pd.concat([df_delhi_data_summer,df_delhi_data_autumn])
        df_delhi_err = pd.concat([df_delhi_err_summer,df_delhi_err_autumn])
        return df_delhi_data, df_delhi_err, df_delhi_metadata, df_delhi_raw
    elif(output=="summer"):
        df_delhi_raw_summer = pd.concat([df_delhi_raw_chem,df_delhi_raw_summer])
        return df_delhi_data_summer, df_delhi_err_summer, df_metadata_summer, df_delhi_raw_summer
    elif(output=="autumn"):
        df_delhi_raw_autumn = pd.concat([df_delhi_raw_chem,df_delhi_raw_autumn])
        return df_delhi_data_autumn, df_delhi_err_autumn, df_metadata_autumn, df_delhi_raw_autumn
    
    return None



#%%
#Map filter times onto night/morning/midday/afternoon as per Hamilton et al 2021
def delhi_calc_time_cat(df_in):
    dict_hour_to_time_cat =	{
      0: "Night",
      1: "Night",
      2: "Night",
      3: "Night",
      4: "Night",
      5: "Night",
      6: "Night",
      7: "Morning",
      8: "Morning",
      9: "Morning",
      10: "Morning",
      11: "Midday",
      12: "Midday",
      13: "Afternoon",
      14: "Afternoon",
      15: "Afternoon",
      16: "Afternoon",
      17: "Afternoon",
      18: "Afternoon",  
      19: "Afternoon",
      20: "Night",
      21: "Night",
      22: "Night",
      23: "Night",
    }
    
    cat1 = pd.Categorical(df_in.index.hour.to_series().map(dict_hour_to_time_cat).values,categories=['Morning','Midday' ,'Afternoon','Night','24hr'], ordered=True)
    #pdb.set_trace()
    time_length = (df_in['date_end'] - df_in['date_start']) / dt.timedelta(hours=1)
    cat1[time_length>22] = ['24hr']
    
    return cat1

#Map filter times onto night/morning/midday/afternoon as per Hamilton et al 2021
#Also 24hr filters as separate category- this requires a time length column
def calc_time_cat(df_metadata):
    dict_hour_to_time_cat =	{
      0: "Night",
      1: "Night",
      2: "Night",
      3: "Night",
      4: "Night",
      5: "Night",
      6: "Night",
      7: "Morning",
      8: "Morning",
      9: "Morning",
      10: "Morning",
      11: "Midday",
      12: "Midday",
      13: "Afternoon",
      14: "Afternoon",
      15: "Afternoon",
      16: "Afternoon",
      17: "Afternoon",
      18: "Afternoon",
      19: "Afternoon",
      20: "Night",
      21: "Night",
      22: "Night",
      23: "Night",
      24: "Night",
      999: "24hr"
    }
    time_hour = df_metadata['mid_datetime'].dt.hour.copy()
    time_hour[df_metadata['timesampled_h'].ge(24)] = 999
    time_cat = pd.Categorical(time_hour.map(dict_hour_to_time_cat).values,['Morning','Midday' ,'Afternoon','Night','24hr'], ordered=True)   
    return time_cat



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
    #Manually remove 4-nitrophenol as it is a huge peak and domnates Delhi
    #df_orbitrap_peaks.drop(df_orbitrap_peaks[df_orbitrap_peaks["Formula"] == "C6 H5 N O3"].index,inplace=True)

    #Filter out peaks with strange formula
    df_orbitrap_peaks = df_orbitrap_peaks[df_orbitrap_peaks["Formula"].apply(lambda x: filter_by_chemform(x))]
    #Merge compound peaks that have the same m/z and retention time
    #Round m/z to nearest integer and RT to nearest 2, as in 1/3/5/7/9 etc
    #Also remove anything with RT > 20min
        
    df_orbitrap_peaks.drop(df_orbitrap_peaks[df_orbitrap_peaks["RT [min]"] > 20].index, inplace=True)    
    #df.drop(df[df.score < 50].index, inplace=True)
    
    #Join the peaks with the same rounded m/z and RT    
    #RT_round =  df_orbitrap_peaks["RT [min]"].apply(lambda x: round_odd(x))
    #mz_round = df_orbitrap_peaks["m/z"].apply(lambda x: round(x, 2))
    
    #ORIGINAL
    #df_orbitrap_peaks = df_orbitrap_peaks.iloc[:,np.r_[0:4]].groupby([mz_round,RT_round]).aggregate("first").join(df_orbitrap_peaks.iloc[:,np.r_[4:len(df_orbitrap_peaks.columns)]].groupby([mz_round,RT_round]).aggregate("sum") )
    #pdb.set_trace()
    #MERGE SAME MOLECULE AND RT <10 OR >10
    RT_round10 =  df_orbitrap_peaks["RT [min]"].apply(lambda x: above_below_10(x))
    
    #a = df_orbitrap_peaks.iloc[:,np.r_[0:4]].groupby([df_orbitrap_peaks["Formula"],RT_round10]).aggregate("first")
    #b = df_orbitrap_peaks.iloc[:,np.r_[4:len(df_orbitrap_peaks.columns)]].groupby([df_orbitrap_peaks["Formula"],RT_round10]).aggregate("sum")
    
    df_orbitrap_peaks = df_orbitrap_peaks.iloc[:,np.r_[0:4]].groupby([df_orbitrap_peaks["Formula"],RT_round10]).aggregate("first").join(df_orbitrap_peaks.iloc[:,np.r_[4:len(df_orbitrap_peaks.columns)]].groupby([df_orbitrap_peaks["Formula"],RT_round10]).aggregate("sum") )
    
    
    return df_orbitrap_peaks      
    

# #Take a string and work out the chemical formula, then return true or false if it's good or bad   
def filter_by_chemform(formula):
    chemformula = ChemForm(formula)
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
    
#Calculate H:C, O:C, S:C, N:C ratios
#Ratio is element1:element2
def chemform_ratios(formula):
    raise("chemform_ratios is depracated and should no longer be used. Use ChemForm().ratios instead, and note that the order of O/C and N/C is different")
    
    
    
  

  
#######################
####CHEMICAL ANALYSIS#######
#######################
# %%
#Function to extract the top n peaks from a cluster in terms of their chemical formula
#cluster must be a data series with the column indices of the different peaks
#You can use this with df_data.T instead of df_raw
def cluster_extract_peaks(cluster, df_raw,num_peaks,chemform_namelist=pd.DataFrame(),dp=1,dropRT=True,printdf=False):
    #Check they are the same length
    if(cluster.shape[0] != df_raw.shape[0]):
        print("cluster_extract_peaks returning null: cluster and peaks dataframe must have same number of peaks")
        print("Maybe your data needs transposing, or '.mean()' -ing?")
        return np.NaN
        quit()#surely this is redundnt??
    
    nlargest = cluster.nlargest(num_peaks)
    nlargest_pct = nlargest / cluster.sum() * 100
    #pdb.set_trace()
    output_df = pd.DataFrame(index=nlargest.index)
   # nlargest.index = pd.MultiIndex.from_tuples(nlargest.index, names=["first", "second"]) #Make the index multiindex again
    output_df["Formula"] = nlargest.index.get_level_values(0)
    if(dropRT is False):
        output_df["RT"] = nlargest.index.get_level_values(1)
    #output_df.set_index(output_df['Formula'],inplace=True)
    output_df["peak_pct"] = nlargest_pct.round(dp).values
    
    if(chemform_namelist.empty == True):
        output_df["Name"] = output_df["Formula"]
    else:
        overlap_indices = output_df.index.intersection(chemform_namelist.index)
        output_df["Name"] = chemform_namelist.loc[overlap_indices]
        output_df.loc[output_df['Name'].isnull(),'Name'] = output_df['Formula']
        output_df.drop('Formula',axis=1,inplace=True)

    if(printdf == True):
        print(output_df)
        
    return output_df



# def plot_top_ae_loss(df_all_data,ds_AE_loss_per_sample,mz_columns,Sari_peaks_list):
#     num_plots=4
#     index_top_loss= ds_AE_loss_per_sample.nlargest(num_plots).index
    
#     fig,axes = plt.subplots(num_plots,2,figsize=(14,1.5*num_plots),gridspec_kw={'width_ratios': [8, 4]})
#     fig.suptitle('Spectra of top AE loss samples, AE trained on real-space data')
#     for y_idx in np.arange(num_plots):
#         this_cluster_profile = df_all_data.loc[index_top_loss[y_idx]].to_numpy()
#         top_peaks = cluster_extract_peaks(df_all_data.loc[index_top_loss[y_idx]], df_all_raw,10,chemform_namelist_all,dp=1,printdf=False)
#         ax = axes[-y_idx-1][0]
#         ax.stem(mz_columns.to_numpy(),this_cluster_profile,markerfmt=' ')
#         ax.set_xlim(left=100,right=400)
#         ax.set_xlabel('m/z')
#         ax.set_ylabel('Relative concentration')
#         #ax.set_title('Cluster' + str(y_idx))
#         ax.text(0.01, 0.95, 'Sample ' + str(y_idx), transform=ax.transAxes, fontsize=12,
#                 verticalalignment='top')
        
#         #Add in a table with the top peaks
#         #pdb.set_trace()
#         ds_this_cluster_profile = pd.Series(this_cluster_profile,index=df_all_data.columns).T
#         df_top_peaks = cluster_extract_peaks(ds_this_cluster_profile, df_all_raw,10,chemform_namelist_all,dp=1,printdf=False)
#         df_top_peaks.index = df_top_peaks.index.str.replace(' ', '')
#         ax2 = axes[-y_idx-1][1]
#         #pdb.set_trace()
#         cellText = pd.merge(df_top_peaks, Sari_peaks_list, how="left",left_index=True,right_index=True)[['peak_pct','Source']]
#         cellText['Source'] = cellText['Source'].astype(str).replace(to_replace='nan',value='')
#         cellText = cellText.reset_index().values
#         the_table = ax2.table(cellText=cellText,loc='center',cellLoc='left',colLabels=['Formula','%','Potential source'],edges='open',colWidths=[0.3,0.1,0.6])
#         the_table.auto_set_font_size(False)
#         the_table.set_fontsize(11)
#         cells = the_table.properties()["celld"]
#         for i in range(0, 11):
#             cells[i, 1].set_text_props(ha="right")
            
#         plt.tight_layout()


#Function to extract the top n peaks from a cluster in terms of their chemical formula
#cluster must be a data series with the column indices of the different peaks
def plot_orbitrap_top_ae_loss(df_data,mz_columns,ds_AE_loss_per_sample,num_top_losses=1,num_peaks=10,chemform_namelist=pd.DataFrame(),Sari_peaks_list=pd.DataFrame(),dp=1,printdf=False):
    #Check they are the same length
    # if(cluster.shape[0] != df_raw.shape[0]):
    #     print("cluster_extract_peaks returning null: cluster and peaks dataframe must have same number of peaks")
    #     print("Maybe your data needs transposing, or '.mean()' -ing?")
    #     return np.NaN
    #     quit()
    #     #print("WARNING: cluster_extract_peaks(): cluster and peaks dataframe do not have the same number of peaks")
    
    index_top_loss= ds_AE_loss_per_sample.nlargest(num_top_losses).index
    fig,axes = plt.subplots(num_top_losses,2,figsize=(14,2.65*num_top_losses),gridspec_kw={'width_ratios': [8, 4]})
    fig.suptitle('Spectra of top AE loss samples, AE trained on real-space data')
    
    for y_idx in np.arange(num_top_losses):
        if(num_top_losses==1):
            ax = axes[0]
            ax2 = axes[1]
        else:
            ax = axes[y_idx][0]
            ax2 = axes[y_idx][1]
            
        this_cluster_profile = df_data.loc[index_top_loss[y_idx]].to_numpy()
                
        ax.stem(mz_columns.to_numpy(),this_cluster_profile,markerfmt=' ')
        ax.set_xlim(left=100,right=400)
        ax.set_xlabel('m/z')
        ax.set_ylabel('Concentration (what units?)')
        #ax.set_title('Cluster' + str(y_idx))
        ax.text(0.01, 0.95, 'Top loss ' + str(y_idx+1) + ' ' + str(index_top_loss[y_idx]), transform=ax.transAxes, fontsize=12,
                verticalalignment='top')
        
        #Add in a table with the top peaks
        #pdb.set_trace()
        ds_this_cluster_profile = pd.Series(this_cluster_profile,index=df_data.columns).T
        df_top_peaks = cluster_extract_peaks(ds_this_cluster_profile, df_data.T,10,chemform_namelist,dp=1,printdf=False)
        #df_top_peaks.index = df_top_peaks.index.str.replace(' ', '')
        #pdb.set_trace()
        df_top_peaks.index = df_top_peaks.index.str.replace(" ","")
#        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
#        SUB = str.maketranse("2":"\u2082","2":"\u2082")       
        cellText = pd.merge(df_top_peaks, Sari_peaks_list, how="left",left_index=True,right_index=True)[['peak_pct','Source']]
        cellText['Source'] = cellText['Source'].astype(str).replace(to_replace='nan',value='')
        #cellText.index = cellText.index.str.translate(SUB)
        cellText = cellText.reset_index().values
        the_table = ax2.table(cellText=cellText,loc='center',cellLoc='left',colLabels=['Formula','%','Potential source'],edges='open',colWidths=[0.3,0.1,0.6])
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(11)
        cells = the_table.properties()["celld"]
        for i in range(0, 11):
            cells[i, 1].set_text_props(ha="right")
            
        plt.tight_layout()


def format(equation):
    parts = []
    partial = 0
    for char in equation+' ':
        if char.isnumeric():
            partial = partial * 10 + int(char)
            continue

        if partial:
            if parts[-1] == ')':
                parts.pop()
                i = -1
                while parts[i] != '(':
                    parts[i][1] *= partial
                    i -= 1
            else:
                parts[-1][1] = partial
            partial = 0

        if char.isupper():
            parts.append( [char, 1] )
        elif char.islower():
            parts[-1][0] += char
        elif char == '(':
            parts.append('(')
        elif char == ')':
            parts.append(')')
    build = []
    for part in parts:
        print(part)
        if part == '(':
            continue
        if part[1] == 1:
            build.append( part[0] )
        else:
            build.append( part[0] + str(part[1]))
    return ''.join(build)



#Extract the top n peaks in terms of their R correlation
def top_corr_peaks(df_corr,chemform_namelist,num_peaks,dp=2):
    #pdb.set_trace()
    nlargest = df_corr.nlargest(num_peaks)
    
    output_df = pd.DataFrame()
    nlargest.index = pd.MultiIndex.from_tuples(nlargest.index, names=["first", "second"]) #Make the index multiindex again
    
    output_df["Formula"] = nlargest.index.get_level_values(0)

    output_df.set_index(output_df['Formula'],inplace=True)
    output_df["R"] = nlargest.round(dp).values
    overlap_indices = output_df.index.intersection(chemform_namelist.index)
    output_df["Name"] = chemform_namelist.loc[overlap_indices]

    output_df.loc[output_df['Name'].isnull(),'Name'] = output_df['Formula'][0]
    output_df.drop('Formula',axis=1,inplace=True)
       
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

#Combine 2 chemform namelists
def combine_chemform_namelists(namelist1,namelist2):
    chemform_namelist_all = namelist1.append(namelist2)
    
    #Deal with duplicates- if they are identical then only one copy, if they are not them combine then
    duplicates = chemform_namelist_all.duplicated()
    #pdb.set_trace()
    duplicates = duplicates[duplicates] #only true values    
    #Group the duplictaes into one string so there's only one copy to deal with
    chemform_namelist_all = chemform_namelist_all.groupby(['Formula'])['Name'].apply(';'.join)
    
    for molecule in duplicates.index:
        if(namelist1.loc[molecule].values == namelist2.loc[molecule].values):
            chemform_namelist_all.loc[molecule] = namelist1.loc[molecule].str.cat(sep=';')
        else:
            chemform_namelist_all.loc[molecule] = namelist1.loc[molecule].str.cat(sep=';') + ';' + namelist2.loc[molecule].str.cat(sep=';')
    
    #Make them all just one string
    for molecule in chemform_namelist_all.index:
        if(type(chemform_namelist_all.loc[molecule]) == pd.core.series.Series):
            chemform_namelist_all.loc[molecule] = chemform_namelist_all.loc[molecule].str.cat(sep=';')
    
    return chemform_namelist_all


#############################################################
#############################################################
#####CLUSTERING WORKFLOW#####################################
#############################################################
#############################################################
#%%Run Clustering a set number of times
def cluster_n_times(df_data,max_num_clusters,min_num_clusters=1,cluster_type='agglom'):
    num_clusters_array = np.arange(min_num_clusters,max_num_clusters+1)
    cluster_labels_mtx = []
    
    for num_clusters in num_clusters_array:
        #First run the clustering
        if(cluster_type=='agglom'):
            cluster_obj = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
        elif(cluster_type=='kmeans' or cluster_type=='Kmeans' or cluster_type=='KMeans'):
            cluster_obj = KMeans(n_clusters = num_clusters)
        elif(cluster_type=='kmedoids' or cluster_type == 'Kmedoids' or cluster_type=='KMedoids'):
            cluster_obj = KMedoids(n_clusters = num_clusters)
        clustering = cluster_obj.fit(df_data.values)
        #c = relabel_clusters_most_freq(clustering.labels_)
        cluster_labels_mtx.append(clustering.labels_)
        
    df_cluster_labels_mtx = pd.DataFrame(cluster_labels_mtx,index=num_clusters_array).T.rename_axis(columns="num_clusters")
    df_cluster_labels_mtx.index=df_data.index
    return df_cluster_labels_mtx


#Count the number of samples in each cluster, for each number of clusters
def count_cluster_labels_from_mtx(df_cluster_labels_mtx):
    df_cluster_counts_mtx = pd.DataFrame(columns=np.arange(0,df_cluster_labels_mtx.max().max()+1))
    for n_clusters in df_cluster_labels_mtx.columns:
        c =  df_cluster_labels_mtx[n_clusters]
        c.reset_index(inplace=True,drop=True)
        df_cluster_counts_mtx.loc[n_clusters] = c.value_counts()
    
    return df_cluster_counts_mtx.rename_axis(index='num_clusters',columns='cluster_index')


#%%Compare clustering metrics for a given dataset
def compare_cluster_metrics(df_data,min_clusters,max_clusters,cluster_type='agglom',suptitle_prefix='', suptitle_suffix=''):
    num_clusters_index = range(min_clusters,(max_clusters+1),1)
    ch_score = np.empty(len(num_clusters_index))
    db_score = np.empty(len(num_clusters_index))
    silhouette_scores = np.empty(len(num_clusters_index))
    
    for num_clusters in num_clusters_index:
        if(cluster_type=='agglom'):
            cluster_obj = AgglomerativeClustering(n_clusters = num_clusters, linkage = 'ward')
            suptitle_cluster_type = 'HCA'
        elif(cluster_type=='kmeans' or cluster_type=='Kmeans' or cluster_type=='KMeans'):
            cluster_obj = KMeans(n_clusters = num_clusters)
            suptitle_cluster_type = 'KMeans'
        elif(cluster_type=='kmedoids' or cluster_type == 'Kmedoids' or cluster_type=='KMedoids'):
            cluster_obj = KMedoids(n_clusters = num_clusters)
            suptitle_cluster_type = 'KMedoids'
        else:
            raise Exception("Incorrect cluster_type")
        
        clustering = cluster_obj.fit(df_data.values)
        ch_score[num_clusters-min_clusters] = calinski_harabasz_score(df_data.values, clustering.labels_)
        db_score[num_clusters-min_clusters] = davies_bouldin_score(df_data.values, clustering.labels_)
        silhouette_scores[num_clusters-min_clusters] = silhouette_score(df_data.values, clustering.labels_)
        
    #Plot results
    fig,ax1 = plt.subplots(figsize=(10,6))
    ax2=ax1.twinx()
    ax3=ax1.twinx()
    ax3.spines.right.set_position(("axes", 1.1))
    p1, = ax1.plot(num_clusters_index,ch_score,label="CH score")
    p2, = ax2.plot(num_clusters_index,db_score,c='red',label="DB score")
    p3, = ax3.plot(num_clusters_index,silhouette_scores,c='black',label="Silhouette score")
    ax1.set_xlabel("Num clusters")
    ax1.set_ylabel("CH score")
    ax2.set_ylabel("DB score")
    ax3.set_ylabel('Silhouette score')
    
    ax1.yaxis.label.set_color(p1.get_color())
    ax2.yaxis.label.set_color(p2.get_color())
    ax3.yaxis.label.set_color(p3.get_color())
    #pdb.set_trace()
    
    ax1.spines['left'].set_color(p1.get_color())
    ax1.spines.right.set_visible(False)
    ax1.tick_params(axis='y', colors=p1.get_color())
    
    ax2.spines['right'].set_color(p2.get_color())
    ax2.tick_params(axis='y', colors=p2.get_color())
    ax2.spines.left.set_visible(False)
    
    ax3.spines['right'].set_color(p3.get_color())
    ax3.tick_params(axis='y', colors=p3.get_color())
    ax3.spines.left.set_visible(False)
    
    ax1.legend(handles=[p1, p2, p3])
    
    #ax1.xaxis.set_major_locator(plticker.MultipleLocator(base=5.0))
    #ax1.xaxis.set_minor_locator(plticker.MultipleLocator(base=1.0))
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.suptitle(suptitle_prefix + suptitle_cluster_type + suptitle_suffix)
    plt.show()



    
    
    
#%%Calc cluster elemental ratios
def calc_cluster_elemental_ratios(df_cluster_labels_mtx,df_all_data,df_element_ratios):
    df_clusters_HC_mtx = pd.DataFrame(np.NaN, index = df_cluster_labels_mtx.columns, columns = np.arange(df_cluster_labels_mtx.columns.max()))
    df_clusters_NC_mtx = pd.DataFrame(np.NaN, index = df_cluster_labels_mtx.columns, columns = np.arange(df_cluster_labels_mtx.columns.max()))
    df_clusters_OC_mtx = pd.DataFrame(np.NaN, index = df_cluster_labels_mtx.columns, columns = np.arange(df_cluster_labels_mtx.columns.max()))
    df_clusters_SC_mtx = pd.DataFrame(np.NaN, index = df_cluster_labels_mtx.columns, columns = np.arange(df_cluster_labels_mtx.columns.max()))
    
    #index is the number of clusters
    #columns is the cluster in question
    for num_clusters in df_cluster_labels_mtx.columns:
        c = df_cluster_labels_mtx[num_clusters]
        c.index = df_cluster_labels_mtx.index
        for this_cluster in np.arange(num_clusters):
            cluster_sum = df_all_data[c==this_cluster].sum().values
            df_clusters_HC_mtx.loc[num_clusters,this_cluster] = (cluster_sum * df_element_ratios['H/C']).sum() / cluster_sum.sum()
            df_clusters_NC_mtx.loc[num_clusters,this_cluster] = (cluster_sum * df_element_ratios['N/C']).sum() / cluster_sum.sum()
            df_clusters_OC_mtx.loc[num_clusters,this_cluster] = (cluster_sum * df_element_ratios['O/C']).sum() / cluster_sum.sum()
            df_clusters_SC_mtx.loc[num_clusters,this_cluster] = (cluster_sum * df_element_ratios['S/C']).sum() / cluster_sum.sum()
    return df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx

#%%Calc cluster AMS averages
def calc_cluster_AMS_means(df_cluster_labels_mtx,df_AQ_all):
    df_clusters_AMS_NO3_mtx = pd.DataFrame(np.NaN, index = df_cluster_labels_mtx.columns, columns = np.arange(df_cluster_labels_mtx.columns.max()))
    df_clusters_AMS_SO4_mtx = pd.DataFrame(np.NaN, index = df_cluster_labels_mtx.columns, columns = np.arange(df_cluster_labels_mtx.columns.max()))
    df_clusters_AMS_NH4_mtx = pd.DataFrame(np.NaN, index = df_cluster_labels_mtx.columns, columns = np.arange(df_cluster_labels_mtx.columns.max()))
    df_clusters_AMS_Org_mtx = pd.DataFrame(np.NaN, index = df_cluster_labels_mtx.columns, columns = np.arange(df_cluster_labels_mtx.columns.max()))
    df_clusters_AMS_Chl_mtx = pd.DataFrame(np.NaN, index = df_cluster_labels_mtx.columns, columns = np.arange(df_cluster_labels_mtx.columns.max()))
    df_clusters_AMS_Total_mtx = pd.DataFrame(np.NaN, index = df_cluster_labels_mtx.columns, columns = np.arange(df_cluster_labels_mtx.columns.max()))
    
    #index is the number of clusters
    #columns is the cluster in question
    for num_clusters in df_cluster_labels_mtx.columns:
        c = df_cluster_labels_mtx[num_clusters]
        for this_cluster in np.arange(num_clusters):
            df_clusters_AMS_NO3_mtx.loc[num_clusters,this_cluster] = df_AQ_all['AMS_NO3'][c==this_cluster].mean()
            df_clusters_AMS_SO4_mtx.loc[num_clusters,this_cluster] = df_AQ_all['AMS_SO4'][c==this_cluster].mean()
            df_clusters_AMS_NH4_mtx.loc[num_clusters,this_cluster] = df_AQ_all['AMS_NH4'][c==this_cluster].mean()
            df_clusters_AMS_Org_mtx.loc[num_clusters,this_cluster] = df_AQ_all['AMS_Org'][c==this_cluster].mean()
            df_clusters_AMS_Chl_mtx.loc[num_clusters,this_cluster] = df_AQ_all['AMS_Chl'][c==this_cluster].mean()
            df_clusters_AMS_Total_mtx.loc[num_clusters,this_cluster] = df_AQ_all['AMS_Total'][c==this_cluster].mean()
            
    return df_clusters_AMS_NO3_mtx,df_clusters_AMS_SO4_mtx,df_clusters_AMS_NH4_mtx,df_clusters_AMS_Org_mtx,df_clusters_AMS_Chl_mtx,df_clusters_AMS_Total_mtx

def calc_cluster_AMS_frac(df_cluster_labels_mtx,df_AQ_all):
    df_clusters_AMS_NO3_frac_mtx = pd.DataFrame(np.NaN, index = df_cluster_labels_mtx.columns, columns = np.arange(df_cluster_labels_mtx.columns.max()))
    df_clusters_AMS_SO4_frac_mtx = pd.DataFrame(np.NaN, index = df_cluster_labels_mtx.columns, columns = np.arange(df_cluster_labels_mtx.columns.max()))
    df_clusters_AMS_NH4_frac_mtx = pd.DataFrame(np.NaN, index = df_cluster_labels_mtx.columns, columns = np.arange(df_cluster_labels_mtx.columns.max()))
    df_clusters_AMS_Org_frac_mtx = pd.DataFrame(np.NaN, index = df_cluster_labels_mtx.columns, columns = np.arange(df_cluster_labels_mtx.columns.max()))
    df_clusters_AMS_Chl_frac_mtx = pd.DataFrame(np.NaN, index = df_cluster_labels_mtx.columns, columns = np.arange(df_cluster_labels_mtx.columns.max()))
    
    
    #index is the number of clusters
    #columns is the cluster in question
    for num_clusters in df_cluster_labels_mtx.columns:
        c = df_cluster_labels_mtx[num_clusters]
        for this_cluster in np.arange(num_clusters):
            df_clusters_AMS_NO3_frac_mtx.loc[num_clusters,this_cluster] = df_AQ_all['AMS_NO3_frac'][c==this_cluster].mean()
            df_clusters_AMS_SO4_frac_mtx.loc[num_clusters,this_cluster] = df_AQ_all['AMS_SO4_frac'][c==this_cluster].mean()
            df_clusters_AMS_NH4_frac_mtx.loc[num_clusters,this_cluster] = df_AQ_all['AMS_NH4_frac'][c==this_cluster].mean()
            df_clusters_AMS_Org_frac_mtx.loc[num_clusters,this_cluster] = df_AQ_all['AMS_Org_frac'][c==this_cluster].mean()
            df_clusters_AMS_Chl_frac_mtx.loc[num_clusters,this_cluster] = df_AQ_all['AMS_Chl_frac'][c==this_cluster].mean()
            
    return df_clusters_AMS_NO3_frac_mtx,df_clusters_AMS_SO4_frac_mtx,df_clusters_AMS_NH4_frac_mtx,df_clusters_AMS_Org_frac_mtx,df_clusters_AMS_Chl_frac_mtx






#%%Plot cluster elemental ratios
def plot_cluster_elemental_ratios(df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,suptitle):
    #Make X and Y for plotting
    
    #X = np.tile(df_clusters_HC_mtx.columns,(df_clusters_HC_mtx.shape[0],1)).T.ravel()
    #Y = np.tile(df_clusters_HC_mtx.index,df_clusters_HC_mtx.shape[1]) 
    
    X = np.arange(df_clusters_HC_mtx.index.min(),df_clusters_HC_mtx.index.max()+2) - 0.5
    Y = np.arange(df_clusters_HC_mtx.columns.min(),df_clusters_HC_mtx.columns.max()+2) - 0.5
    
    cmap = cmap_EOS11()
    
    fig,ax = plt.subplots(2,2,figsize=(12,8))
    ax = ax.ravel()
    plot0 = ax[0].pcolor(X,Y,df_clusters_HC_mtx.T,cmap=cmap)
    ax[0].set_xlabel('Num clusters')
    ax[0].set_ylabel('Cluster index')
    plt.colorbar(plot0, label='H/C',ax=ax[0])
    ax[0].set_title('H/C ratio')
    
    
    plot1 = ax[1].pcolor(X,Y,df_clusters_NC_mtx.T,cmap=cmap)
    ax[1].set_xlabel('Num clusters')
    ax[1].set_ylabel('Cluster index')
    plt.colorbar(plot1, label='N/C',ax=ax[1])
    ax[1].set_title('N/C ratio')
    
    plot2 = ax[2].pcolor(X,Y,df_clusters_OC_mtx.T,cmap=cmap)
    ax[2].set_xlabel('Num clusters')
    ax[2].set_ylabel('Cluster index')
    plt.colorbar(plot2, label='O/C',ax=ax[2])
    ax[2].set_title('O/C ratio')
    
    plot3 = ax[3].pcolor(X,Y,df_clusters_SC_mtx.T,cmap=cmap)
    ax[3].set_xlabel('Num clusters')
    ax[3].set_ylabel('Cluster index')
    ax[3].set_ylabel('Cluster index')
    ax[3].set_title('S/C ratio')
    plt.colorbar(plot3, label='S/C',ax=ax[3])
    
    fig.suptitle(suptitle)
    fig.subplots_adjust(top=0.88)
    fig.tight_layout()
    plt.show()
    
    return

#%%Plot cluster AMS Means
def plot_cluster_AMS_means(df_clusters_AMS_NO3_mtx,df_clusters_AMS_SO4_mtx,df_clusters_AMS_NH4_mtx,df_clusters_AMS_Org_mtx,df_clusters_AMS_Chl_mtx,df_clusters_AMS_Total_mtx,suptitle):
    #Make X and Y for plotting
    
    #X = np.tile(df_clusters_HC_mtx.columns,(df_clusters_HC_mtx.shape[0],1)).T.ravel()
    #Y = np.tile(df_clusters_HC_mtx.index,df_clusters_HC_mtx.shape[1]) 
    
    X = np.arange(df_clusters_AMS_NO3_mtx.index.min(),df_clusters_AMS_NO3_mtx.index.max()+2) - 0.5
    Y = np.arange(df_clusters_AMS_NO3_mtx.columns.min(),df_clusters_AMS_NO3_mtx.columns.max()+2) - 0.5
    
    cmap = cmap_EOS11()
    
    fig,ax = plt.subplots(2,3,figsize=(12,8))
    ax = ax.ravel()
    plot0 = ax[0].pcolor(X,Y,df_clusters_AMS_NO3_mtx.T,cmap=cmap)
    ax[0].set_xlabel('Num clusters')
    ax[0].set_ylabel('Cluster index')
    plt.colorbar(plot0, label='µg/m3',ax=ax[0])
    ax[0].set_title('AMS NO3')
    
    plot1 = ax[1].pcolor(X,Y,df_clusters_AMS_SO4_mtx.T,cmap=cmap)
    ax[1].set_xlabel('Num clusters')
    ax[1].set_ylabel('Cluster index')
    plt.colorbar(plot1, label='µg/m3',ax=ax[1])
    ax[1].set_title('AMS SO4')
    
    plot2 = ax[2].pcolor(X,Y,df_clusters_AMS_NH4_mtx.T,cmap=cmap)
    ax[2].set_xlabel('Num clusters')
    ax[2].set_ylabel('Cluster index')
    plt.colorbar(plot2, label='µg/m3',ax=ax[2])
    ax[2].set_title('AMS NH4')
    
    plot3 = ax[3].pcolor(X,Y,df_clusters_AMS_Org_mtx.T,cmap=cmap)
    ax[3].set_xlabel('Num clusters')
    ax[3].set_ylabel('Cluster index')
    ax[3].set_ylabel('Cluster index')
    ax[3].set_title('AMS Org')
    plt.colorbar(plot3, label='µg/m3',ax=ax[3])
    
    plot3 = ax[4].pcolor(X,Y,df_clusters_AMS_Chl_mtx.T,cmap=cmap)
    ax[3].set_xlabel('Num clusters')
    ax[3].set_ylabel('Cluster index')
    ax[3].set_ylabel('Cluster index')
    ax[3].set_title('AMS Chl')
    plt.colorbar(plot3, label='µg/m3',ax=ax[4])
    
    plot3 = ax[5].pcolor(X,Y,df_clusters_AMS_Total_mtx.T,cmap=cmap)
    ax[3].set_xlabel('Num clusters')
    ax[3].set_ylabel('Cluster index')
    ax[3].set_ylabel('Cluster index')
    ax[3].set_title('AMS Total')
    plt.colorbar(plot3, label='µg/m3',ax=ax[5])
    
    fig.suptitle(suptitle)
    fig.subplots_adjust(top=0.88)
    fig.tight_layout()
    plt.show()
    
    return 1


#%% Average the cluster profiles
def average_cluster_profiles(df_cluster_labels_mtx,df_data): 
    cluster_profiles_mtx = np.empty((df_cluster_labels_mtx.shape[1],df_cluster_labels_mtx.columns.max(),df_data.shape[1]))
    cluster_profiles_mtx.fill(np.NaN)
    cluster_profiles_mtx_norm = cluster_profiles_mtx.copy()
    
    num_clusters_index = df_cluster_labels_mtx.columns.values
    cluster_index = np.arange(df_cluster_labels_mtx.columns.max())
    
    #Dimensions goes like [num_clusters_index,cluster_index,molecule]
    #where num_clusters_index is from df_cluster_labels_mtx.columns
    
    #index is the number of clusters
    #columns is the cluster in question
    for num_clusters in num_clusters_index:
        c = df_cluster_labels_mtx[num_clusters]
        for this_cluster in np.arange(num_clusters):
            #Check if the indices match
            if c.index.equals(df_data.index):
                cluster_sum = df_data[c==this_cluster].mean(axis=0)
            else:   #reset both indices
                cluster_sum = df_data.reset_index(drop=True)[c.reset_index(drop=True)==this_cluster].mean(axis=0)
            #pdb.set_trace()    
            cluster_profiles_mtx[(num_clusters-df_cluster_labels_mtx.columns[0]),this_cluster,:] = cluster_sum
            cluster_profiles_mtx_norm[(num_clusters-df_cluster_labels_mtx.columns[0]),this_cluster,:] = cluster_sum / cluster_sum.sum()
    
    return cluster_profiles_mtx, cluster_profiles_mtx_norm, num_clusters_index, cluster_index


#%%Correlate cluster mass spectral profiles

def correlate_cluster_profiles(cluster_profiles_mtx_norm, num_clusters_index, cluster_index):
    
    df_cluster_corr_mtx = pd.DataFrame(np.NaN, index = num_clusters_index, columns = cluster_index)
    df_prevcluster_corr_mtx = pd.DataFrame(np.NaN, index = num_clusters_index, columns = cluster_index)
    

    #index is the number of clusters
    #columns is the cluster in question
    for x_idx in np.arange(cluster_profiles_mtx_norm.shape[0]):
        num_clusters = num_clusters_index[x_idx]
        #print(num_clusters)
        if(num_clusters>1):
            for this_cluster in np.arange(num_clusters):
                #Find correlations with other clusters from the same num_clusters
                this_cluster_profile = cluster_profiles_mtx_norm[x_idx,this_cluster,:]  
                other_clusters_profiles = cluster_profiles_mtx_norm[x_idx,cluster_index!=this_cluster,:]
                profiles_corr = np.zeros(other_clusters_profiles.shape[0])
                for y_idx in np.arange(num_clusters-1):
                    this_other_cluster_profile = other_clusters_profiles[y_idx,:]
                    profiles_corr[y_idx] = pearsonr(this_cluster_profile,this_other_cluster_profile)[0]
                df_cluster_corr_mtx.loc[num_clusters,this_cluster] = profiles_corr.max()
        if(num_clusters>1 and x_idx > 0):
            for this_cluster in np.arange(num_clusters):
                #Find correlations with the clusters from the previous num_clusters
                this_cluster_profile = cluster_profiles_mtx_norm[x_idx,this_cluster,:]  
                prev_clusters_profiles = cluster_profiles_mtx_norm[x_idx-1,:,:]
                profiles_corr = np.zeros(prev_clusters_profiles.shape[0])
                for y_idx in np.arange(num_clusters-1):
                    this_prev_cluster_profile = prev_clusters_profiles[y_idx,:]
                    #pdb.set_trace()
                    profiles_corr[y_idx] = pearsonr(this_cluster_profile,this_prev_cluster_profile)[0]
                df_prevcluster_corr_mtx.loc[num_clusters,this_cluster] = profiles_corr.max()
            
            
    return df_cluster_corr_mtx, df_prevcluster_corr_mtx


#%%Plot cluster profile correlations
def plot_cluster_profile_corrs(df_cluster_corr_mtx, df_prevcluster_corr_mtx,suptitle=''):
    #Make X and Y for plotting
        
    X = np.arange(df_cluster_corr_mtx.index.min(),df_cluster_corr_mtx.index.max()+2) - 0.5
    Y = np.arange(df_cluster_corr_mtx.columns.min(),df_cluster_corr_mtx.columns.max()+2) - 0.5
    cmap = cmap_EOS11()
    
    fig,ax = plt.subplots(1,2,figsize=(12,5))
    ax = ax.ravel()
    plot0 = ax[0].pcolor(X,Y,df_cluster_corr_mtx.T,cmap=cmap)
    ax[0].set_xlabel('Num clusters')
    ax[0].set_ylabel('Cluster index')
    ax[0].set_title('Highest R with other clusters')
    plt.colorbar(plot0, label='Pearson\'s R',ax=ax[0])
    
    plot1 = ax[1].pcolor(X,Y,df_prevcluster_corr_mtx.T,cmap=cmap)
    ax[1].set_xlabel('Num clusters')
    ax[1].set_ylabel('Cluster index')
    ax[1].set_title('Highest R with other clusters from previous num clusters')
    plt.colorbar(plot1, label='Pearson\'s R',ax=ax[1])
    fig.suptitle(suptitle)
    
    
#Correlate each row with every other row
#My version that is slow but actually works for matrices of different number of rows
def corr_coeff_rowwise_loops(A,B):
    corr_mtx = np.empty([A.shape[0],B.shape[0]])
    for i in range(A.shape[0]):
       #pdb.set_trace()
        Arow = A[i,:]
        for j in range(B.shape[0]):
            Brow = B[j,:]
            #pdb.set_trace()
            corr_mtx[i,j] = pearsonr(Arow,Brow)[0]
    return corr_mtx


#%%Plot cluster profiles
def plot_all_cluster_profiles(df_all_data,cluster_profiles_mtx_norm, num_clusters_index,mz_columns,df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,df_cluster_corr_mtx,df_prevcluster_corr_mtx,df_cluster_counts_mtx=pd.DataFrame(),peaks_list=pd.DataFrame(columns=['Source']),title_prefix=''):
    for num_clusters in num_clusters_index:
        suptitle = title_prefix + str(int(num_clusters)) + ' clusters'
        plot_one_cluster_profile(df_all_data,cluster_profiles_mtx_norm, num_clusters_index,num_clusters, mz_columns,df_clusters_HC_mtx,df_clusters_NC_mtx,df_clusters_OC_mtx,df_clusters_SC_mtx,df_cluster_corr_mtx,df_prevcluster_corr_mtx,df_cluster_counts_mtx,peaks_list,suptitle)
    
            
def plot_one_cluster_profile(df_all_data,cluster_profiles_mtx_norm, num_clusters_index, num_clusters, mz_columns,
                             df_clusters_HC_mtx=pd.DataFrame(),df_clusters_NC_mtx=pd.DataFrame(),
                             df_clusters_OC_mtx=pd.DataFrame(),df_clusters_SC_mtx=pd.DataFrame(),
                             df_cluster_corr_mtx=pd.DataFrame(),df_prevcluster_corr_mtx=pd.DataFrame(),
                             df_cluster_counts_mtx=pd.DataFrame(),peaks_list=pd.DataFrame(columns=['Source']),suptitle=''):
    
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
    
    fig,axes = plt.subplots(num_clusters,2,figsize=(14,2.9*num_clusters),gridspec_kw={'width_ratios': [8, 4]},constrained_layout=True)
    fig.suptitle(suptitle)
    #cluster_profiles_2D = cluster_profiles_mtx_norm[x_idx,:,:]
    for y_idx in np.arange(num_clusters):
        this_cluster_profile = cluster_profiles_mtx_norm[x_idx,y_idx,:]
        ax = axes[-y_idx-1][0]
        ax.stem(mz_columns.to_numpy(),this_cluster_profile,markerfmt=' ')
        ax.set_xlim(left=100,right=400)
        ax.set_xlabel('m/z')
        ax.set_ylabel('Relative concentration')
        #ax.set_title('Cluster' + str(y_idx))
        ax.text(0.01, 0.95, 'Cluster ' + str(y_idx), transform=ax.transAxes, fontsize=12,
                verticalalignment='top')
        
        #Add in elemental ratios
        if(df_clusters_HC_mtx.empty == False ):
            ax.text(0.69, 0.95, 'H/C = ' + "{:.2f}".format(df_clusters_HC_mtx.loc[num_clusters][y_idx]), transform=ax.transAxes, fontsize=12,
                    verticalalignment='top')
        if(df_clusters_NC_mtx.empty == False ):
            ax.text(0.84, 0.95, 'N/C = ' + "{:.3f}".format(df_clusters_NC_mtx.loc[num_clusters][y_idx]), transform=ax.transAxes, fontsize=12,
                    verticalalignment='top')
        if(df_clusters_OC_mtx.empty == False ):
            ax.text(0.69, 0.85, 'O/C = ' + "{:.2f}".format(df_clusters_OC_mtx.loc[num_clusters][y_idx]), transform=ax.transAxes, fontsize=12,
                    verticalalignment='top')
        if(df_clusters_SC_mtx.empty == False ):
            ax.text(0.84, 0.85, 'S/C = ' + "{:.3f}".format(df_clusters_SC_mtx.loc[num_clusters][y_idx]), transform=ax.transAxes, fontsize=12,
                    verticalalignment='top')
        
        #Add in number of data points for this cluster
        if(df_cluster_counts_mtx.empty == False):
            #Find num samples in this cluster
            num_samples_this_cluster = df_cluster_counts_mtx.loc[num_clusters][y_idx]
            if(num_samples_this_cluster==1):
                ax.text(0.69, 0.75, str(int(num_samples_this_cluster)) + ' sample', transform=ax.transAxes, fontsize=12,
                verticalalignment='top')
            else:
                ax.text(0.69, 0.75, str(int(num_samples_this_cluster)) + ' samples', transform=ax.transAxes, fontsize=12,
                verticalalignment='top')
            
        #Add in best correlation
        if(df_cluster_corr_mtx.empty == False ):
            #Find best cluster correlation
            best_R = df_cluster_corr_mtx.loc[num_clusters][y_idx]
            ax.text(0.69, 0.65, 'Highest R = ' + str(round(best_R,2) ), transform=ax.transAxes, fontsize=12,
                    verticalalignment='top')
        if(df_prevcluster_corr_mtx.empty == False):
            #Find best cluster correlation
            best_R_prev = df_prevcluster_corr_mtx.loc[num_clusters][y_idx]
            if(best_R_prev < 0.9999999):
                ax.text(0.69, 0.55, 'Highest R_prev = ' + str(round(best_R_prev,2) ), transform=ax.transAxes, fontsize=12,
                    verticalalignment='top')
        
        
        
    
        #Add in a table with the top peaks
        ds_this_cluster_profile = pd.Series(this_cluster_profile,index=df_all_data.columns).T
        df_top_peaks = cluster_extract_peaks(ds_this_cluster_profile, df_all_data.T,10,dp=1,dropRT=False)
        #pdb.set_trace()
        df_top_peaks.index = df_top_peaks.index.get_level_values(0).str.replace(' ', '')
        ax2 = axes[-y_idx-1][1]
        cellText = pd.merge(df_top_peaks, peaks_list, how="left",left_index=True,right_index=True)[['RT','peak_pct','Source']]
        cellText.sort_values('peak_pct',inplace=True,ascending=False)
        cellText['Source'] = cellText['Source'].astype(str).replace(to_replace='nan',value='')
        cellText = cellText.reset_index().values
        the_table = ax2.table(cellText=cellText,loc='center',
                              colLabels=['Formula','RT','%','Potential source'],
                              cellLoc = 'left',
                              colLoc = 'left',
                              edges='open',colWidths=[0.3,0.1,0.1,0.5])
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(11)
        cells = the_table.properties()["celld"]
        #pdb.set_trace()
        #Set alignment of column headers
        cells[0,1].set_text_props(ha="right")
        cells[0,2].set_text_props(ha="right")
        #Set alignment of cells
        for i in range(1, 11):
            cells[i, 1].set_text_props(ha="right")
            cells[i, 2].set_text_props(ha="right")
        
        
        #the_table.scale(1, 1.5)  # may help
        #plt.tight_layout()
        
    
    plt.show()


#%%

        
        
#%%Count clusters by project and time
def count_clusters_project_time(df_cluster_labels_mtx,ds_dataset_cat,ds_time_cat,title_prefix='',title_suffix=''):
    for num_clusters in df_cluster_labels_mtx.columns:
        c = df_cluster_labels_mtx[num_clusters]
        a = pd.DataFrame(c.values,columns=['clust'],index=ds_dataset_cat.index)
        a1 = pd.DataFrame(c.values,columns=['clust'],index=ds_time_cat.index)
        #b = df_dataset_cat

        #IF THIS FAILS ITS BECAUSE IT NEEDS DF NOT DS
        df_clust_cat_counts = a.groupby(ds_dataset_cat)['clust'].value_counts(normalize=True).unstack()
        df_cat_clust_counts = ds_dataset_cat.groupby(a['clust']).value_counts(normalize=True).unstack()
        df_clust_time_cat_counts = a1.groupby(ds_time_cat)['clust'].value_counts(normalize=True).unstack()
        df_time_cat_clust_counts = ds_time_cat.groupby(a1['clust']).value_counts(normalize=True).unstack()

        #Previous version, area plot and bar plot
        # fig,ax = plt.subplots(2,2,figsize=(9,10))
        # ax = ax.ravel()
        # plt0 = df_clust_cat_counts.plot.area(ax=ax[0],colormap='tab20')
        # df_cat_clust_counts.plot.bar(ax=ax[1],stacked=True,colormap='RdBu',width=0.8)
        # df_clust_time_cat_counts.plot.area(ax=ax[2],colormap='tab20',legend=False)
        # df_time_cat_clust_counts.plot.bar(ax=ax[3],stacked=True,colormap='PuOr',width=0.8)
        # suptitle = title_prefix + str(num_clusters) + ' clusters' + title_suffix
        # plt.suptitle(suptitle)
        # ax[0].set_ylabel('Fraction')
        # ax[1].set_ylabel('Fraction')
        # ax[0].set_xlabel('')
        # ax[2].set_ylabel('Fraction')
        # ax[2].set_xlabel('')
        # ax[3].set_xlabel('Cluster number')
        # ax[3].set_ylabel('Fraction')
        # ax[1].set_xlabel('Cluster number')
        
        fig,ax = plt.subplots(2,2,figsize=(9,10))
        ax = ax.ravel()
        plt0 = df_clust_cat_counts.plot.bar(ax=ax[0],colormap='viridis',stacked=True)
        df_cat_clust_counts.plot.bar(ax=ax[1],stacked=True,colormap='RdBu',width=0.8)
        df_clust_time_cat_counts.plot.bar(ax=ax[2],colormap='viridis',legend=False,stacked=True)
        df_time_cat_clust_counts.plot.bar(ax=ax[3],stacked=True,colormap='PuOr',width=0.8)
        suptitle = title_prefix + str(num_clusters) + ' clusters' + title_suffix
        plt.suptitle(suptitle)
        ax[0].set_ylabel('Fraction')
        ax[1].set_ylabel('Fraction')
        ax[0].set_xlabel('')
        ax[2].set_ylabel('Fraction')
        ax[2].set_xlabel('')
        ax[3].set_xlabel('Cluster number')
        ax[3].set_ylabel('Fraction')
        ax[1].set_xlabel('Cluster number')


        handles, labels = ax[0].get_legend_handles_labels()
        ax[0].legend(reversed(handles), reversed(labels),title='Cluster number', bbox_to_anchor=(0.5, -0.43),loc='lower center',ncol=5)
        handles, labels = ax[1].get_legend_handles_labels()
        ax[1].legend(reversed(handles), reversed(labels), bbox_to_anchor=(0.5, -0.3),loc='lower center',ncol=2)
        handles, labels = ax[3].get_legend_handles_labels()
        ax[3].legend(reversed(handles), reversed(labels), bbox_to_anchor=(0.5, -0.27),loc='lower center',ncol=3)
        #ax[0].set_xticks(ax[0].get_xticks(), ax[0].get_xticklabels(), rotation=60)
        ax[0].tick_params(axis='x', labelrotation=25)
        ax[2].tick_params(axis='x', labelrotation=25)
        plt.tight_layout()
        plt.show()

    return df_clust_cat_counts, df_cat_clust_counts, df_clust_time_cat_counts, df_time_cat_clust_counts


#%%Top feature explorer
#kwargs: logx, suptitle, supxlabel, supylabel, feature_labels
def top_features_hist(input_data,num_features,figsize='DEFAULT',num_bins=25,**kwargs):
    if str(figsize) == 'DEFAULT':
        figsize=(12,10)
        
    if "logx" in kwargs:
        logx = True
    else:
        logx = False
    
    if(logx==True):
        df_input = pd.DataFrame(input_data).clip(lower=0.01)
    else:
        df_input = pd.DataFrame(input_data)
    
    #Catch if more features requested than there are features in the data
    if(num_features > input_data.shape[1]):
         num_features = input_data.shape[1]
    
    peaks_sum = df_input.sum()
    index_top_features = peaks_sum.nlargest(num_features).index
    
    
    cols = round(math.sqrt(num_features))
    rows = cols
    while rows * cols < num_features:
        rows += 1
    fig, ax_arr = plt.subplots(rows, cols,figsize=figsize)
    
    if(logx==True):
        fig.suptitle('Logscale histograms of top ' + str(num_features) + ' features',size=14)
    else:
        fig.suptitle('Histograms of top ' + str(num_features) + ' features',size=14)
    ax_arr = ax_arr.reshape(-1)
    
    
    for i in range(len(ax_arr)):
        if i >= num_features:
            ax_arr[i].axis('off')
        else:
            data = df_input.iloc[:,index_top_features[i]]
            if(logx==True):
                logbins = np.logspace(np.log10(data.min()),np.log10(data.max()),num_bins)
                ax_arr[i].hist(data,bins=logbins)
                ax_arr[i].set_xscale('log')
            else:
                ax_arr[i].hist(data,bins=num_bins)
                
            if "feature_labels" in kwargs:
                feature_labels = kwargs['feature_labels']
                label = feature_labels[index_top_features[i]]
                ax_arr[i].set_title(label)
    
    if "suptitle" in kwargs:
        fig.suptitle(kwargs['suptitle'])
    if "supxlabel" in kwargs:
        fig.supxlabel(kwargs['supxlabel'])   
    if "supylabel" in kwargs:
        fig.supylabel(kwargs['supylabel'])           
    plt.tight_layout()
    plt.show()




#%%Compare cluster profiles
#First gather cluster profiles
#Start with a few clusters BUT I think it's very ambiguous how many to use going forward. Nbclust and n_clust in R were no help
#Maybe try 8 eventually??? I guess you don't have to use the same number for each dataset

#This will have a thing where you get the uncentred R for each cluster compared to each other cluster

#labels1 & labels 2 should be numpy arrays
#input data should be numpy
def correlate_clusters(labels1,labels2,input_data,norm=True,sub_from_mean=False):
    if(sub_from_mean==True):
        norm=True
        datamean = input_data.mean(axis=0)
        datamean = datamean / (datamean.sum(axis=0,keepdims=1))
        
    labels1 = relabel_clusters_most_freq(labels1)
    labels2 = relabel_clusters_most_freq(labels2)
    
    num_clusters_1 = len(np.unique(labels1))
    num_clusters_2 = len(np.unique(labels2))
    
    correlation_matrix = np.ndarray([num_clusters_1,num_clusters_2])
    for i in np.arange(num_clusters_1):
        for j in np.arange(num_clusters_2):
            cluster1_i = input_data[labels1==i].mean(axis=0)
            cluster2_j = input_data[labels2==j].mean(axis=0)
            
            if(norm==True):
                cluster1_i = cluster1_i/(cluster1_i.sum(axis=0,keepdims=1))
                cluster2_j = cluster2_j/(cluster2_j.sum(axis=0,keepdims=1))
            if(sub_from_mean==True):
                cluster1_i = cluster1_i - datamean
                cluster2_j = cluster2_j - datamean
                #pdb.set_trace()
                #Standard centred Pearson's R, because there could be negative data, it does not sum to zero. Or does it?
                correlation_matrix[i][j] = pearsonr(cluster1_i,cluster2_j)[0]
                
            else:
                #Uncentred R aka normalised dot product
                correlation_matrix[i][j] = np.dot(cluster1_i,cluster2_j) / np.sqrt( np.dot(cluster1_i,cluster1_i) * np.dot(cluster2_j,cluster2_j)   )
    
    return correlation_matrix


def summarise_cluster_comparison(input_data1,input_data2,nclust1,nclust2_min,nclust2_max,norm=True,sub_from_mean=False):
    #Clustering for input_data1
    agglom1 = AgglomerativeClustering(n_clusters = nclust1, linkage = 'ward')
    clustering1 = agglom1.fit(input_data1)
    #pdb.set_trace()
    max_R_mtx = np.empty([(nclust2_max),(nclust2_max-nclust2_min+1)])
    max_R_mtx.fill(np.nan)
    df_max_R_mtx = pd.DataFrame(max_R_mtx, index=np.arange(nclust2_max).tolist(), columns=np.arange(nclust2_min,nclust2_max+1).tolist())
    df_max_R_mtx.rename_axis(index='cluster_index', columns="num_clusters",inplace=True)
    df_clust_freq_mtx = df_max_R_mtx.copy()
    
    
    for nclust2 in range(nclust2_min,nclust2_max+1):
        agglom2 = AgglomerativeClustering(n_clusters = nclust2, linkage = 'ward')
        clustering2 = agglom2.fit(input_data2)    
        correlation_matrix = correlate_clusters(clustering1.labels_,clustering2.labels_,input_data1,norm=norm,sub_from_mean=sub_from_mean)
        #pdb.set_trace()
        df_max_R_mtx[nclust2].iloc[0:nclust2] = correlation_matrix.max(axis=0)
        
        #Number of data points for each cluster
        df_clust_freq_mtx[nclust2].iloc[0:nclust2] = np.unique(relabel_clusters_most_freq(clustering2.labels_), return_counts=True)[1]
    
    
    return df_max_R_mtx, df_clust_freq_mtx

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
    if(x>10):
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


#Find correlation pearson's R for columns in 2 matrices
def corr_2df(df1,df2):
    #pdb.set_trace()
    #Check for columns with duplicate names
    df_duplicates = df1[df1.columns.intersection(df2.columns)]
    if(df_duplicates.shape[1] > 0):
        print("corr_2df returning None. Duplicate column names found in df1 and df2. Duplicates are:")
        print(df_duplicates.columns)
        print("")
        return None
    else:
        return pd.concat([df1, df2], axis=1).corr().filter(df2.columns).filter(df1.columns, axis=0)


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

