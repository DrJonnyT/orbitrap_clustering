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
import re


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
    pdb.set_trace()
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

#Map filter times onto a categorical for what dataset it is
def delhi_beijing_datetime_cat(df_in):
    #Make a numerical flag for each dataset based on the datetime
    datetimecat_num = pd.Series(index=df_in.index)
    for index in df_in.index:
        if((index >= dt.datetime(2016,11,10)) and (index < dt.datetime(2016,12,10))):
            datetimecat_num.loc[index] = 0
        elif((index >= dt.datetime(2017,5,18)) and (index < dt.datetime(2017,6,26))):
            datetimecat_num.loc[index] = 1    
        elif((index >= dt.datetime(2018,5,28)) and (index < dt.datetime(2018,6,6))):
            datetimecat_num.loc[index] = 2
        elif((index >= dt.datetime(2018,10,9)) and (index < dt.datetime(2018,11,7))):
            datetimecat_num.loc[index] = 3
    
    #A dictionary to map the numerical onto a categoricla
    dict_datetime_to_cat =	{
          0: "Beijing_winter",
          1: "Beijing_summer",
          2: "Delhi_summer",
          3: "Delhi_autumn",
        }
    return pd.Categorical(datetimecat_num.map(dict_datetime_to_cat).values,['Beijing_winter','Beijing_summer' ,'Delhi_summer','Delhi_autumn'], ordered=True)


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
    pdb.set_trace()
    #MERGE SAME MOLECULE AND RT <10 OR >10
    RT_round10 =  df_orbitrap_peaks["RT [min]"].apply(lambda x: above_below_10(x))
    
    a = df_orbitrap_peaks.iloc[:,np.r_[0:4]].groupby([df_orbitrap_peaks["Formula"],RT_round10]).aggregate("first")
    b = df_orbitrap_peaks.iloc[:,np.r_[4:len(df_orbitrap_peaks.columns)]].groupby([df_orbitrap_peaks["Formula"],RT_round10]).aggregate("sum")
    
    df_orbitrap_peaks = df_orbitrap_peaks.iloc[:,np.r_[0:4]].groupby([df_orbitrap_peaks["Formula"],RT_round10]).aggregate("first").join(df_orbitrap_peaks.iloc[:,np.r_[4:len(df_orbitrap_peaks.columns)]].groupby([df_orbitrap_peaks["Formula"],RT_round10]).aggregate("sum") )
    
    
    return df_orbitrap_peaks

#A class for chemical formula
class chemform:
  def __init__(self, formula):
      #OLD VERSION DOESNT WORK FOR NUMBERS >9
    # #fiddle with the string so you can get the number of each element out, including 1 and 0
    # formula = formula + " "
    # formula = formula.replace(" ","1")
    # formula = "0" + formula
    
    # self.C = int(formula[formula.find("C")+1])
    # self.H = int(formula[formula.find("H")+1])
    # self.O = int(formula[formula.find("O")+1])
    # self.N = int(formula[formula.find("N")+1])
    # self.S = int(formula[formula.find("S")+1])
    
    try:
        self.C = int(re.findall(r'C(\d+)',formula)[0])
    except:
        self.C = len(re.findall(r'C',formula))
        
    try:
        self.H = int(re.findall(r'H(\d+)',formula)[0])
    except:
        self.H = len(re.findall(r'H',formula))
    
    try:
        self.O = int(re.findall(r'O(\d+)',formula)[0])
    except:
        self.O = len(re.findall(r'O',formula))
        
    try:
        self.S = int(re.findall(r'S(\d+)',formula)[0])
    except:
        self.S = len(re.findall(r'S',formula))
        
    try:
        self.N = int(re.findall(r'N(\d+)',formula)[0])
    except:
        self.N = len(re.findall(r'N',formula))
        
    

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
    
#Calculate H:C, O:C, S:C, N:C ratios
#Ratio is element1:element2
def chemform_ratios(formula):
    chemformula = chemform(formula)
    if(chemformula.C == 0):
        return np.NaN, np.NaN, np.NaN, np.NaN
    else:
        H_C = chemformula.H / chemformula.C
        O_C = chemformula.O / chemformula.C
        N_C = chemformula.N / chemformula.C
        S_C = chemformula.S / chemformula.C
        return H_C, O_C, N_C, S_C
    
    
    
  

  
#######################
####CHEMICAL ANALYSIS#######
#######################
# %%
#Function to extract the top n peaks from a cluster in terms of their chemical formula
#cluster must be a data series with the column indices of the different peaks
def cluster_extract_peaks(cluster, df_raw,num_peaks,chemform_namelist=pd.DataFrame(),dp=1,printdf=False):
    #Check they are the same length
    if(cluster.shape[0] != df_raw.shape[0]):
        print("cluster_extract_peaks returning null: cluster and peaks dataframe must have same number of peaks")
        print("Maybe your data needs transposing, or '.mean()' -ing?")
        return np.NaN
        quit()
        #print("WARNING: cluster_extract_peaks(): cluster and peaks dataframe do not have the same number of peaks")
    
    nlargest = cluster.nlargest(num_peaks)
    nlargest_pct = nlargest / cluster.sum() * 100
    #pdb.set_trace()
    output_df = pd.DataFrame()
    nlargest.index = pd.MultiIndex.from_tuples(nlargest.index, names=["first", "second"]) #Make the index multiindex again
    output_df["Formula"] = nlargest.index.get_level_values(0)
    output_df.set_index(output_df['Formula'],inplace=True)
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