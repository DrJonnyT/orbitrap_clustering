# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
def load_beijingdelhi_merge(newindex=None):

    #Load the met data
    rootpath = r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2"
    df_merge_beijing_winter = pd.read_csv(rootpath + r'\Beijing winter 2016\aphh_winter_filter_aggregate_merge_JT.csv',
                                      parse_dates=[0,1,2],dayfirst=True)
    df_merge_beijing_winter.set_index('date_mid',inplace=True)
    df_merge_beijing_summer = pd.read_csv(rootpath + r'\Beijing summer 2017\aphh_summer_filter_aggregate_merge_JT.csv',
                                      parse_dates=[0,1,2],dayfirst=True)
    df_merge_beijing_summer.set_index('date_mid',inplace=True)

    df_merge_delhi_summer = pd.read_csv(rootpath + r'\Delhi summer 2018\Pre_monsoon_final_merge_JT.csv',
                                      parse_dates=[0,1,2],dayfirst=True)
    df_merge_delhi_summer.set_index('date_mid',inplace=True)

    df_merge_delhi_autumn = pd.read_csv(rootpath + r'\Delhi autumn 2018\Delhi_Autumn_merge_JT.csv',
                                      parse_dates=[0,1,2],dayfirst=True)
    df_merge_delhi_autumn.set_index('date_mid',inplace=True)      


    
    ##Prepare the AMS PMF data
    #Beijing winter
    df_merge_beijing_winter['AMS_HOA'] = df_merge_beijing_winter['AMS_FFOA']
    df_merge_beijing_winter['AMS_OOA'] = df_merge_beijing_winter['AMS_OOA'] + df_merge_beijing_winter['AMS_OPOA'] + df_merge_beijing_winter['AMS_aqOOA']
    #COA already there
    #BBOA already there
    
    #Beijing summer
    #HOA already there
    df_merge_beijing_summer['AMS_OOA'] = df_merge_beijing_summer['AMS_OOA1'] + df_merge_beijing_summer['AMS_OOA2'] + df_merge_beijing_summer['AMS_OOA3']
    #COA already there
    #No BBOA data so put nans in
    a = np.empty(df_merge_beijing_summer.shape[0])
    a[:] = np.nan
    df_merge_beijing_summer['BBOA'] = a
    
    #Delhi summer
    #HOA already there as sum of HOA0 + nHOA
    df_merge_delhi_summer['AMS_OOA'] = df_merge_delhi_summer['AMS_SVOOA']
    #COA already there
    #BBOA already there as sum of SFOA and SVBBOA
    
    #Delhi autumn
    #HOA already there as sum of HOA0 + nHOA
    df_merge_delhi_autumn['AMS_OOA'] = df_merge_delhi_autumn['AMS_SVOOA']
    #COA already there
    #BBOA already there as sum of SFOA and SVBBOA
    

    #Join together data frames
    df_all_merge = pd.concat([df_merge_beijing_winter,df_merge_beijing_summer,df_merge_delhi_summer,df_merge_delhi_autumn],join='inner')
    
    #Add in AMS BBOA
       
    a = np.empty(df_merge_beijing_summer.shape[1])
    a[:] = np.nan
    df_all_merge['AMS_BBOA'] = pd.concat([df_merge_beijing_winter['AMS_BBOA'],
                                          pd.Series(a),
                                          df_merge_delhi_summer['AMS_BBOA'],
                                          df_merge_delhi_autumn['AMS_BBOA']])


    
    #Fuzzy merge with time index of the orbitrap data
    df_all_merge = pd.merge_asof(pd.DataFrame(index=newindex),df_all_merge,left_index=True,right_index=True,direction='nearest',tolerance=pd.Timedelta(hours=1.25))

    #Scale the AMS PMF to sum to AMS_Org
    # import pdb
    # pdb.set_trace()
    ds_scalefac = df_all_merge['AMS_Org'] / (df_all_merge['AMS_OOA'] + df_all_merge['AMS_HOA'] + df_all_merge['AMS_COA'] + df_all_merge['AMS_BBOA'])
    df_all_merge['AMS_OOA'] = df_all_merge['AMS_OOA'] * ds_scalefac
    df_all_merge['AMS_HOA'] = df_all_merge['AMS_HOA'] * ds_scalefac
    df_all_merge['AMS_COA'] = df_all_merge['AMS_COA'] * ds_scalefac
    df_all_merge['AMS_BBOA'] = df_all_merge['AMS_BBOA'] * ds_scalefac
    
    
    
    
    return df_all_merge, df_merge_beijing_winter,df_merge_beijing_summer,df_merge_delhi_summer,df_merge_delhi_autumn