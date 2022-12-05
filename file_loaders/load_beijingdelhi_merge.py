# -*- coding: utf-8 -*-
import pandas as pd
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
    
    
    # #Merge PMF for comparability
    #     Beijing winter
    # AMS_COA, AMS_aqOOA, AMS_FFOA, AMS_BBOA, AMS_OOA, AMS_OPOA

    # Beijing summer
    # AMS_OOA1/2/3, AMS_HOA, AMS_COA

    # Delhi summer and autumn
    # AMS_COA, AMS_HOA0, AMS_HOA, AMS_BBOA, AMS_NHOA, AMS_SFOA, AMS_SVBBOA, AMS_SVOOA

    # AMS_HOA0+AMS_NHOA = AMS_HOA => just use AMS_HOA
    
    #BUT should BBOA be in OOA or HOA?? That's the big question

    
    
    
    
    # df_merge_beijing_winter['AMS_HOA'] = df_merge_beijing_winter['AMS_FFOA']
    # df_merge_beijing_winter['AMS_OOA'] = df_merge_beijing_winter['BBOA'] + df_merge_beijing_winter['aqOOA'] + df_merge_beijing_winter['OOA'] + df_merge_beijing_winter['OPOA']
    
    # df_merge_beijing_summer['AMS_OOA'] = df_merge_beijing_summer['AMS_OOA1'] + df_merge_beijing_summer['AMS_OOA2'] + df_merge_beijing_summer['AMS_OOA3']
    
    
    # df_merge_delhi_summer['HOA'] = df_merge_delhi_summer['HOA'] + df_merge_delhi_summer['SFOA']
    
    # df_merge_delhi_summer.drop(['AMS_HOA0','AMS_NHOA'],axis=1,inplace=True)
    # df_merge_delhi_autumn.drop(['AMS_HOA0','AMS_NHOA'],axis=1,inplace=True)
    

    
    #Join together data frames
    df_all_merge = pd.concat([df_merge_beijing_winter,df_merge_beijing_summer,df_merge_delhi_summer,df_merge_delhi_autumn],join='inner')
    
    #Fuzzy merge with time index of the orbitrap data
    df_all_merge = pd.merge_asof(pd.DataFrame(index=newindex),df_all_merge,left_index=True,right_index=True,direction='nearest',tolerance=pd.Timedelta(hours=1.25))

    return df_all_merge, df_merge_beijing_winter,df_merge_beijing_summer,df_merge_delhi_summer,df_merge_delhi_autumn