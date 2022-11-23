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


    #Join together data frames
    df_all_merge = pd.concat([df_merge_beijing_winter,df_merge_beijing_summer,df_merge_delhi_summer,df_merge_delhi_autumn],join='inner')
    
    #Fuzzy merge with time index of the orbitrap data
    df_all_merge = pd.merge_asof(pd.DataFrame(index=newindex),df_all_merge,left_index=True,right_index=True,direction='nearest',tolerance=pd.Timedelta(hours=1.25))

    return df_all_merge