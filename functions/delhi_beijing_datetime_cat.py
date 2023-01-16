# -*- coding: utf-8 -*-
import datetime as dt
import pandas as pd

def delhi_beijing_datetime_cat(df_index):
    """
    Map filter times onto a categorical for what dataset it is
    Only for beijing and delhi data

    Parameters
    ----------
    df_index : pandas datetimeindex
        

    Returns
    -------
    Categorical of which dataset the data belong to

    """
    #Make a numerical flag for each dataset based on the datetime
    datetimecat_num = pd.Series(index=df_index,dtype='int')
    for index in df_index:
        if((index >= dt.datetime(2016,11,9)) and (index < dt.datetime(2016,12,12))):
            datetimecat_num.loc[index] = 0
        elif((index >= dt.datetime(2017,5,17)) and (index < dt.datetime(2017,6,27))):
            datetimecat_num.loc[index] = 1    
        elif((index >= dt.datetime(2018,5,27)) and (index < dt.datetime(2018,6,7))):
            datetimecat_num.loc[index] = 2
        elif((index >= dt.datetime(2018,10,8)) and (index < dt.datetime(2018,11,8))):
            datetimecat_num.loc[index] = 3
    
    #A dictionary to map the numerical onto a categoricla
    dict_datetime_to_cat =	{
          0: "Beijing_winter",
          1: "Beijing_summer",
          2: "Delhi_summer",
          3: "Delhi_autumn",
        }
    cat = pd.Categorical(datetimecat_num.map(dict_datetime_to_cat).values,
                         ['Beijing_winter','Beijing_summer' ,'Delhi_summer','Delhi_autumn'], ordered=True)
    return pd.Series(cat,index=df_index)




def delhi_calc_time_cat(df_in):
    """
    Map filter times onto night/morning/midday/afternoon as per Hamilton et al 2021

    Parameters
    ----------
    df_in : dataframe
        Must have 'date_start', 'date_end' columns

    Returns
    -------
    cat1 : pandas categorial
        Categories are Morning/Midday/Afternoon/Night based on the local time

    """
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



