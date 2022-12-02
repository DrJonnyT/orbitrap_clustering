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
