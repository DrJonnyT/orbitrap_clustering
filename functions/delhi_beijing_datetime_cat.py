# -*- coding: utf-8 -*-
import datetime as dt
import pandas as pd
from astral import LocationInfo
from astral.sun import sun
import pdb
import numpy as np

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




def calc_daytime_frac_BeijingDelhi(df_times):
    
    city_Beijing = LocationInfo("Beijing", "China", "Asia/Shanghai", 39.97444, 116.3711)
    city_Delhi = LocationInfo("Delhi", "India", "Asia/Calcutta", 28.664, 77.232)
    
    df_output = pd.DataFrame(index=df_times['date_mid'])
    dataset_cat = delhi_beijing_datetime_cat(df_times['date_mid'])
    
    daylight_hours = []
    night_hours = []
    
    

    #Loop through each filter and calculate day and night hours
    #This seems like a janky way of doing the loop but is is not faster than looping through a dataframe with df.iter()?
    for time_start, time_mid, time_end, cat in zip(df_times['date_start'],df_times['date_mid'],df_times['date_end'],dataset_cat):
        
        if (cat == 'Beijing_summer') or (cat == 'Beijing_winter'):
            city = city_Beijing
        elif (cat == 'Delhi_summer') or (cat == 'Delhi_autumn'):
            city = city_Delhi
        
        #pdb.set_trace()
        daylight = calc_daylight_deltat(time_start,time_end,city)
        daylight_hours.append(daylight)
        night_hours.append((time_end-time_start) - daylight)

    return daylight_hours

def calc_daylight_deltat(time_start,time_end,astral_city):
    """
    Calculate the total number of daylight hours between two times

    Parameters
    ----------
    time_start : datetime
        Start time in LOCAL TIME
    time_end : datetime
        End time in LOCAL TIME
    astral_city : LocationInfo from astral package
        This contains the details of lat/long/timezone

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    
    time_start = time_start.replace(tzinfo=astral_city.tzinfo)
    time_end = time_end.replace(tzinfo=astral_city.tzinfo)
    
    #The total number of calendar days between the start and end times
    num_days = (time_end.date() - time_start.date()).days
    
    #pdb.set_trace()
    
    sun0 = sun(astral_city.observer, date=time_start)
    
    if num_days == 0:
        #Filter is only on one day and ends before midnight
        if time_end < sun0['sunrise'] or time_start > sun0['sunset']:
            daylight0 = dt.timedelta(0)
        else:        
            daylight0 = min(sun0['sunset'],time_end) - max(sun0['sunrise'],time_start)
        return daylight0
    
    elif num_days == 1:
        #Filter starts on one day and ends the next day
        #pdb.set_trace()
        
        if time_start > sun0['sunset']:
            daylight0 = dt.timedelta(0)
        else:
            #Just go up to sunset on first day
            daylight0 = sun0['sunset'] - max(sun0['sunrise'],time_start)
            #pdb.set_trace()
        
        #Second (or last) day
        sun9 = sun(astral_city.observer, date=time_end)
        if time_end > sun9['sunrise']:
            daylight9 = min(sun9['sunset'],time_end) - sun9['sunrise']
        else:
            daylight9 = dt.timedelta(0)
        return daylight0 + daylight9
    elif num_days > 1:
        #Not coded up but you would have to loop through all the other days adding up the daylight hours
        return np.nan
    
    
    

