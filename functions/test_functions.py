# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 12:37:40 2022

To run this file, run 'python -m pytest' from the main directory

@author: mbcx5jt5
"""

from functions.combine_multiindex import combine_multiindex
from functions.prescale_whole_matrix import prescale_whole_matrix
from functions.optimal_nclusters_r_card import optimal_nclusters_r_card
from functions.avg_array_clusters import avg_array_clusters
from functions.delhi_beijing_time import delhi_beijing_datetime_cat, delhi_calc_time_cat, calc_daylight_deltat, calc_daylight_hours_BeijingDelhi

from functions.math import round_to_nearest_x_even, round_to_nearest_x_odd, sqrt_sum_squares, num_frac_above_val
from functions.math import normdot, normdot_1min

import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import pytest
from astral import LocationInfo

def test_combine_multiindex():
    assert combine_multiindex(pd.Index([0])) == pd.Index([0])  
    arrays = [[1, 2], ['red', 'blue']]
    assert (combine_multiindex(pd.MultiIndex.from_arrays(arrays, names=('number', 'color'))) == pd.Index(['1, red', '2, blue'])).all()
    assert (combine_multiindex(pd.MultiIndex.from_arrays(arrays, names=('number', 'color')), sep = "P") == pd.Index(['1Pred', '2Pblue'])).all()
    
    #Error if input is not a pandas index
    with pytest.raises(Exception) as e_info:
        combine_multiindex(pd.Series([55]))
        
        
def test_prescale_whole_matrix():
    unscaled = np.array([[1,2,3],[4,5,6]]).T
    scaled, minmax = prescale_whole_matrix(unscaled,MinMaxScaler())
    assert scaled == pytest.approx(np.array([[0.,0.2,0.4],[0.6,0.8,1.0]]).T)
    
    
def test_optimal_nclusters_r_card():
    assert optimal_nclusters_r_card([1,2,3],[0.1,0.5,0.7],[50,2,1]) == 1
    assert optimal_nclusters_r_card([1,2,3],[0.1,0.5,0.99],[50,40,30]) == 2
    assert optimal_nclusters_r_card([1,2],[0.1,0.5],[50,40],maxr_threshold=0.15) == 1
    assert optimal_nclusters_r_card([1,2],[0.1,0.5],[50,40],mincard_threshold=45) == 1
    
    #Warning if no suboptimal cluster numbers are found
    with pytest.warns(UserWarning):
        assert np.isnan(optimal_nclusters_r_card([1,2],[0.1,0.5],[50,40]))
                
    with pytest.warns(UserWarning):
        assert np.isnan(optimal_nclusters_r_card([1,2,3],[0.99,0.5,0.7],[50,2,1]))
        

def test_avg_array_clusters():
    labels = [0,1,1,0]
    data1D = [10.,20.,10.,5.]
    data2D = [[10.,20.],[20.,40.],[10.,20.],[0.,0.]]
    
    assert avg_array_clusters(labels,data1D).equals(pd.Series([7.5,15.],index=[0,1]))
    assert avg_array_clusters(labels,data1D,weights=[1.,1.,1.,1.]).equals(pd.Series([7.5,15.],index=[0,1]))
    assert avg_array_clusters(labels,data1D,weights=[3.,2.,2.,3.]).equals(pd.Series([7.5,15.],index=[0,1]))
    assert avg_array_clusters(labels,data1D,weights=[1,2,2,4]).equals(pd.Series([6.,15.],index=[0,1]))
    #Test with nan
    assert avg_array_clusters(np.append(labels,0),np.append(data1D,np.nan)).equals(pd.Series([7.5,15.],index=[0,1]))
    assert np.isnan(avg_array_clusters(np.append(labels,0),np.append(data1D,np.nan),removenans=False)[0])
    
    
    #Input is not 1D
    with pytest.raises(Exception):
        avg_array_clusters(labels,data2D)
    
    #Input not the right dimensions
    with pytest.raises(Exception):
        avg_array_clusters(labels,[0,1])
    
    #Weights not the right dimensions
    with pytest.raises(Exception):
        avg_array_clusters(labels,data1D,weights=[1,2])
        
    
    
#Test datetime categoricals
def test_delhi_beijing_datetime_cat():
    idx = pd.Index([dt.datetime(2016,11,15),dt.datetime(2017,5,25),dt.datetime(2018,6,2),dt.datetime(2018,11,5)])
    ds_cat = delhi_beijing_datetime_cat(idx)
    assert ds_cat[0] == 'Beijing_winter'
    assert ds_cat[1] == 'Beijing_summer'
    assert ds_cat[2] == 'Delhi_summer'
    assert ds_cat[3] == 'Delhi_autumn'
    assert ds_cat.index.equals(idx)
    
def test_delhi_calc_time_cat():
    date_start = [dt.datetime(2016,11,15,1),dt.datetime(2017,5,25,7),dt.datetime(2018,6,2,12),dt.datetime(2018,11,5,18),dt.datetime(2018,11,5,14)]
    date_end = [dt.datetime(2016,11,15,2),dt.datetime(2017,5,25,8),dt.datetime(2018,6,2,13),dt.datetime(2018,11,5,19),dt.datetime(2018,11,6,15)]
    df_in = pd.DataFrame({'date_start': date_start, 'date_end': date_end})
    df_in = df_in.set_index((df_in['date_end']-df_in['date_start'])/2 + df_in['date_start'])
    ds_cat = delhi_calc_time_cat(df_in)
    
    assert ds_cat[0] == 'Night'
    assert ds_cat[1] == 'Morning'
    assert ds_cat[2] == 'Midday'
    assert ds_cat[3] == 'Afternoon'
    assert ds_cat[4] == '24hr'
    
    
def test_calc_daylight_deltat():
    city_Beijing = LocationInfo("Beijing", "China", "Asia/Shanghai", 39.97444, 116.3711)
    assert calc_daylight_deltat(dt.datetime(2018,8,8,12),dt.datetime(2018,8,8,13),city_Beijing).total_seconds()/3600 == 1
    assert calc_daylight_deltat(dt.datetime(2018,8,8,0),dt.datetime(2018,8,8,1),city_Beijing).total_seconds()/3600 == 0.
    assert calc_daylight_deltat(dt.datetime(2018,8,7,23),dt.datetime(2018,8,8,1),city_Beijing).total_seconds()/3600 == 0
    #compare to known daylight hours
    assert calc_daylight_deltat(dt.datetime(2023,1,18,1),dt.datetime(2023,1,18,23),city_Beijing).total_seconds()/3600 == pytest.approx(9.716,abs=0.1)

def test_calc_daylight_hours_BeijingDelhi():
    #Additional to test_calc_daylight_deltat
    df = pd.DataFrame()
    df['date_start'] = [dt.datetime(2017,6,20,12),dt.datetime(2017,6,20,0),dt.datetime(2017,6,19,23)]
    df['date_end'] = [dt.datetime(2017,6,20,13),dt.datetime(2017,6,20,1),dt.datetime(2017,6,20,1)]
    df['date_mid'] = (df['date_end'] -df['date_start'])/2 + df['date_start']
    
    df_daytime = calc_daylight_hours_BeijingDelhi(df)
    
    assert df_daytime['daylight_hours'].iloc[0] == 1
    assert df_daytime['daylight_hours'].iloc[1] == 0
    assert df_daytime['daylight_hours'].iloc[2] == 0
    assert df_daytime['night_hours'].iloc[0] == 0
    assert df_daytime['night_hours'].iloc[1] == 1
    assert df_daytime['night_hours'].iloc[2] == 2
    
    
    
    
#Test math
def test_round_to_nearest_x_even():
    assert round_to_nearest_x_even(2.5) == 2
    assert round_to_nearest_x_even(-0.9) == 0
    assert np.array_equal( round_to_nearest_x_even([5,0.1]) ,  [4.,0.])
    
def test_round_to_nearest_x_odd():
    assert round_to_nearest_x_odd(2.5) == 3
    assert round_to_nearest_x_odd(-0.9) == -1
    assert np.array_equal( round_to_nearest_x_odd([4,0.1]) ,  [5.,1.])
    
def test_sqrt_sum_squares():
    assert sqrt_sum_squares([0,5,10,10,10,10,10,10]) == 25
    
def test_num_frac_above_val():
    assert num_frac_above_val([5,5,5,5,5,5,5,5,0,-1],4) == 0.8
    assert num_frac_above_val([5,5,5,5,5,5,5,5,0,-1,np.nan],4) == 0.8
    assert np.isnan(num_frac_above_val([5,5,5,5,5,5,5,0,-1,np.inf],4))
    assert np.isnan(num_frac_above_val([np.nan,np.nan],4))
    assert np.isnan(num_frac_above_val([],4))
    assert np.isnan(num_frac_above_val([5,5],np.nan))
    
    
    
    
def test_normdot():
    assert normdot(1,2) == 1
    assert normdot([1,2,3],[1,2,3]) == 1
    assert normdot([1,2,3],[2,4,6]) == 1
    assert normdot([1,2,0],[0,0,1]) == 0
    assert normdot([1,2,3],[-1,-2,-3]) == -1
    
    a, b = [1,2,3], [6,8,-15]
    assert normdot_1min(a,b) == 1 - normdot(a,b)
    a, b = [0,-0.1,88], [6,8,-15]
    assert normdot_1min(a,b) == 1 - normdot(a,b)
    
