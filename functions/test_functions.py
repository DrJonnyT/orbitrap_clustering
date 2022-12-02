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
from functions.delhi_beijing_datetime_cat import delhi_beijing_datetime_cat

import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import pytest

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
        
    
    
def test_delhi_beijing_datetime_cat():
    df = pd.DataFrame([0],index=[dt.datetime(2016,11,15)])
    ds_cat = delhi_beijing_datetime_cat(df.index)
    assert type(ds_cat) == pd.core.series.Series
    
    