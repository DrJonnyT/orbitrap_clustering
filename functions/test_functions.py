# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 12:37:40 2022

To run this file, run 'python -m pytest' from the main directory

@author: mbcx5jt5
"""

from functions.combine_multiindex import combine_multiindex
from functions.prescale_whole_matrix import prescale_whole_matrix
from functions.optimal_nclusters_r_card import optimal_nclusters_r_card
import pandas as pd
import numpy as np
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
    assert optimal_nclusters_r_card([1,2],[0.1,0.5],[50,2]) == 2
    assert optimal_nclusters_r_card([1,2],[0.1,0.99],[50,40]) == 2
    assert optimal_nclusters_r_card([1,2],[0.1,0.5],[50,40],maxr_threshold=0.05) == 1
    assert optimal_nclusters_r_card([1,2],[0.1,0.5],[50,40],mincard_threshold=60) == 1
    
    #Warning if no suboptimal cluster numbers are found
    with pytest.warns(UserWarning):
        assert optimal_nclusters_r_card([1,2],[0.1,0.5],[50,40]) == 2
        
        
        
    