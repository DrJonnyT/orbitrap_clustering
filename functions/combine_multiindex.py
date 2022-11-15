# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:30:59 2022

@author: mbcx5jt5
"""

def combine_multiindex(pd_index, sep=", "):
    """
    Combine a pandas multiindex into a flat string index
    The different levels are separated by sep
    """
    
    if(pd_index.nlevels == 1):
        return pd_index
    
    new_index = pd_index.get_level_values(0).astype(str)
    level = 1
    
    while True:
        try:
            new_index = new_index + sep + pd_index.get_level_values(level).astype(str)
            level += 1
        except:
            return new_index