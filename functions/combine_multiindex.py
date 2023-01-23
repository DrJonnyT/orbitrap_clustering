# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:30:59 2022

@author: mbcx5jt5
"""

def combine_multiindex(pd_index, sep=", ", nospaces=False):
    """
    Combine a pandas multiindex into a flat string index
    The different levels are separated by sep
    If nospaces is True, it will remove all spaces within the values
    """
    
    if(pd_index.nlevels == 1):
        return pd_index
    import pdb
    #pdb.set_trace()
    
    if(nospaces):
        new_index = pd_index.get_level_values(0).astype(str).str.replace(" ", "")
    else:
        new_index = pd_index.get_level_values(0).astype(str)
    level = 1
    
    while level < 100:
        try:
            if(nospaces):
                new_index = new_index + sep + pd_index.get_level_values(level).astype(str).str.replace(" ", "")
            else:
                new_index = new_index + sep + pd_index.get_level_values(level).astype(str)
            level += 1
        except:
            return new_index
    
    raise Exception("This error is only raised if there is an error in the while loop") 