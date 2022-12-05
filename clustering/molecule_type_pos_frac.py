# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pdb
def molecule_type_pos_frac(mass_spectrum,molecule_types,**kwargs):
    """
    Return the fractions of the listed molecule types that have positive data in mass_spectrum

    Parameters
    ----------
    mass_spectrum : array
        a mass spectrum
    molecule_types : array of strings
        molecule types, e.g. 'CHO'
    **kwargs : 
        mols_list: list of molecules, otherwise the default will be used

    Returns
    -------
    df_output : pandas Series
        the fraction of positive values, with index as the molecule type

    """
    #Use custom list of molecules if required
    if 'mols_list' in kwargs:
        mols_list = kwargs.get('mols_list')
    else:
        #Default list of molecule types
        mols_list = ['CHN', 'CHNS', 'CHO', 'CHON', 'CHONS', 'CHOS', 'CHS']
    
    
    ds_output = pd.Series(index=mols_list,dtype='float')
    
    #Loop through all the different molecules
    for molecule in mols_list:
        molecule_data = mass_spectrum[np.array(molecule_types) == molecule]
        #pdb.set_trace()
        ds_output.loc[molecule] = np.greater_equal(molecule_data,0).mean()
            
    return ds_output





def molecule_type_pos_frac_clusters(qt_data_2D,molecule_types,cluster_labels_1D,**kwargs):
    """
    Loop through molecule_type_pos_frac, selecting different clusters
    """
    
    #Use custom list of molecules if required
    if 'mols_list' in kwargs:
        mols_list = kwargs.get('mols_list')
    else:
        #Default list of molecule types
        mols_list = ['CHN', 'CHNS', 'CHO', 'CHON', 'CHONS', 'CHOS', 'CHS']
    
    
    df_output = pd.DataFrame(index=np.unique(cluster_labels_1D),columns=mols_list)
    
    for cluster in df_output.index:
       # pdb.set_trace()
        data_this_cluster = qt_data_2D[np.array(cluster_labels_1D) == cluster]

        #Loop through all the different molecules
        for molecule in mols_list:
            #pdb.set_trace()
            molecule_data = data_this_cluster.T[np.array(molecule_types) == molecule]
            df_output.loc[cluster][molecule] = np.greater_equal(molecule_data,0).mean()
            


            
    return df_output
    
    
    