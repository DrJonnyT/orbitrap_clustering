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
    
    
    df_output = pd.Series(index=mols_list,dtype='float')
    
    #Loop through all the different molecules
    for molecule in mols_list:
        molecule_data = mass_spectrum[np.array(molecule_types) == molecule]
        #pdb.set_trace()
        df_output.loc[molecule] = np.greater_equal(molecule_data,0).mean()
            
    return df_output





def molecule_type_pos_frac_clusters(cluster_labels_qt,qt_data,molecule_types):
    
#     df_output = pd.DataFrame(index=np.unique(cluster_labels_qt),columns=np.unique(molecule_types))
    
#     for cluster in df_output.index:
#         data_this_cluster = qt_data[cluster_labels_qt == cluster]
#         #Loop through all the different molecules
#         for molecule in df_output.columns:
#             this_molecule_this_cluster = data_this_cluster[molecule_types == molecule]
#             pdb.set_trace()
#             df_output.loc[cluster][molecule] = np.greater_equal(this_molecule_this_cluster,0).mean()
            
#     return df_output
    
    
    