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





def molecule_type_pos_frac_clusters(data_2D,molecule_types,cluster_labels_1D,**kwargs):
    """
    Like molecule_type_pos_frac, but selecting data from different clusters
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
        data_this_cluster = data_2D[np.array(cluster_labels_1D) == cluster]

        #Loop through all the different molecules
        for molecule in mols_list:
            #pdb.set_trace()
            molecule_data = data_this_cluster.T[np.array(molecule_types) == molecule]
            df_output.loc[cluster][molecule] = np.greater_equal(molecule_data,0).mean()
                        
    return df_output


def molecule_type_pos_frac_clusters_mtx(data_2D,molecule_types,df_cluster_label_mtx,**kwargs):
    """
    Loop through molecule_type_pos_frac_clusters, taking cluster labels from a 2D matrix
    
    Parameters
    ----------
    see molecule_type_pos_frac_clusters
    
    df_cluster_label_mtx: pandas dataframe
    index is different samples, like the index of data_2D
    columns is the number of clusters
    
    Returns
    -------
    output_array: array
    an array of dataframes, each of which is an output of molecule_type_pos_frac_clusters
    
    num_clusters: array
    the corresponding number of clusters

    """
    
    #Use custom list of molecules if required
    if 'mols_list' in kwargs:
        mols_list = kwargs.get('mols_list')
    else:
        #Default list of molecule types
        mols_list = ['CHN', 'CHNS', 'CHO', 'CHON', 'CHONS', 'CHOS', 'CHS']
        
        
    num_clusters = df_cluster_label_mtx.columns
    output_array = []    
    
    for n_clusters in num_clusters:
        #pdb.set_trace()
        output_array.append(molecule_type_pos_frac_clusters(data_2D,molecule_types,df_cluster_label_mtx[n_clusters],mols_list=mols_list))
        
    return output_array, num_clusters


    