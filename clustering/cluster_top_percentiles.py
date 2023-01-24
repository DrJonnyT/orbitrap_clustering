from scipy.stats import percentileofscore
import pandas as pd
import numpy as np
from functions.combine_multiindex import combine_multiindex
import pdb

def cluster_top_percentiles(df_data,cluster_labels,num,highest=True,dropRT=False,**kwargs):
    """
    For each cluster label, extract a list of the top num peaks that are unusually high or low.
    This means peaks that have a high/low percentile within their entire distribution

    Parameters
    ----------
    df_data : Pandas DataFrame
        DataFrame of orbitrap filter data. Index is normally time, columns are peak labels, z is concentration
    cluster_labels : numpy array
        Time series of cluster labels, with the same time basis as df_data
    num : int
        Number of peaks to extract
    highest : bool, optional
        Pick out the num highest peaks. The default is True. If False, it picks out the num lowest peaks.
    dropRT : bool, optional
        If True, return only the molecular formula. If False, return also the retention time (RT). The default is False.
    kwargs : keyword arguments
        mol_labels : pandas series containing labels for each molecule. E.g. a list of potential sources
    
    Returns
    -------
    df_top_pct : Pandas DataFrame
        DataFrame containing the molecule names of the top/bottom num peaks, as the relevant column label from df_data.

    """
    
    if "mol_labels" in kwargs:
        mol_labels = True
        mol_labels_list = kwargs.get("mol_labels")
        #Remove duplicates
        mol_labels_list = mol_labels_list[~mol_labels_list.index.duplicated(keep='first')]
    else:
        mol_labels = False
    
    
    
    unique_labels = np.unique(cluster_labels)
       
    #Define dataframe with multiindex. First level is cluster number, second level is the property        
    if (mol_labels):
        array1 = np.repeat(unique_labels,3)
    else:
        array1 = np.repeat(unique_labels,2)
    array2 = []
    for cluster in unique_labels:
        if(dropRT):
            array2.append('Formula')
        else:
            array2.append('(Formula/RT)')
        array2.append('Pct')
        if(mol_labels):
            array2.append('Source')
    
    column_labels_multi = pd.MultiIndex.from_arrays([array1,array2], names=('Cluster', ''))
                              
        
    df_top_pct = pd.DataFrame(index=np.arange(1,num+1,1),columns=column_labels_multi)
    
    for cluster in unique_labels:
        data_thisclust = df_data.loc[cluster_labels==cluster]
        
        ds_pct = pd.Series([percentileofscore(df_data[mol],data_thisclust[mol].median()) for mol in df_data.columns],index=df_data.columns, dtype='float')

        #Extract the top num peaks
        if(highest):
            ds_pct_top = ds_pct.sort_values(ascending=False).iloc[0:num]
        else:
            ds_pct_top = ds_pct.sort_values(ascending=True).iloc[0:num]
        
        if(dropRT):
            df_top_pct[(cluster, 'Formula')] = ds_pct_top.index.get_level_values(0)
        else:
            df_top_pct[(cluster, '(Formula/RT)')] = "(" + combine_multiindex(ds_pct_top.index,nospaces=True) + ")"
        
        #Round to 1DP
        ds_pct_top.loc[:] = ["%.1f" % value for value in ds_pct_top.values]
        
        df_top_pct[(cluster, 'Pct')] = ds_pct_top.values
        
        #Get labels from the list of labels, if given
        if(mol_labels):
            df_top_pct[(cluster, 'Source')] = mol_labels_list[ds_pct_top.index.get_level_values(0)].values
    return df_top_pct