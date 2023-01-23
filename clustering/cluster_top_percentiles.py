from scipy.stats import percentileofscore
import pandas as pd
import numpy as np
from functions.combine_multiindex import combine_multiindex

def cluster_top_percentiles(df_data,cluster_labels,num,highest=True,dropRT=False):
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

    Returns
    -------
    df_top_pct : Pandas DataFrame
        DataFrame containing the molecule names of the top/bottom num peaks, as the relevant column label from df_data.

    """
    unique_labels = np.unique(cluster_labels)
        
    column_labels = []
    for cluster in unique_labels:
        column_labels.append(str(cluster) + "_compound")
        column_labels.append(str(cluster) + "_pct")
        
    df_top_pct = pd.DataFrame(columns=column_labels)
    
    for cluster in unique_labels:
        data_thisclust = df_data.loc[cluster_labels==cluster]
        
        ds_pct = pd.Series([percentileofscore(df_data[mol],data_thisclust[mol].median()) for mol in df_data.columns],index=df_data.columns, dtype='float')

        #Extract the top num peaks
        if(highest):
            ds_pct_top = ds_pct.sort_values(ascending=False).iloc[0:num]
        else:
            ds_pct_top = ds_pct.sort_values(ascending=True).iloc[0:num]
        
               
        if(dropRT):
            df_top_pct[str(cluster) + "_compound"] = ds_pct_top.index.get_level_values(0)
        else:
            df_top_pct[str(cluster) + "_compound"] = "(" + combine_multiindex(ds_pct_top.index,nospaces=True) + ")"
        df_top_pct[str(cluster) + "_pct"] = ds_pct_top.values.round(1)
        
    return df_top_pct