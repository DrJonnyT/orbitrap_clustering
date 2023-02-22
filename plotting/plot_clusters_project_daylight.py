import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pdb


def plot_clusters_project_daylight(cluster_labels,ds_dataset_cat,ds_day_frac,**kwargs):
    """
    Plot the fraction of each cluster from the different datasets (panel (a)), and the fraction from day/night (panel (b))

    Parameters
    ----------
    cluster_labels : array
        Cluster labels
    ds_dataset_cat : Pandas Series
        Index is time, values are the category of the dataset from Beijing/Delhi
    ds_day_frac : Pandas Series
        Index is time, values are the fraction of daylight vs night
    
    """
    
    #Count clusters by project and time, and plot them
    sns.set_context('talk')
    
    cluster_labels = np.array(cluster_labels)
    a = pd.DataFrame(cluster_labels,columns=['clust'],index=ds_dataset_cat.index)
    
    #IF THIS FAILS ITS BECAUSE IT NEEDS DF NOT DS
    df_clust_cat_counts = a.groupby(ds_dataset_cat)['clust'].value_counts(normalize=True).unstack()
    df_cat_clust_counts = ds_dataset_cat.groupby(a['clust']).value_counts(normalize=True).unstack()
    
    df_day_frac = pd.DataFrame()
    df_day_frac['Day'] = ds_day_frac
    df_day_frac['Night'] = 1 - ds_day_frac
    
    #Make the figure
    fig,ax = plt.subplots(1,2,figsize=(12,5),sharey=True)
    ax = ax.ravel()
    cmap = 'RdYlBu'
    df_cat_clust_counts.plot.bar(ax=ax[0],stacked=True,colormap='RdBu',width=0.8)
    
    df_day_frac.plot.bar(ax=ax[1],stacked=True,colormap='viridis_r',width=0.8)
    
    #Set labels          
    ax[0].set_xlabel('Cluster')
    ax[0].set_ylabel('Fraction')
    ax[1].set_xlabel('Cluster')
    ax[1].set_ylabel('Fraction')


    #Set legend handles and size
    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].legend(handles, ['Beijing Winter','Beijing Summer','Delhi Premonsoon', 'Delhi Postmonsoon'], bbox_to_anchor=(0.5, -0.4),loc='lower center',ncol=2,handletextpad=0.4)            
    ax[1].legend(bbox_to_anchor=(0.5, -0.32),loc='lower center',ncol=3,handletextpad=0.4)
        
    if "suptitle" in kwargs:
        plt.suptitle(kwargs.get("suptitle"))
        
    sns.reset_orig()
    