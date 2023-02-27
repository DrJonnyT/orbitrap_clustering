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
    kwargs: list of keyword arguments
        'suptitle' : String, figure suptitle.
    
    """
    
    #Count clusters by project and time, and plot them
    sns.set_context('talk')
    
    cluster_labels = np.array(cluster_labels)
    a = pd.DataFrame(cluster_labels,columns=['clust'],index=ds_dataset_cat.index)
    
    
    #Data grouped by cluster
    df_cat_clust_counts = ds_dataset_cat.groupby(a['clust']).value_counts(normalize=True).unstack()
    
    #Day frac day and night grouped by cluster
    df_day_frac = pd.DataFrame()
    df_day_frac['Day'] = ds_day_frac
    df_day_frac['Night'] = 1 - ds_day_frac 
    
    #Make the figure
    fig,ax = plt.subplots(2,1,figsize=(5,12),sharey=True)
    ax = ax.ravel()
    
    #Adjust margins
    plt.subplots_adjust(left=0, bottom=0, right=1, top=0.92, wspace=0, hspace=0.4)
    
    #Plot data
    df_cat_clust_counts.plot.bar(ax=ax[0],stacked=True,colormap='RdBu',width=0.8)
    df_day_frac.plot.bar(ax=ax[1],stacked=True,colormap='viridis_r',width=0.8)
    
    #Set labels          
    ax[0].set_xlabel('Cluster')
    ax[0].set_ylabel('Fraction')
    ax[1].set_xlabel('Cluster')
    ax[1].set_ylabel('Fraction')


    #Set legend handles and size
    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].legend(handles, ['Beijing Winter','Beijing Summer','Delhi Premonsoon', 'Delhi Postmonsoon'], bbox_to_anchor=(0.5, -0.35),loc='lower center',ncol=2,handletextpad=0.4)            
    ax[1].legend(['Majority day','Majority night'],bbox_to_anchor=(0.5, -0.35),loc='lower center')
    
    #Add boxes for (a) and (b) outside axes
    ax[0].text(-0.2,0.89,'(a)',ha='left',va='top',transform=ax[0].transAxes)
    ax[1].text(-0.2,0.89,'(b)',ha='left',va='top',transform=ax[1].transAxes)
        
    if "suptitle" in kwargs:
        plt.suptitle(kwargs.get("suptitle"))
    
    
    sns.reset_orig()
    