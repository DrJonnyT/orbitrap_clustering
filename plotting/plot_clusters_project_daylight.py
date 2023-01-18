import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pdb

#%%Count clusters by project and time, and plot them
def plot_clusters_project_daylight(df_cluster_labels_mtx,ds_dataset_cat,ds_day_frac,title_prefix='',title_suffix=''):
    sns.set_context("talk", font_scale=0.8)    
    
    for num_clusters in df_cluster_labels_mtx.columns:
        c = df_cluster_labels_mtx[num_clusters]
        a = pd.DataFrame(c.values,columns=['clust'],index=ds_dataset_cat.index)
        #a1 = pd.DataFrame(c.values,columns=['clust'],index=ds_time_cat.index)
        #b = df_dataset_cat

        #IF THIS FAILS ITS BECAUSE IT NEEDS DF NOT DS
        df_clust_cat_counts = a.groupby(ds_dataset_cat)['clust'].value_counts(normalize=True).unstack()
        df_cat_clust_counts = ds_dataset_cat.groupby(a['clust']).value_counts(normalize=True).unstack()
        #df_clust_time_cat_counts = a1.groupby(ds_time_cat)['clust'].value_counts(normalize=True).unstack()
        #df_time_cat_clust_counts = ds_time_cat.groupby(a1['clust']).value_counts(normalize=True).unstack()
        #pdb.set_trace()
        df_day_frac = pd.DataFrame()
        df_day_frac['Day'] = ds_day_frac
        df_day_frac['Night'] = 1 - ds_day_frac
        
        
        
        fig,ax = plt.subplots(1,3,figsize=(12,5),sharey=True)
        ax = ax.ravel()
        cmap = 'RdYlBu'
        df_clust_cat_counts.plot.bar(ax=ax[1],colormap=cmap,stacked=True,legend=False)
        df_cat_clust_counts.plot.bar(ax=ax[0],stacked=True,colormap='RdBu',width=0.8)
        #df_clust_time_cat_counts.plot.bar(ax=ax[2],colormap=cmap,legend=False,stacked=True)
        #df_time_cat_clust_counts.plot.bar(ax=ax[3],stacked=True,colormap='PuOr',width=0.8)
        df_day_frac.plot.bar(ax=ax[2],stacked=True,colormap='viridis_r',width=0.8)
        
        
        
        
        
        suptitle = title_prefix + str(num_clusters) + ' clusters' + title_suffix
        plt.suptitle(suptitle)
        ax[0].set_xlabel('')
        ax[0].set_xlabel('Cluster')
        ax[0].set_ylabel('Fraction')
        
        #ax[1].set_ylabel('Fraction')
        ax[2].set_xlabel('Cluster')
        #ax[2].set_ylabel('Fraction')
        
        #ax[0].tick_params(axis='x', labelrotation=35)
        ax[1].set_xticklabels(['BW','BS','DS', 'DA'])


        #Set legend handles and size
        handles, labels = ax[0].get_legend_handles_labels()
        ax[0].legend(handles, ['BW','BS','DS', 'DA'], bbox_to_anchor=(0.5, -0.5),loc='lower center',ncol=2,handletextpad=0.4)

        handles, labels = ax[1].get_legend_handles_labels()
        #pdb.set_trace()
        if(len(labels) > 5):
            ncols = int(np.ceil(len(labels)/2))
            ax[1].legend(title='Cluster number', bbox_to_anchor=(0.5, -0.5),loc='lower center',ncol=ncols,handletextpad=0.4,labelspacing=0.8)
        else:
            ncols = len(labels)
            ax[1].legend(title='Cluster number', bbox_to_anchor=(0.5, -0.5),loc='lower center',ncol=ncols,handletextpad=0.4,labelspacing=0.8)
            
        
            
        
        
        ax[2].legend(bbox_to_anchor=(0.5, -0.5),loc='lower center',ncol=3,handletextpad=0.4)
        
        
        
        
        
    
    sns.reset_orig()
    #return df_clust_cat_counts, df_cat_clust_counts, df_clust_time_cat_counts, df_time_cat_clust_counts