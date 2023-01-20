import pandas as pd
import glob, os
import datetime as dt



#%%


#Load all time data, ie start/mid/end
df_all_times = pd.read_csv(r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\PMF data\Times_all.csv")
df_all_times['date_start'] = pd.to_datetime(df_all_times['date_start'],dayfirst=True)
df_all_times['date_mid'] = pd.to_datetime(df_all_times['date_mid'],dayfirst=True)
df_all_times['date_end'] = pd.to_datetime(df_all_times['date_end'],dayfirst=True)

df_all_times.set_index(df_all_times['date_mid'],inplace=True)



#%%
def load_precip_from_HYSPLIT(filepath):
    
    #First find the length of the header. In these files, we look for the word RAINFALL, and data start on the next line
    with open(filepath, 'r') as fp:
        for l_no, line in enumerate(fp):
            # search string
            if 'RAINFALL' in line:
                headerline = l_no
                break

    #This loads the dataframe into a slightly weird way but it works for our purposes
    df_traj = pd.read_csv(filepath,skiprows=headerline,delim_whitespace=True)
    total_precip = df_traj['RAINFALL'].sum()
    return total_precip


#%%
os.chdir(r"C:\hysplit\working\trajectories")


##BEIJING WINTER
file_times = []
precip_totals = []

for file in glob.glob("*beijing_winter*"):
    
    #A string of the release date/time
    #The 20 is to get from e.g. 16 to 2016 for the year
    datetimestr = file[-8:]
    
    filetime = dt.datetime(int(datetimestr[0:2])+2000,int(datetimestr[2:4]),int(datetimestr[4:6]),int(datetimestr[6:8]))
    
    total_precip = load_precip_from_HYSPLIT(file)
    
    file_times.append(filetime)
    precip_totals.append(total_precip)

ds_precip_Beijing_winter = pd.Series(precip_totals,index=file_times,dtype='float')


##BEIJING SUMMER
file_times = []
precip_totals = []

for file in glob.glob("*beijing_summer*"):
    
    #A string of the release date/time
    datetimestr = file[-8:]
    
    filetime = dt.datetime(int(datetimestr[0:2])+2000,int(datetimestr[2:4]),int(datetimestr[4:6]),int(datetimestr[6:8]))
    
    total_precip = load_precip_from_HYSPLIT(file)
    
    file_times.append(filetime)
    precip_totals.append(total_precip)

ds_precip_Beijing_summer = pd.Series(precip_totals,index=file_times,dtype='float')


##DELHI SUMMER
file_times = []
precip_totals = []

for file in glob.glob("*delhi_summer*"):
    
    #A string of the release date/time
    datetimestr = file[-8:]
    
    filetime = dt.datetime(int(datetimestr[0:2])+2000,int(datetimestr[2:4]),int(datetimestr[4:6]),int(datetimestr[6:8]))
    
    total_precip = load_precip_from_HYSPLIT(file)
    
    file_times.append(filetime)
    precip_totals.append(total_precip)

ds_precip_Delhi_summer = pd.Series(precip_totals,index=file_times,dtype='float')


##DELHI AUTUMN
file_times = []
precip_totals = []

for file in glob.glob("*delhi_autumn*"):
    
    #A string of the release date/time
    datetimestr = file[-8:]
    
    filetime = dt.datetime(int(datetimestr[0:2])+2000,int(datetimestr[2:4]),int(datetimestr[4:6]),int(datetimestr[6:8]))
    
    total_precip = load_precip_from_HYSPLIT(file)
    
    file_times.append(filetime)
    precip_totals.append(total_precip)

ds_precip_Delhi_autumn = pd.Series(precip_totals,index=file_times,dtype='float')



#%%Concatenate all the precip data together
ds_precip_all = pd.concat([ds_precip_Beijing_winter,ds_precip_Beijing_summer,ds_precip_Delhi_summer,ds_precip_Delhi_autumn])
ds_precip_all.to_csv(r"C:\Users\mbcx5jt5\Google Drive\Shared_York_Man2\HYSPLIT_precip.csv")
#Now we want the average accumulated precip over each filter time period
#I did this in Igor because I already had a function to do it, seems surprisingly difficult in pandas?
#Going from a regular to an irregular time series