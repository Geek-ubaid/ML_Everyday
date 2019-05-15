
# coding: utf-8

# # Assignment 2
# 
# Before working on this assignment please read these instructions fully. In the submission area, you will notice that you can click the link to **Preview the Grading** for each step of the assignment. This is the criteria that will be used for peer grading. Please familiarize yourself with the criteria before beginning the assignment.
# 
# An NOAA dataset has been stored in the file `data/C2A2_data/BinnedCsvs_d400/fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89.csv`. The data for this assignment comes from a subset of The National Centers for Environmental Information (NCEI) [Daily Global Historical Climatology Network](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/readme.txt) (GHCN-Daily). The GHCN-Daily is comprised of daily climate records from thousands of land surface stations across the globe.
# 
# Each row in the assignment datafile corresponds to a single observation.
# 
# The following variables are provided to you:
# 
# * **id** : station identification code
# * **date** : date in YYYY-MM-DD format (e.g. 2012-01-24 = January 24, 2012)
# * **element** : indicator of element type
#     * TMAX : Maximum temperature (tenths of degrees C)
#     * TMIN : Minimum temperature (tenths of degrees C)
# * **value** : data value for element (tenths of degrees C)
# 
# For this assignment, you must:
# 
# 1. Read the documentation and familiarize yourself with the dataset, then write some python code which returns a line graph of the record high and record low temperatures by day of the year over the period 2005-2014. The area between the record high and record low temperatures for each day should be shaded.
# 2. Overlay a scatter of the 2015 data for any points (highs and lows) for which the ten year record (2005-2014) record high or record low was broken in 2015.
# 3. Watch out for leap days (i.e. February 29th), it is reasonable to remove these points from the dataset for the purpose of this visualization.
# 4. Make the visual nice! Leverage principles from the first module in this course when developing your solution. Consider issues such as legends, labels, and chart junk.
# 
# The data you have been given is near **Ann Arbor, Michigan, United States**, and the stations the data comes from are shown on the map below.

# In[59]:

import matplotlib.pyplot as plt
import mplleaflet
import pandas as pd
import numpy as np


def leaflet_plot_stations(binsize, hashid):

    df = pd.read_csv('data/C2A2_data/BinSize_d{}.csv'.format(binsize))

    station_locations_by_hash = df[df['hash'] == hashid]

    lons = station_locations_by_hash['LONGITUDE'].tolist()
    lats = station_locations_by_hash['LATITUDE'].tolist()

    plt.figure(figsize=(8,8))

    plt.scatter(lons, lats, c='r', alpha=0.7, s=200)

    return mplleaflet.display()

leaflet_plot_stations(400,'fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89')


# In[60]:

def get_2015_data(index,df):
    df_2015 = df[index >= '2015-01-01']
    df_past = df[index < '2015-01-01']
    return df_2015,df_past


# In[61]:

def setup_data():

    df = pd.read_csv('data/C2A2_data/BinnedCsvs_d400/fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89.csv')
    
    df.drop('ID',axis=1,inplace=True)
    df['Data_Value'] = df['Data_Value']/10
    data_index = pd.DatetimeIndex(df['Date'])
    
    #removing leap year days
    df = df[~((data_index.is_leap_year)& (data_index.month == 2) & (data_index.day == 29))]
    
    #getting 2015 and past dataframes
    index = pd.DatetimeIndex(df['Date'])
    df_2015,df_past = get_2015_data(index,df)
    
    #getting the day month and year from the date
    df_2015['month'] = pd.to_datetime(df_2015['Date']).dt.month
    df_2015['year'] = pd.to_datetime(df_2015['Date']).dt.year
    df_2015['day'] = pd.to_datetime(df_2015['Date']).dt.day
    
    df_past['month'] = pd.to_datetime(df_past['Date']).dt.month
    df_past['year'] = pd.to_datetime(df_past['Date']).dt.year
    df_past['day'] = pd.to_datetime(df_past['Date']).dt.day
    

    return df_2015,df_past


# In[62]:

def extract_high_lows(df):
    
    df_min = df.loc[df['Element'] == 'TMIN']
    df_max = df.loc[df['Element'] == 'TMAX']
    df_max_group = df_max.groupby(['month','day']).agg('max')
    df_min_group = df_min.groupby(['month','day']).agg('min')
    
    return df_max_group,df_min_group


# In[82]:

def extract_2015_high_lows(data_max,data_min,df):
    df['Date'] = pd.to_datetime(df['Date'])
    df_min = df.loc[df['Element'] == 'TMIN']
    df_max = df.loc[df['Element'] == 'TMAX']
    
    df_2015_max_group = df_max.groupby(['month','day']).agg('max')
    df_2015_min_group = df_min.groupby(['month','day']).agg('min')
    
    max_2015_df, min_2015_df, max_index, min_index = [], [], [], []
    for i in range(len(data_max)):
        if (data_max['Data_Value'].iloc[i] - df_2015_max_group['Data_Value'].iloc[i])<0:
            max_2015_df.append(df_2015_max_group['Data_Value'].iloc[i])
            max_index.append(i)
        
    for i in range(len(data_min)):
        if (data_min['Data_Value'].iloc[i] - df_2015_min_group['Data_Value'].iloc[i])>0:
            min_2015_df.append(df_2015_min_group['Data_Value'].iloc[i])
            min_index.append(i)
    
    
    
    return min_2015_df,max_2015_df,min_index,max_index


# In[86]:

import pandas as pd
import numpy as np
import datetime

df_2015, df_past = setup_data()
df_highs ,df_lows = extract_high_lows(df_past)
df_2015_lows, df_2015_highs,min_index,max_index = extract_2015_high_lows(df_highs,df_lows,df_2015)

#func = lambda x: pd.to_datetime(x).strftime('%m/%d')
#x_axis_labels = np.array(list(map(func,x_axis_labels.tolist())))

plt.figure(figsize=(15,10))
plt.plot(df_lows['Data_Value'].tolist(), c='green', alpha = 0.5, label = 'Minimum Temperature (2005-14)')
plt.plot(df_highs['Data_Value'].tolist(), c ='red', alpha = 0.5, label = 'Maximum Temperature (2005-14)')

plt.scatter(min_index, df_2015_lows, s = 10, c = 'blue', label = 'Record Break Minimum (2015)')
plt.scatter(max_index, df_2015_highs, s = 10, c = 'black', label = 'Record Break Maximum (2015)')

plt.gca().fill_between(range(len(df_lows['Data_Value'])), 
                       df_lows['Data_Value'].tolist(), df_highs['Data_Value'].tolist(), 
                       facecolor='blue', 
                       alpha=0.25)

plt.legend(loc = 8, frameon=False, title='Temperature', fontsize=8)
plt.xticks( np.linspace(0, 365 , num = 12), (r'Jan', r'Feb', r'Mar', r'Apr', r'May',                                                    r'Jun', r'Jul', r'Aug', r'Sep', r'Oct', r'Nov', r'Dec') )
plt.xlabel('Months')
plt.ylabel('Temperature (tenths of degrees C)')
plt.title(r'Extreme temperature of "College Station, Texas" by months, with outliers')
plt.show()

plt.savefig('Image.Png')


# In[ ]:



