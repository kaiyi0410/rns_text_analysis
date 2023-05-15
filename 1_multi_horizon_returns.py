# -*- coding: utf-8 -*-
"""
Created on Tue May  2 05:20:11 2023

@author: kaiyi
"""

import json
import pandas as pd
import numpy as np
import os

wd = 'C:/Users/kaiyi/OneDrive/Desktop/Kernow Asset/KERNOWAM'
os.chdir(wd)

#%% multi horizon return
# opening company returns JSON file
return_json = open('additional-data/company_returns.json')
  
# returns JSON object as a dictionary
return_data = json.load(return_json)

# convert to df
return_df = pd.Series(return_data, name='return')
return_df.index.name = 'Date'
return_df = return_df.reset_index()

# use the str.extract() method to extract date substring
return_df['RIC'] = return_df['Date'].str.extract(r"'(.*?)'")
return_df['date'] = return_df['Date'].str.extract(r"'[^']*'[^']*'([^']*)'")
return_df['date'] = return_df['date'].str[:10]
return_df = return_df.drop('Date', axis=1)

# separate different RIC with NaN so that multi-horizon return does not overlap
mask = return_df['RIC'] != return_df['RIC'].shift(periods=-1)
return_df['mask'] = mask   
return_df.loc[return_df['mask'] == True] = np.nan
return_df = return_df.drop('mask', axis=1)

# calculate multi horizon return
return_df['multi_horizon_return'] = ((return_df['return'][::-1] + 1).rolling(window=20).apply(np.prod, raw=True) - 1)[::-1]	
return_df = return_df.drop('return', axis=1)
return_df = return_df.dropna()
print(return_df.head())

#%% rns
counter = 0

# loop through rns_data_json files (keep all of them in the same folder)
for y in range(2010, 2023):  
    counter += 1
    print(str(y) + '...')
    
    # Opening company rns JSON file
    rns_json = open('rns_data/rns_data_json_' + str(y) + '.json')
      
    # returns JSON object as a dictionary
    rns_data = json.load(rns_json)
    
    # convert effective date to df
    rns_date = pd.Series(rns_data['effective_date'], name='date')
    rns_date.index.name = 'key'
    rns_date = rns_date.reset_index()
    rns_date['date'] = rns_date['date'].str[:10] # extract date from string
    
    # convert RIC to df
    rns_ric = pd.Series(rns_data['RIC'], name='RIC')
    rns_ric.index.name = 'key'
    rns_ric = rns_ric.reset_index()
    
    # convert text_body to df
    rns_text = pd.Series(rns_data['text_body'], name='text_body')
    rns_text.index.name = 'key'
    rns_text = rns_text.reset_index()
    
    # merge date, RIC, and text body
    rns_df = (rns_date.merge(rns_ric, how='inner')).merge(rns_text, how='inner')
    
    # merge returns and rns
    merged_df = rns_df.merge(return_df, how='inner')
    merged_df = merged_df.drop_duplicates(subset=['date', 'RIC'])
    merged_df = merged_df.reset_index()
    merged_df = merged_df.drop('index', axis=1)
    
    # append to final df
    if counter == 1:
        # initialise final df
        final_df = merged_df
    else:
        # Concatenate the two DataFrames row-wise
        final_df = pd.concat([final_df, merged_df], axis=0)

#%% include industry type in final_df
# Opening company meta JSON file
meta_json = open('additional-data/company_metadata.json')
  
# meta JSON object as a dictionary
meta_data = json.load(meta_json)

# convert sector to df
meta_sector = pd.Series(meta_data['TRBC Economic Sector Name'], name='sector')
meta_sector.index.name = 'RIC'
meta_sector = meta_sector.reset_index()

# merge with final df
final_df = final_df.merge(meta_sector, how='inner')
final_df = final_df.dropna()

#%% random sampling
final_df = final_df.dropna()
final_df_sampled = final_df.sample(n=10000, random_state=10)

# save as csv
final_df_sampled.to_csv('final_df_sampled.csv')
