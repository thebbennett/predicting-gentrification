# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 13:13:45 2018

@author: bbenn
"""

#########################################################################################

## Author: Brittany Bennett
## 8/14/2018
## Predicting Gentrification in Denver, CO

#########################################################################################
## Import necessary packages
import os
import pandas as pd
import glob
import csv
import numpy as np
import shapefile
import geopy.distance
from geopy.geocoders import Nominatim
from time import sleep
import requests
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
import geopandas

#########################################################################################
## Set the working directory to the project path
path = r"C:\Users\bbenn\Documents\GitHub\predicting-gentrification"
os.chdir(path)
#########################################################################################
## Read in data for 2011 from the U.S. Census

#Gross Rent
gross_2011 = pd.read_csv('./data/2011/2011-gross-rent.csv', header = 0)
keys_gross2011 = {}
reader = csv.DictReader(open('./data/keys/2011/2011-gross-rent-keys.csv'))
for row in reader:
    keys_gross2011[row['GEO.id']] = row['Id']
gross_2011.columns = gross_2011.columns.to_series().map(keys_gross2011)

# Median Rent
median_2011 = pd.read_csv('./data/2011/2011-median-rent.csv', header = 0)
keys_median2011 = {}
reader = csv.DictReader(open('./data/keys/2011/2011-median-rent-keys.csv'))
for row in reader:
    keys_median2011[row['GEO.id']] = row['Id']
median_2011.columns = median_2011.columns.to_series().map(keys_median2011)

#Rent by Income
by_income_2011 = pd.read_csv('./data/2011/2011-rent-by-income.csv', header = 0)
keys_income2011 = {}
reader = csv.DictReader(open('./data/keys/2011/2011-rent-by-income-keys.csv'))
for row in reader:
    keys_income2011[row['GEO.id']] = row['Id']
by_income_2011.columns = by_income_2011.columns.to_series().map(keys_income2011)

#Create one, large 2011 dataframe of all rent variables
df_2011 = pd.concat([gross_2011, median_2011, by_income_2011],axis=1)
df_2011 = df_2011.T.drop_duplicates().T

#########################################################################################
## Read in data for 2016 from the U.S. Census

#Gross Rent
gross_2016 = pd.read_csv('./data/2016/2016-gross-rent.csv', header = 0)
keys_gross2016 = {}
reader = csv.DictReader(open('./data/keys/2016/2016-gross-rent-keys.csv'))
for row in reader:
    keys_gross2016[row['GEO.id']] = row['Id']
gross_2016.columns = gross_2016.columns.to_series().map(keys_gross2016)

# Median Rent
median_2016 = pd.read_csv('./data/2016/2016-median-rent.csv', header = 0)
keys_median2016 = {}
reader = csv.DictReader(open('./data/keys/2016/2016-median-rent-keys.csv'))
for row in reader:
    keys_median2016[row['GEO.id']] = row['Id']
median_2016.columns = median_2016.columns.to_series().map(keys_median2016)

# Rent by income
by_income_2016 = pd.read_csv('./data/2016/2016-rent-by-income.csv', header = 0)
keys_income2016 = {}
reader = csv.DictReader(open('./data/keys/2016/2016-rent-by-income-keys.csv'))
for row in reader:
    keys_income2016[row['GEO.id']] = row['Id']
by_income_2016.columns = by_income_2016.columns.to_series().map(keys_income2016)

#Create one, large 2016 dataframe of all rent variables
df_2016 = pd.concat([gross_2016, median_2016, by_income_2016],axis=1)
df_2016 = df_2016.T.drop_duplicates().T


#########################################################################################
col_names = df_2016.filter(regex='^Margin.*')
column_numbers = []
for name in col_names:
    column_numbers.append(df_2016.columns.get_loc(name))
    
df_2016 = df_2016.drop(df_2016.columns[column_numbers], axis = 1)


col_names = df_2011.filter(regex='^Margin.*')
column_numbers = []
for name in col_names:
    column_numbers.append(df_2011.columns.get_loc(name))
    
df_2011 = df_2011.drop(df_2011.columns[column_numbers], axis = 1)

df_2011.drop(df_2011.index[0], inplace=True)
df_2016.drop(df_2016.index[0], inplace=True)

#########################################################################################
## Create columns for percentile of median gross rent in 2011 and 2016

rent_increase = pd.concat([
        df_2011['Id2'],
        df_2011['Estimate; Median gross rent'],
        df_2016['Estimate; Median gross rent']
        ],axis =1)
rent_increase.drop(rent_increase.index[143], inplace=True)
rent_increase.drop(rent_increase.index[77], inplace=True)

rent_increase.columns = ["Census Tract", "2011_Median_gross_rent","2016_Median_gross_rent"]    

rent_increase['2011_Median_gross_rent'] = rent_increase['2011_Median_gross_rent'].str.extract('(\d+)')
rent_increase['2011_Median_gross_rent'] = rent_increase['2011_Median_gross_rent'].astype(str).astype(int)

rent_increase['2016_Median_gross_rent'] = rent_increase['2016_Median_gross_rent'].str.extract('(\d+)')
rent_increase['2016_Median_gross_rent'] = rent_increase['2016_Median_gross_rent'].astype(str).astype(int)

rent_increase['Increase'] = rent_increase['2016_Median_gross_rent'] - rent_increase['2011_Median_gross_rent']


#########################################################################################


### PREDICTOR VARIABLES ###

#########################################################################################

## All shape files in CO, not just Denver

downtown = (39.744830,-104.994407)

sf = shapefile.Reader("./data/census_tracts_2010/census_tracts_2010.shp")
result = []
for shape in list(sf.iterShapes()):
   for ip in range(len(shape.points)):
       x_lon = shape.points[ip][0]
       y_lat= shape.points[ip][1]
   result.append((x_lon, y_lat))
    
tract_names = []
records = sf.records()

for tract in range(len(records)):
    tract_names.append(sf.record(tract)[2])

tract_names = list(map(int, tract_names))

lon = np.array(x_lon).tolist()
lat = np.array(y_lat).tolist()

predictor_vars = pd.DataFrame(
    {'Census Tract ID':tract_names,
     'Lon and Lat' : result
    })
    
predictor_vars[['Lon', 'Lat']] = predictor_vars['Lon and Lat'].apply(pd.Series)
predictor_vars = predictor_vars.drop(predictor_vars.columns[1],axis = 1)

distance_from_cbd = []

for index, row in predictor_vars.iterrows():
   lon = row[1]
   lat = row[2]
   tract_pair = (lat,lon)
   distance_from_cbd.append(geopy.distance.vincenty(tract_pair, downtown).mi)
   sleep(1)
   
distance_from_cbd_df = pd.DataFrame(
    {'Distance from CBD': distance_from_cbd,
    })
    
predictor_vars = pd.concat([predictor_vars, distance_from_cbd_df], axis=1)

#########################################################################################

landmarks_df = pd.read_csv("./data/historic_landmarks.csv",dtype=str,)
geolocator = Nominatim(user_agent="brittany",timeout=None)    

## clean the landmark data
landmarks_df = landmarks_df[landmarks_df.ADDRESS_LINE1 != "None"]
landmarks_df['ADDRESS'] = landmarks_df['ADDRESS_LINE1'].apply(lambda x: x.split(',')[0])

landmarks_df.ADDRESS.str.contains('^[0-9].*')
num_numeric_sites = landmarks_df[landmarks_df.ADDRESS.str.contains('^[a-zA-Z].*')]
numeric_sites = landmarks_df[landmarks_df.ADDRESS.str.contains('^[0-9].*')].reset_index(drop=True)


##############################################################################

## drop rows manually by visual inspection, because why not 
pruned_numeric_sites = numeric_sites.drop(numeric_sites.index[[7,12,20,22,41,58,59,66,69,70,95,121,
                                        122,151,155,157,174,179,207,213,214,
                                        219,235,236,261,265,269,271,280,283,292,299,302
        ]])
pruned_numeric_sites['ADDRESS'] = pruned_numeric_sites['ADDRESS'].astype(str) 

landmark_points = []
for index, row in pruned_numeric_sites.iterrows():
    location = geolocator.geocode(row[21]+" Denver CO")
    if location is not None:
        landmark_points.append((location.latitude, location.longitude))
    sleep(1)
#########################################################################################
## Map the lon lat data of historical landmarks to their census tract

historical_census_tracts = []
for latlon in landmark_points:
    lat = latlon[0]
    lon = latlon[1]
    parameters = {"lat": lat, "lon": lon}
    response = requests.get('https://geo.fcc.gov/api/census/area', params=parameters)
    data = response.json()
    historical_census_tracts.append(data["results"][0]["block_fips"])

historical_census_tracts = [x.encode('UTF8') for x in historical_census_tracts]
aggregated_historical_census_tracts = Counter(historical_census_tracts)
df = pd.DataFrame.from_dict(aggregated_historical_census_tracts, orient='index').reset_index()
df.columns = ["Census Tract ID", "Number of historical sites"]
df['Census Tract ID']=df['Census Tract ID'].apply(int)
predictor_vars = pd.merge(predictor_vars, df, on='Census Tract ID',right_index=True, left_index=True)

#########################################################################################
## Read in data for 2000 from the U.S. Census for the age variable
age_2000 = pd.read_csv('./data/2000/2000-age.csv', header = 0)
keys_age2000 = {}
reader = csv.DictReader(open('./data/keys/2000/2000-age-keys.csv'))
for row in reader:
    keys_age2000[row['GEO.id']] = row['Id']
age_2000.columns = age_2000.columns.to_series().map(keys_age2000)

col_names = age_2000.filter(regex='.*Margin.*')
column_numbers = []
for name in col_names:
    column_numbers.append(age_2000.columns.get_loc(name))
    
age_2000 = age_2000.drop(age_2000.columns[column_numbers], axis = 1)
age_2000.drop(age_2000.index[0], inplace=True)

under_18 =['Id2', 'Number; Total population - SEX AND AGE - Under 5 years',
           'Number; Total population - SEX AND AGE - 5 to 9 years',
           'Number; Total population - SEX AND AGE - 10 to 14 years',
           'Number; Total population - SEX AND AGE - 15 to 19 years'

]
children_2000 = age_2000[under_18]

#########################################################################################
## Read in data for 2011 from the U.S. Census for the age variable
age_2011 = pd.read_csv('./data/2011/2011-age.csv', header = 0)
keys_age2011 = {}
reader = csv.DictReader(open('./data/keys/2011/2011-age-keys.csv'))
for row in reader:
    keys_age2011[row['GEO.id']] = row['Id']
age_2011.columns = age_2011.columns.to_series().map(keys_age2011)

col_names = age_2011.filter(regex='.*Margin.*')
column_numbers = []
for name in col_names:
    column_numbers.append(age_2011.columns.get_loc(name))
    
age_2011 = age_2011.drop(age_2011.columns[column_numbers], axis = 1)
age_2011.drop(age_2011.index[0], inplace=True)

under_18 =['Id2', 'Total; Estimate; Total population', 'Total; Estimate; AGE - Under 5 years', 
           'Total; Estimate; AGE - 5 to 9 years','Total; Estimate; AGE - 10 to 14 years',
           'Total; Estimate; AGE - 15 to 19 years']

children_2011 = age_2011[under_18]
children_2011 = children_2011.drop(144,axis=0)
#########################################################################################
## Create columns for total children for both 2011 and 2000 data frames
children_2011 = children_2011.apply(pd.to_numeric)
children_2011['percent_children'] = children_2011['Total; Estimate; AGE - Under 5 years'] + children_2011['Total; Estimate; AGE - 5 to 9 years'] + children_2011['Total; Estimate; AGE - 10 to 14 years'] + children_2011['Total; Estimate; AGE - 15 to 19 years']
children_2011['total_children'] = children_2011['Total; Estimate; Total population'] * children_2011['percent_children'] * 0.01
children_2011.drop(children_2011.columns[[1, 2,3,4,5,6]], axis=1, inplace=True)
children_2011.columns = ["Census Tract ID", "Total children 2011"]


children_2000 = children_2000.apply(pd.to_numeric)
children_2000['total_children'] = children_2000['Number; Total population - SEX AND AGE - Under 5 years'] + children_2000['Number; Total population - SEX AND AGE - 5 to 9 years'] + children_2000['Number; Total population - SEX AND AGE - 10 to 14 years']+ children_2000['Number; Total population - SEX AND AGE - 15 to 19 years']
children_2000.drop(children_2000.columns[[1, 2,3,4]], axis=1, inplace=True)
children_2000.columns = ["Census Tract ID", "Total children 2000"]

#########################################################################################
## Create a col for the change in children between 2011 and 2000 and append it to the 
## predictor variable dataframe

change_in_children = pd.merge(children_2011, children_2000, on='Census Tract ID',right_index=True, left_index=True)
change_in_children["Change in children"] = change_in_children["Total children 2011"] - change_in_children["Total children 2000"]
change_in_children.drop(change_in_children.columns[[1,2]], axis=1, inplace=True)

predictor_vars = pd.merge(predictor_vars, change_in_children, on='Census Tract ID',right_index=True, left_index=True)
#########################################################################################
## Change in the number of single households from 2000 to 2011
households_2011 = pd.read_csv('./data/2011/2011-households.csv', header = 0)
keys_households2011 = {}
reader = csv.DictReader(open('./data/keys/2011/2011-households-keys.csv'))
for row in reader:
    keys_households2011[row['GEO.id']] = row['Id']
households_2011.columns = households_2011.columns.to_series().map(keys_households2011)

col_names = households_2011.filter(regex='.*Margin.*')
column_numbers = []
for name in col_names:
    column_numbers.append(households_2011.columns.get_loc(name))
    
households_2011 = households_2011.drop(households_2011.columns[column_numbers], axis = 1)
households_2011.drop(households_2011.index[0], inplace=True)

households_2011 = households_2011[['Id2', 'Estimate; Family households: - 2-person household']]
households_2011.columns = ["Census Tract ID", "Single households 2011"]
households_2011 = households_2011.astype(str).astype(float)

#########################################################################################
households_2000 = pd.read_csv('./data/2000/2000-household.csv', header = 0)
keys_households2000 = {}
reader = csv.DictReader(open('./data/keys/2000/2000-household-keys.csv'))
for row in reader:
    keys_households2000[row['GEO.id']] = row['Id']
households_2000.columns = households_2000.columns.to_series().map(keys_households2000)

col_names = households_2000.filter(regex='.*Margin.*')
column_numbers = []
for name in col_names:
    column_numbers.append(households_2000.columns.get_loc(name))
    
households_2000 = households_2000.drop(households_2000.columns[column_numbers], axis = 1)
households_2000.drop(households_2000.index[0], inplace=True)

households_2000 = households_2000[['Id2', '2-person household']]
households_2000.columns = ["Census Tract ID", "Single households 2000"]
households_2000 = households_2000.astype(str).astype(float)

change_in_households = pd.merge(households_2011, households_2000, on='Census Tract ID',right_index=True, left_index=True)
change_in_households["Change in households"] = change_in_households["Single households 2011"] - change_in_households["Single households 2000"]
change_in_households.drop(change_in_households.columns[[1,2]], axis=1, inplace=True)

predictor_vars = pd.merge(predictor_vars, change_in_households, on='Census Tract ID',right_index=True, left_index=True)

#########################################################################################

mobility_2011 = pd.read_csv('./data/2011/2011-mobility.csv', header = 0)
keys_mobility2011 = {}
reader = csv.DictReader(open('./data/keys/2011/2011-mobility-keys.csv'))
for row in reader:
    keys_mobility2011[row['GEO.id']] = row['Id']
mobility_2011.columns = mobility_2011.columns.to_series().map(keys_mobility2011)

col_names = mobility_2011.filter(regex='.*Margin.*')
column_numbers = []
for name in col_names:
    column_numbers.append(mobility_2011.columns.get_loc(name))
    
mobility_2011 = mobility_2011.drop(mobility_2011.columns[column_numbers], axis = 1)
mobility_2011.drop(mobility_2011.index[0], inplace=True)
mobility_2011.drop(mobility_2011.index[143], inplace=True)

moved_recently = mobility_2011[['Id2', 'Estimate; Total: - 1 to 4 years']]
moved_recently.columns = ["Census Tract ID", "Moved recently"]
moved_recently = moved_recently.astype(str).astype(float)

predictor_vars = pd.merge(predictor_vars, moved_recently, on='Census Tract ID',right_index=True, left_index=True)

#########################################################################################
## Read in data for working women for 2011
employment_2011 = pd.read_csv('./data/2011/2011-employment.csv', header = 0)
keys_employment2011 = {}
reader = csv.DictReader(open('./data/keys/2011/2011-employment-keys.csv'))
for row in reader:
    keys_employment2011[row['GEO.id']] = row['Id']
employment_2011.columns = employment_2011.columns.to_series().map(keys_employment2011)

col_names = employment_2011.filter(regex='.*Margin.*')
column_numbers = []
for name in col_names:
    column_numbers.append(employment_2011.columns.get_loc(name))
    
employment_2011 = employment_2011.drop(employment_2011.columns[column_numbers], axis = 1)
employment_2011.drop(employment_2011.index[0], inplace=True)
employment_2011.drop(employment_2011.index[143], inplace=True)

working_women = employment_2011[['Id2', 'Total; Estimate; SEX - Female']]
working_women.columns = ["Census Tract ID", "Working women 2011"]
working_women = working_women.astype(str).astype(float)
#########################################################################################
employment_2000 = pd.read_csv('./data/2000/2000-population.csv', header = 0)
keys_employment2000 = {}
reader = csv.DictReader(open('./data/keys/2000/2000-population-keys.csv'))
for row in reader:
    keys_employment2000[row['GEO.id']] = row['Id']
employment_2000.columns = employment_2000.columns.to_series().map(keys_employment2000)

employment_2000.drop(employment_2000.index[0], inplace=True)
employment_2000.drop(employment_2000.index[143], inplace=True)

working_women_2000 = employment_2000[['Id2', 'Number - Both sexes; Population 16 years and over', 'Percent - Female; Population 16 years and over - In labor force']]
working_women_2000 = working_women_2000.astype(str).astype(float)

working_women['Working women 2000'] = working_women_2000['Number - Both sexes; Population 16 years and over'] * working_women_2000['Percent - Female; Population 16 years and over - In labor force'] *0.01
working_women['Working women change'] = working_women['Working women 2011'] - working_women['Working women 2000']

working_women.drop(working_women.columns[[1,2]], axis=1, inplace=True)

predictor_vars = pd.merge(predictor_vars, working_women, on='Census Tract ID',right_index=True, left_index=True)

#########################################################################################

### Preidctive Modeling ###

#########################################################################################





"""
df = pd.concat([rent_increase, predictor_vars], axis=1)
fig, axs = plt.subplots(nrows = 3, ncols=3)
sns.regplot(x='Distance from CBD', y='Increase', data=df, ax=axs[0,0])
sns.regplot(x='Number of historical sites', y='Increase', data=df, ax=axs[0,1])
sns.regplot(x='Total children',y='Increase', data=df, ax=axs[0,2])


Single households
Moved recently 
Working women 
"""