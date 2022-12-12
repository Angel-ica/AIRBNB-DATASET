#import necessary libraries
import pandas as pd
import numpy as np
# import boltons.iterutils as bi 
# pd.options.mode.use_inf_as_na = True
names=['ID', 'Category', 'Title', 'Description','Amenities','Location','Guests','Beds','Bathrooms','Price_Night','Cleanliness_rate','Accuracy_rate','Location_rate','Check-in-rate','Value_rate','Amenities_count','Url','Bedrooms']

df=pd.read_csv('~/Airbnb/AIRBNB-DATASET/airbnb-property-listings/tabular_data/listing.csv',header=None,names=names)

#task is to remove the rows with missing values in the ratings columns
def remove_rows_with_missing_ratings():
    print(df.info())
    df.replace('', np.nan,inplace=True)
    df.dropna(subset=['Accuracy_rate','Cleanliness_rate','Location_rate','Value_rate','Check-in-rate'],inplace=True)
    df.to_json('dataset.json')
    print('done')
    return df

# def combine_description_strings():
#     # listOfStrings=df['Description']
#     # for row in df['Description'].values:
#     #     ''.join(row) 
#     #     return row
#     print(df['Description'].values)
# # listOfStrings=df['Description']
# # combine_description_strings(listOfStrings)
# print(combine_description_strings())


remove_rows_with_missing_ratings()