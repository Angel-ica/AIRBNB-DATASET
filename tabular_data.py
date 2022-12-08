import pandas as pd
import numpy as np
from openpyxl.workbook import Workbook
import itertools
import boltons.iterutils as bi 
pd.options.mode.use_inf_as_na = True
def remove_rows_with_missing_ratings():
    names=["ID", "Category", 'Title', 'Description','Amenities','Location','Guests','Beds','Bathrooms','Price_Night','Cleanliness_rate','Accuracy_rate','Location_rate','Check-in-rate','Value_rate','Amenities_count','Url','Bedrooms']

    df=pd.read_csv('~/Airbnb/AIRBNB-DATASET/airbnb-property-listings/tabular_data/listing.csv',header=None,names=names, na_values='""')
    print(df.info())
    # print(df)
    print(np.sort(df.isnull()))
    for row in df['Value_rate']:
        if np.any(df['Value_rate'].isnull())==True:
            df.drop(row)
    print(df.info())

    # new_dict=bi.remap(df_dict.values,({key: value is not 'nan' for key, value in df_dict.values}))

remove_rows_with_missing_ratings()