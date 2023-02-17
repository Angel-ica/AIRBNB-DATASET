import pandas as pd
import numpy as np
# import boltons.iterutils as bi 


def remove_rows_with_missing_rating(df):
    # print(df.info(df))
    df.replace('', np.nan,inplace=True)
    df.dropna(subset=['Accuracy_rating','Cleanliness_rating','Location_rating','Value_rating','Check-in_rating'],inplace=True)
    return df

def combine_description_strings(df):
    df['Description'] = df['Description'].str.strip("[]").str.replace(",","").str.replace('\"','').str.replace("\'","").str.strip('About this space')
    return df

def set_feature_default_values(df):
    replace = ["guests", "beds", "bathrooms", "bedrooms"]
    df[replace] = df[replace].fillna(1)
    # print(df['bathrooms'])
    return df

def clean_tabular_data(df):
    df = remove_rows_with_missing_rating(df)
    df = combine_description_strings(df)
    df = set_feature_default_values(df)
    return df

def load_airbnb(df,pred_value):
    df =  clean_tabular_data(df)
    labels=df[pred_value]
    features=df.drop(pred_value,axis=1)
    return (features,labels)


if __name__=='__main__':
    df=pd.read_csv('~/Airbnb/AIRBNB-DATASET/airbnb-property-listings/tabular_data/listing.csv')
    clean_df=clean_tabular_data(df)
    clean_df.to_csv('~/Airbnb/AIRBNB-DATASET/airbnb-property-listings/tabular_data/clean_data.csv')
    
