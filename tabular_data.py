import pandas as pd
import numpy as np
# import boltons.iterutils as bi 


def remove_rows_with_missing_rating():
    # print(df.info())
    df.replace('', np.nan,inplace=True)
    df.dropna(subset=['Accuracy_rating','Cleanliness_rating','Location_rating','Value_rating','Check-in_rating'],inplace=True)
    return df

def combine_description_strings():
    df['Description'] = df['Description'].str.strip("[]").str.replace(",","").str.replace('\"','').str.replace("\'","").str.strip('About this space')
    print(df['Description'])

def set_feature_default_values():
    replace = ["guests", "beds", "bathrooms", "bedrooms"]
    df[replace] = df[replace].fillna(1)
    # print(df['bathrooms'])
    return df

def clean_tabular_data():
    remove_rows_with_missing_rating()
    combine_description_strings()
    set_feature_default_values()
    return df

# def load_airbnb(df, label):
#     labels = df[label]
#     features = df.drop(label,axis=1)
#     return (features, labels)

def load_airbnb(df,pred_value):
    labels=df[pred_value]
    features=df.drop(labels,axis=1)
    return (features,labels)



if __name__=='__main__':
    df=pd.read_csv('~/Airbnb/AIRBNB-DATASET/airbnb-property-listings/tabular_data/listing.csv')
    clean_df=clean_tabular_data()
    clean_df.to_csv('~/Airbnb/AIRBNB-DATASET/airbnb-property-listings/tabular_data/clean_data.csv')
    
