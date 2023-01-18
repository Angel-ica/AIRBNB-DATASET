'''here, we perfrom simple linear regression on the airbnb dataset using Stochastic Gradient Descent.'''

#import necessary libraries 
import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor 
from tabular_data import load_airbnb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# load the data
df= pd.read_csv('airbnb-property-listings/tabular_data/clean_data.csv')
df['Price_Night'].dropna(inplace=True)
df.drop('Unnamed: 19',axis=1,inplace=True)
X,y =load_airbnb(df,"Price_Night")
X = df.select_dtypes(include=['int','float'])
#TODO
#1.TRY ONEHOTENCODER FOR CATEGORICAL DATA 

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

#create the sgd model 
sgd=SGDRegressor(loss='squared_error',penalty='l2',alpha=0.001,max_iter=1000)

#fit the model to the training data 
my_sgd=sgd.fit(X_train, y_train)
# my_linear_model=LinearRegression().fit(X_train,y_train)
y_pred=my_sgd.predict(X_test)
# print(X_test,y_test)
#evaluate the model on the test data
score = sgd.score(X_test, y_test)
print(f'Test score: {score}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_pred,y_test))}')
print(f'R^2: {r2_score(y_pred,y_test)}')
