#import necessary libraries
import sys
sys.path.insert(0,'/Users/angelicaaluo/Airbnb/AIRBNB-DATASET/')
from tabular_data import load_airbnb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import recall_score, precision_score,f1_score
import pandas as pd

#load the data
df = pd.read_csv('/Users/angelicaaluo/Airbnb/AIRBNB-DATASET/airbnb-property-listings/tabular_data/clean_data.csv')
df.drop('Unnamed: 19',axis=1,inplace=True)
X,y=load_airbnb(df,"Category")
#split the data into train,test and val sets 
X_train, X_test, y_train,y_test=train_test_split(X, y, test_size=0.3,random_state=42)
X_train,X_val, y_train,y_val=train_test_split(X_train,y_train, test_size=0.5,random_state=42)
#create an instance of a logistsic regression
lgr = LogisticRegression()
#fit the logistic regression on our data 
lgr.fit(X_train,y_train)
#predict the data and evaluate the outcome of our model 
evaluation_metrics={
    "Confusion matrix: ": confusion_matrix,
    "accuracy_score: ": accuracy_score,
    "recall_score: ": recall_score,
    "f1_score: ": f1_score,
    "precision_score: ": precision_score
}
for name,metric in evaluation_metrics.items():
    y_pred= lgr.predict(X_test)
    eval_metric = f"{name}: {metric(y_pred,y_test)}"
    print(eval_metric)
