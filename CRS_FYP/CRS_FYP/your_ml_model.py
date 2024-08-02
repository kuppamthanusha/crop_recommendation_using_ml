import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier

df = pd.read_csv("C:/Users/mails/Downloads/CRS_FYP/CRS_FYP/Crop_recommendation.csv")

#SKEWNESS IN DATA

#TARGET COLUMN

class_labels = df['label'].unique().tolist()

le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

class_labels = le.classes_

#SPLIT THE DATA
x = df.drop('label',axis=1)
y = df['label']

features_data = {'columns': list(x.columns)}

#Train the model
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.10,shuffle=True)

#BUILD MODEL
rf_model = RandomForestClassifier()
rf_model.fit(x_train,y_train)

y_pred = rf_model.predict(x_test)

#HYPER PARAMETER TUNING
rf = RandomForestClassifier()
param_grid = {'n_estimators':np.arange(50,200),
    'criterion':['gini','entropy'],
    'max_depth':np.arange(2,25),
    'min_samples_split':np.arange(2,25),
    'min_samples_leaf':np.arange(2,25)}

rscv_model = RandomizedSearchCV(rf,param_grid, cv=5)
rscv_model.fit(x_train,y_train)
rscv_model.best_estimator_

#MODEL EVALUATION TEST
new_rf_model = rscv_model.best_estimator_
y_pred = new_rf_model.predict(x_test)

y_pred_train = new_rf_model.predict(x_train)

features_data = {'columns':list(x.columns)}

test_series = pd.Series(np.zeros(len(features_data['columns'])),index=features_data['columns'])

import pickle

with open('new_rf_model.pickle','wb') as file:
    pickle.dump(new_rf_model, file)

#USER INPUTS
#test_series['N'] = int(input("N:"))
#test_series['P'] = int (input("P:"))
#test_series['K'] = int(input("K:"))
#test_series['temperature'] = int(input("temperature:"))
#test_series['humidity'] = int(input("Humidity:"))
#test_series['ph'] = int(input("ph:"))
#test_series['rainfall'] = int(input("rainfall:"))

#OUTPUT
#output = new_rf_model.predict([test_series])[0]
#print("Recommended Crop:",class_labels[output])