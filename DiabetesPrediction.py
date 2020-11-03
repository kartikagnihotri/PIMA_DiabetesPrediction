# -*- coding: utf-8 -*-
"""
@author: kartik

"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data=pd.read_csv("C:/Users/kartik/Desktop/projects/PIMA_DiabetesPrediction/diabetes.csv")

print(data.shape)
print(data.head(5))

# checkING if any null value is present
print(data.isnull().values.any())


print(data.corr())

# Correlation
#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")

diabetes_true_count = len(data.loc[data['Outcome'] == 1])
diabetes_false_count = len(data.loc[data['Outcome'] == 0])


print((diabetes_true_count,diabetes_false_count))

## Train Test Split

from sklearn.model_selection import train_test_split
feature_columns=data.columns.tolist()

feature_columns=[c for c in feature_columns if c not in ["Outcome"]]

predicted_class="Outcome"

X=data[feature_columns]
y=data[predicted_class]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=10)

# mark zero values as missing or NaN
from numpy import nan
X_train[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']] = X_train[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].replace(0, nan)
X_test[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']] = X_test[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].replace(0, nan)
# fill missing values with mean column values
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)
# count the number of NaN values in each column
print(X_train.isnull().sum())
print(X_test.isnull().sum())

from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(random_state=10)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
            oob_score=False, random_state=10, verbose=0, warm_start=False)
random_forest_model.fit(X_train, y_train.ravel())

predict_train_data = random_forest_model.predict(X_test)

from sklearn import metrics

print("Accuracy = {0:.5f}".format(metrics.accuracy_score(y_test, predict_train_data)))

