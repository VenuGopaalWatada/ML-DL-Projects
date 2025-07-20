# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import dump

# Loading the Dataset
telecom_cust = pd.read_csv('D:/BIA Data Science & AI/ML/Telco_Customer_Churn.csv')

# Data Preprocessing
# Filling missing values in Total Charges and convert to numeric
telecom_cust['TotalCharges'] = pd.to_numeric(telecom_cust['TotalCharges'], errors='coerce')
telecom_cust['TotalCharges'].fillna(0, inplace = True)

# Converting 'Churn' to binary variables
label_encoder = LabelEncoder()
telecom_cust['Churn'] = label_encoder.fit_transform(telecom_cust['Churn'])

# Using LabelEncoder for 'InternetService' and 'Contract'
telecom_cust['InternetService'] = label_encoder.fit_transform(telecom_cust['InternetService'])
telecom_cust['Contract'] = label_encoder.fit_transform(telecom_cust['Contract'])

# Selecting Features
selected_features = ['tenure', 'InternetService', 'Contract', 'MonthlyCharges', 'TotalCharges']

X = telecom_cust[selected_features]
y = telecom_cust['Churn']

# Training the Random Forest Model
model = RandomForestClassifier(n_estimators = 100, random_state = 101)
model.fit(X, y)

# Saving the trained model to a file
dump(model, 'random_forest_model.joblib')