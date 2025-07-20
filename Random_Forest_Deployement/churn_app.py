# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 16:50:45 2025

@author: watad
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 16:33:52 2025

@author: watad
"""

import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder

# Loading the trained random forest model
model = load('C:/Users/watad/.spyder-py3/random_forest_model.joblib')

# Creating a Streamlit App
st.title('Customer Churn Prediction App')

# Input fields for feature values on the main screen
st.header("Enter Customer Information")
tenure = st.number_input("Tenure (in months)", min_value = 0, max_value = 100, value = 1)
internet_service = st.selectbox("Internet Service", ('DSL', 'Fibre Optic', 'No'))
contract = st.selectbox("Contract", ('Month-to-month', 'One year', 'Two year'))
monthly_charges = st.number_input("Monthly Charges", min_value = 0, max_value = 200, value = 50)
total_charges = st.number_input("Total Charges", min_value = 0, max_value = 10000, value = 0)

# Map input values to numberic using the label mapping
label_mapping = {
    'DSL': 0,
    'Fiber Optic': 1,
    'No': 2,
    'Month-to-month': 0,
    'One year': 1,
    'Two year': 2,
    }

internet_service = label_mapping[internet_service]
contract = label_mapping[contract]

# Making prediction using the model
prediction = model.predict([[tenure, internet_service, contract, monthly_charges, total_charges]])

# Displaying prediction result on main screen
if prediction[0] == 0:
    st.success("This customer is likely to stay")
else:
    st.error("This customer is likely to churn")