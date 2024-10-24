import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Title of the web app
st.title("Linear Regression Web App")

# Upload the dataset
data = pd.read_csv("C:/Users/lalan/OneDrive/Desktop/Case Study Practical Exam/salary_data.csv")
data.head()
# Choose the target and feature columns
columns = data.columns.tolist()
target_column = st.selectbox("Salary", columns)
feature_columns = st.multiselect("YearsExperience", columns, default=columns[:-1])
# Splitting the data
X = data[feature_columns]
y = data[target_column]
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Building the model
model = LinearRegression()
model.fit(X_train, y_train)
# Making predictions
y_pred = model.predict(X_test)   
# Calculating R-squared score
r2 = r2_score(y_test, y_pred)
# Display the R2 score
st.write(f"R2 Score: {r2}")
# Plot the results
plt.scatter(y_test, y_pred)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs Predicted values')
st.pyplot(plt)
