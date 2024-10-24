import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset (for the web app)
salary_data = pd.read_csv('Salary_Data.csv')

# Clean the data by dropping rows with missing values
salary_data = salary_data.dropna(subset=['Years of Experience', 'Salary'])

# Select relevant columns
X = salary_data[['Years of Experience']]
y = salary_data['Salary']

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# App Title
st.title("Salary Prediction Based on Years of Experience")

# User input for years of experience
years_of_experience = st.slider('Years of Experience', 0, 40, 1)

# Predict salary based on input
predicted_salary = model.predict(np.array([[years_of_experience]]))[0]

# Display the prediction
st.write(f"Predicted Salary for {years_of_experience} years of experience: ${predicted_salary:,.2f}")

# Plotting the data and regression line
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs. Years of Experience')
plt.legend()

# Display the plot
st.pyplot(plt)

# Display model performance (R² score)
r2_score = model.score(X, y)
st.write(f"Model R² Score: {r2_score:.2f}")
