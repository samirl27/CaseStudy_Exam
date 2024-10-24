import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Title of the web app
st.title("Diabetes Linear Regression Web App")

# Load the diabetes dataset
diabetes_data = load_diabetes(as_frame=True)
df = diabetes_data.frame

# Display the dataset
st.write("Diabetes Dataset:")
st.write(df)

# Choose the target and feature columns
columns = df.columns.tolist()
target_column = st.selectbox("Outcome", columns)

# Ensure that the default values are included in the options list
# Here, we are using the same options as before, but we need to ensure the default values are valid
options = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
# Corrected the multiselect function to pass the options as a list
# Make sure the default values are valid options
feature_columns = st.multiselect("Select Features", 
                                   options=options, 
                                   default=[col for col in options if col in columns[:-1]])  # Only include valid defaults


# Check if any feature columns are selected
if feature_columns:
    # Splitting the data
    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Building the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Making predictions
    y_pred = model.predict(X_test)

    # Calculating R-squared score
    r2 = r2_score(y_test, y_pred)

    # Display the R² score
    st.write(f"R² Score: {r2:.2f}")

    # Display model coefficients
    coefficients = pd.DataFrame(model.coef_, feature_columns, columns=['Coefficient'])
    st.write("Model Coefficients:")
    st.write(coefficients)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.title('Actual vs Predicted values')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Diagonal line
    st.pyplot(plt)

    # Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs Predicted')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    st.pyplot(plt)

else:
    st.warning("Please select at least one feature column.")

# Instructions for the user
st.write("""
### Instructions:
1. Select the target column (usually 'progression').
2. Select one or more feature columns from the diabetes dataset.
3. The app will perform linear regression and display the R² score, model coefficients, and plots of actual vs predicted values.
""")
