import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime
import time

# Custom styles for a modern look
st.markdown(
    """
    <style>
    body {
        background-image: url('https://your-image-url.com');  /* Replace with your image URL */
        background-size: cover;
        background-position: center;
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        max-width: 1200px;
        margin: auto;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        background-color: rgba(255, 255, 255, 0.8); /* Semi-transparent background */
        border-radius: 10px;
    }
    .stButton button {
        background-color: #007BFF;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #0056b3;
    }
    .stTitle {
        color: #007BFF;
        font-weight: bold;
        font-size: 36px;
        text-align: center;
        margin-bottom: 20px;
    }
    .stSubheader {
        color: #0056b3;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #007BFF;
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: rgba(255, 255, 255, 0.8);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to get current time in HH:MM:SS format
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Function to get current date
def get_current_date():
    return datetime.now().date()

# Display live clock using a placeholder
clock_placeholder = st.empty()

# App title
st.title("🏠 House Price Prediction")

# Display moving clock in real time
while True:
    current_time = get_current_time()
    clock_placeholder.write(f"**Current Date and Time**: {current_time}")
    time.sleep(1)  # Update the clock every second

    # Display current date on calendar (date picker)
    st.subheader("📅 Current Date:")
    selected_date = st.date_input("Pick a date:", value=get_current_date())
    st.write(f"**Selected Date:** {selected_date}")

# The rest of your code for data processing and model prediction
st.write("Welcome to the modern house price prediction app. Upload your dataset, explore the data, and make predictions with ease.")

# File uploader for dataset
df_file = st.file_uploader("📂 Upload your dataset (CSV format):", type="csv")
if df_file is not None:
    df = pd.read_csv(df_file)
    st.subheader("📊 Dataset Preview (First 5 Rows):")
    st.dataframe(df.head())  # Display only the first 5 rows

    # Splitting dataset into features and target
    if 'Y house price of unit area' in df.columns:
        X = df.drop(columns=['Y house price of unit area'])
        y = df['Y house price of unit area']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model training
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        st.subheader("📈 Model Performance:")
        st.write(f"**Mean Squared Error:** {mse}")

        # Input features for prediction
        st.subheader("🛠️ Make a Prediction:")
        st.write("Enter the values for the input features below:")
        input_data = {}
        for col in X.columns:
            input_data[col] = st.number_input(f"{col}", value=0.0)

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        st.subheader("🔮 Prediction Result:")
        st.write(f"**Predicted House Price:** {prediction}")
    else:
        st.error("The dataset must contain the 'Y house price of unit area' column.")
else:
    st.info("Awaiting CSV file upload. Upload a file to get started.")

# Copyright footer
st.markdown(
    """
    <div class="footer">
        &copy; 2025 Evans Okoth. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
