import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.title("Stock Prediction App")

# Load data
df = pd.read_csv("stock_market_dataset.csv")

# Preprocessing
df["Date"] = pd.to_datetime(df["Date"])
df["day"] = df["Date"].dt.day
df["month"] = df["Date"].dt.month
df["dayofweek"] = df["Date"].dt.dayofweek

le = LabelEncoder()
df["Stock_id"] = le.fit_transform(df["Stock"])

df = df.drop(columns=["Stock", "Date"])

X = df.drop(columns=["Next_Close", "Target"])
y = df["Next_Close"]

# Train model
model = XGBRegressor()
model.fit(X, y)

# UI inputs
st.write("Enter values:")

open_price = st.number_input("Open Price")
high_price = st.number_input("High Price")
low_price = st.number_input("Low Price")

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "Open": open_price,
        "High": high_price,
        "Low": low_price,
        "Close": open_price,
        "Volume": 1000,
        "SMA_10": open_price,
        "RSI": 50,
        "MACD": 0,
        "Bollinger_Upper": high_price,
        "Bollinger_Lower": low_price,
        "GDP_Growth": 2,
        "Inflation_Rate": 5,
        "Interest_Rate": 3,
        "Sentiment_Score": 0.5,
        "day": 1,
        "month": 1,
        "dayofweek": 1,
        "Stock_id": 0
    }])

    prediction = model.predict(input_data)

    st.success(f"Predicted Price: {prediction[0]}")