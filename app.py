import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and columns
model = joblib.load("car_model.pkl")
columns = joblib.load("model_features.pkl")

st.title("🚗 Car Price Prediction App")

# Sidebar inputs
st.sidebar.header("Enter Car Details")

present_price = st.sidebar.slider("Present Price (Lakhs)", 0.0, 50.0, 5.0)
kms_driven = st.sidebar.slider("Kilometers Driven", 0, 200000, 30000)
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
selling_type = st.sidebar.selectbox("Selling Type", ["Dealer", "Individual"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.sidebar.selectbox("Previous Owners", [0, 1, 2, 3])
car_age = st.sidebar.slider("Car Age", 0, 20, 5)

# Encoding
fuel_enc = {"Petrol": 2, "Diesel": 0, "CNG": 1}[fuel_type]
sell_enc = {"Dealer": 0, "Individual": 1}[selling_type]
trans_enc = {"Manual": 1, "Automatic": 0}[transmission]

input_data = np.array([[present_price, kms_driven, fuel_enc, sell_enc, trans_enc, owner, car_age]])
df_input = pd.DataFrame(input_data, columns=columns)

prediction = model.predict(df_input)[0]

# Display Prediction
st.subheader("Predicted Selling Price")
st.success(f"₹ {prediction:.2f} Lakhs")

# Visualize prediction distribution (optional)
if st.checkbox("Show Sample Actual vs Predicted Plot"):
    y_true = np.linspace(1, 10, 20)
    y_pred = y_true + np.random.normal(0, 1, 20)
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, color='green')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted (Sample)")
    st.pyplot(fig)