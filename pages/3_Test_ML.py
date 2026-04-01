import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/ensemble_model.pkl")
scaler = joblib.load("models/scaler.pkl")
columns = joblib.load("models/columns.pkl")

st.title("🏠 Test ML Model")

st.info("💡 ใส่ค่าบ้านเพื่อทำนายราคา")

# 🔥 เพิ่ม feature
area = st.number_input("Lot Area (sqft)", 1000, 20000, 8000)
quality = st.slider("Overall Quality (1-10)", 1, 10, 7)
year = st.number_input("Year Built", 1900, 2025, 2000)
living = st.number_input("Living Area (sqft)", 500, 5000, 1500)
garage = st.number_input("Garage Cars", 0, 5, 1)

if st.button("Predict"):
    input_df = pd.DataFrame({
        "LotArea": [area],
        "OverallQual": [quality],
        "YearBuilt": [year],
        "GrLivArea": [living],
        "GarageCars": [garage]
    })

    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=columns, fill_value=0)

    input_scaled = scaler.transform(input_df)

    result = model.predict(input_scaled)

    st.metric("💰 ML Prediction", f"{result[0]:,.0f} บาท")
    