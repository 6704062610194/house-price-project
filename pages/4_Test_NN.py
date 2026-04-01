import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.cache_resource.clear()

model = joblib.load("models/nn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
columns = joblib.load("models/columns.pkl")

st.title("🤖 Test Neural Network")
st.warning("⚠️ Neural Network อาจคลาดเคลื่อนเล็กน้อย")

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

    # 🔥 กันค่าเพี้ยน
    if result[0] > 20:
        st.error("⚠️ โมเดล NN ยังไม่เสถียร กรุณา train ใหม่")
    else:
        result = np.expm1(result)
        st.metric("🤖 NN Prediction", f"{result[0]:,.0f} บาท")