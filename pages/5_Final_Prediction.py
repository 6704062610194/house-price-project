import streamlit as st
import joblib
import pandas as pd
from tensorflow.keras.models import load_model

# โหลดโมเดล
ml_model = joblib.load("models/ensemble_model.pkl")
nn_model = load_model("models/nn_model.h5", compile=False)

scaler = joblib.load("models/scaler.pkl")
columns = joblib.load("models/columns.pkl")

st.title("🏠 House Price Prediction (Final)")

st.info("💡 ใช้ Machine Learning + Neural Network และค่าเฉลี่ยเพื่อเพิ่มความแม่นยำ")

# -------------------------
# INPUT
# -------------------------
area = st.number_input("Lot Area (sqft)", 1000, 20000, 8000)
quality = st.slider("Overall Quality (1-10)", 1, 10, 7)
year = st.number_input("Year Built", 1900, 2025, 2000)
living = st.number_input("Living Area (sqft)", 500, 5000, 1500)
garage = st.number_input("Garage Cars", 0, 5, 1)

# -------------------------
# PREDICT
# -------------------------
if st.button("Predict Price"):

    input_df = pd.DataFrame({
        "LotArea": [area],
        "OverallQual": [quality],
        "YearBuilt": [year],
        "GrLivArea": [living],
        "GarageCars": [garage]
    })

    # ทำให้ตรงกับตอน train
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # scale
    input_scaled = scaler.transform(input_df)

    # predict
    ml_price = ml_model.predict(input_scaled)[0]
    nn_price = nn_model.predict(input_scaled)[0][0]

    # final
    final_price = (ml_price + nn_price) / 2

    # -------------------------
    # OUTPUT
    # -------------------------
    st.subheader("📊 Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("💰 ML", f"{ml_price:,.0f} บาท")

    with col2:
        st.metric("🤖 NN", f"{nn_price:,.0f} บาท")

    with col3:
        st.metric("📊 Final", f"{final_price:,.0f} บาท")

    st.success("✅ Final Prediction เป็นค่าเฉลี่ยของทั้งสองโมเดล")

    st.warning("⚠️ Neural Network อาจคลาดเคลื่อน เนื่องจากใช้ feature บางส่วนในการ demo")