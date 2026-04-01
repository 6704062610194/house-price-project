import streamlit as st

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction")

st.markdown("""
## 📌 รายละเอียดโปรเจค

โปรเจคนี้พัฒนาโมเดล Machine Learning และ Neural Network  
เพื่อทำนายราคาบ้านจากคุณสมบัติต่าง ๆ เช่น:

- 📐 ขนาดพื้นที่ (Lot Area)
- ⭐ คุณภาพบ้าน (Overall Quality)
- 🏗️ ปีที่สร้าง (Year Built)
- 🏠 พื้นที่ใช้สอย (Living Area)
- 🚗 จำนวนที่จอดรถ (Garage Cars)

---

## 🤖 โมเดลที่ใช้

### 🔹 Machine Learning (Ensemble)
ใช้การรวมโมเดลหลายตัว:
- Random Forest
- Gradient Boosting
- XGBoost

### 🔹 Neural Network
ใช้ Multi-Layer Perceptron (MLP)  
เพื่อเรียนรู้ความสัมพันธ์เชิงซ้อนของข้อมูล

---

## 🚀 วิธีใช้งาน
👉 เลือกเมนูด้านซ้ายเพื่อ:
- ทดสอบ Machine Learning Model
- ทดสอบ Neural Network Model
""")