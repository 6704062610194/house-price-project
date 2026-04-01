import streamlit as st

st.title("📊 Machine Learning Model")

st.header("🔹 Data Preparation")
st.write("""
- จัดการ missing values
- แปลง categorical เป็นตัวเลข
- ทำ feature scaling
""")

st.header("🔹 Model ที่ใช้")
st.write("""
- Random Forest
- Gradient Boosting
- XGBoost
""")

st.header("🔹 แนวคิด")
st.write("""
ใช้ Ensemble Model เพื่อรวมผลลัพธ์จากหลายโมเดล
ช่วยลด error และเพิ่มความแม่นยำ
""")