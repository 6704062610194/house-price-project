import pandas as pd
import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# -------------------
# โหลดข้อมูล
# -------------------
df = pd.read_csv("data/train.csv")

# 🔥 ใช้เฉพาะ feature สำคัญ
features = ["LotArea", "OverallQual", "YearBuilt", "GrLivArea", "GarageCars"]

X = df[features]
y = np.log1p(df["SalePrice"])

# -------------------
# เตรียมข้อมูล
# -------------------
X = X.fillna(X.mean())

joblib.dump(features, "models/columns.pkl")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "models/scaler.pkl")

# -------------------
# Ensemble Model
# -------------------
rf = RandomForestRegressor(random_state=42)
gb = GradientBoostingRegressor(random_state=42)
xgb = XGBRegressor(random_state=42)

ensemble = VotingRegressor([
    ('rf', rf),
    ('gb', gb),
    ('xgb', xgb)
])

ensemble.fit(X_scaled, y)
joblib.dump(ensemble, "models/ensemble_model.pkl")

# -------------------
# Neural Network (นิ่งแน่นอน)
# -------------------
nn = MLPRegressor(
    hidden_layer_sizes=(16, 8),   # 🔥 เล็กลงมาก
    learning_rate_init=0.001,
    max_iter=2000,
    early_stopping=True,
    random_state=42
)

nn.fit(X_scaled, y)
joblib.dump(nn, "models/nn_model.pkl")

print("✅ Train เสร็จแล้ว (เวอร์ชันเสถียร)")