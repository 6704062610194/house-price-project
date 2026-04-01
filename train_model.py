import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -------------------
# โหลดข้อมูล
# -------------------
df = pd.read_csv("data/train.csv")

# แยก target
y = df["SalePrice"]
X = df.drop("SalePrice", axis=1)

# -------------------
# เตรียมข้อมูล
# -------------------

# จัดการ missing
X = X.fillna(X.mean(numeric_only=True))

# แปลง categorical
X = pd.get_dummies(X)

# 🔥 เก็บ column names ไว้ใช้ตอน test (สำคัญมาก)
joblib.dump(X.columns, "models/columns.pkl")

# scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔥 save scaler ด้วย
joblib.dump(scaler, "models/scaler.pkl")

# -------------------
# Ensemble Model
# -------------------
rf = RandomForestRegressor()
gb = GradientBoostingRegressor()
xgb = XGBRegressor()

ensemble = VotingRegressor([
    ('rf', rf),
    ('gb', gb),
    ('xgb', xgb)
])

ensemble.fit(X_scaled, y)

joblib.dump(ensemble, "models/ensemble_model.pkl")

# -------------------
# Neural Network
# -------------------
nn = Sequential([
    Dense(128, activation='relu', input_shape=(X_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])

nn.compile(optimizer='adam', loss='mse')

nn.fit(X_scaled, y, epochs=20, batch_size=32)

nn.save("models/nn_model.h5")

print("Train เสร็จแล้ว!")
print("✅ ถึงขั้นตอน save columns แล้ว")
joblib.dump(X.columns, "models/columns.pkl")