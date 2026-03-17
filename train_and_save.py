"""
Run this ONCE to train the LSTM on full AAPL history and save it.
After running, app.py will load the saved model instantly every time.

Usage:
    E:\anaconda3\python.exe train_and_save.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

TIME_STEP  = 100
EPOCHS     = 100
BATCH_SIZE = 64

print("Downloading AAPL data (2010 → today)…")
df = yf.download("AAPL", start="2010-01-01", progress=False)[["Close"]].dropna()
close = df["Close"].astype(float).values.reshape(-1, 1)
print(f"  {len(close)} trading days loaded.")

# Scale
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(close)

# Create sequences
def create_dataset(data, time_step=100):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:i+time_step, 0])
        y.append(data[i+time_step, 0])
    return np.array(X), np.array(y)

# Use 65% for train, rest for val
split = int(len(scaled) * 0.65)
train_data = scaled[:split]
val_data   = scaled[split:]

X_train, y_train = create_dataset(train_data, TIME_STEP)
X_val,   y_val   = create_dataset(val_data,   TIME_STEP)

X_train = X_train.reshape(*X_train.shape, 1)
X_val   = X_val.reshape(*X_val.shape, 1)

# Build model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(TIME_STEP, 1)),
    LSTM(50, return_sequences=True),
    LSTM(50),
    Dense(1),
])
model.compile(loss="mean_squared_error", optimizer="adam")

print(f"\nTraining for {EPOCHS} epochs…")
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
)

# Save
model.save("aapl_lstm_model.keras")
joblib.dump(scaler, "aapl_scaler.pkl")
df.to_csv("AAPL_full.csv")

print("\nSaved:")
print("  aapl_lstm_model.keras")
print("  aapl_scaler.pkl")
print("  AAPL_full.csv")
print("\nDone! Run 'streamlit run app.py' to launch the dashboard.")
