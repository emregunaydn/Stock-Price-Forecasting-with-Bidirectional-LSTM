# Stock-Price-Forecasting-with-Bidirectional-LSTM
# This project explores how different modeling approaches behave on long-horizon equity data.

# Tested traditional ML models (RF, SVR, XGBoost) and deep learning architectures (LSTM, GRU, CNN-LSTM).
# The Bidirectional LSTM achieved the highest statistical fit on daily closing prices (2015–2025).

# Key takeaway:
# High predictive accuracy does not imply tradable insight.
# The model captures strong temporal dependence in prices, but does not incorporate exogenous information, regime shifts, or transaction costs.

# This project was primarily a learning exercise to understand:

# where deep learning outperforms classical models

# where it can be misleading without economic judgment

# Tools: Python, TensorFlow/Keras, yfinance, scikit-learn.


--------------------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense

# Step 1: Download IBM stock data ------ You can also basically switch IBM with any other companies' ticket which available in yfinance library
ticker = 'IBM'
start_date = '2015-01-01'
end_date = '2025-01-01'
df = yf.download(ticker, start=start_date, end=end_date)
df_close = df[['Close']].dropna()

# Step 2: Normalize closing prices
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_close)

# Step 3: Create sequences
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(df_scaled, seq_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Step 4: Split data into training and test sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Step 5: Build Bidirectional LSTM model
model = Sequential([
    Bidirectional(LSTM(50, return_sequences=True), input_shape=(seq_length, 1)),
    Bidirectional(LSTM(50)),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 6: Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Inverse transform predictions
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Step 9: Evaluation
mse = mean_squared_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)

print("\n📈 Bidirectional LSTM Model Performance:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")

# Step 10: Plot predictions
plt.figure(figsize=(12, 6))
dates = df_close.index[seq_length + split:]
plt.plot(dates, y_test_inv, label='Actual Prices')
plt.plot(dates, y_pred_inv, label='Predicted Prices')
plt.title(f'{ticker} Closing Price Prediction using Bidirectional LSTM')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
