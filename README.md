# Stock-Price-Forecasting-with-Bidirectional-LSTM
This project explores how different modeling approaches behave on long-horizon equity data.

Tested traditional ML models (RF, SVR, XGBoost) and deep learning architectures (LSTM, GRU, CNN-LSTM).
The Bidirectional LSTM achieved the highest statistical fit on daily closing prices (2015–2025).

Key takeaway:
High predictive accuracy does not imply tradable insight.
The model captures strong temporal dependence in prices, but does not incorporate exogenous information, regime shifts, or transaction costs.

This project was primarily a learning exercise to understand:

where deep learning outperforms classical models

where it can be misleading without economic judgment

Tools: Python, TensorFlow/Keras, yfinance, scikit-learn.
