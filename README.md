# Retail Demand Forecasting using LSTM

A deep learning solution for predicting next-day product demand across multiple retail stores using LSTM networks.

## 📌 Overview

This project implements a Long Short-Term Memory (LSTM) neural network to forecast retail product demand at a daily granularity. The model processes historical transactional data along with engineered temporal features to make accurate next-day predictions, enabling better inventory management and business planning.

## 🛠️ Technical Stack

- **Programming Language**: Python
- **Libraries**: 
  - TensorFlow/Keras (Deep Learning)
  - Pandas (Data Manipulation)
  - NumPy (Numerical Operations)
  - Scikit-learn (Data Preprocessing & Metrics)
  - Matplotlib/Seaborn (Visualization)

## 📂 Dataset

- **Size**: 73,000+ rows of daily transactional records
- **Features**:
  - Historical demand quantities
  - Store/product identifiers
  - Engineered temporal features:
    - Rolling averages (7-day, 14-day, 30-day)
    - Seasonal encodings (cyclic sin/cos for day/month)
    - Lag features (t-1, t-7, t-30 demand)
    - Day-of-week indicators
  - External variables (if available)

## 🧠 Model Architecture

```python
# Example LSTM architecture
model = Sequential([
    LSTM(64, input_shape=(lookback_window, n_features),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
