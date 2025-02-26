# LSTM-StockPrice-Advanced
With feature engineering
# Advanced LSTM Stock Price Prediction

An enhanced deep learning model for predicting stock prices using LSTMs with hyperparameter tuning.

## Overview

This project implements a comprehensive stock price prediction system using Long Short-Term Memory (LSTM) neural networks. The model incorporates multiple technical indicators, price patterns, and market features to forecast future stock prices with improved accuracy.

Price Forecast
![price_forecast](https://github.com/user-attachments/assets/8acb549a-48c0-408b-ba7a-dfcb3096cbb8)


## Features

- **Advanced Feature Engineering**
  - Technical indicators (RSI, MACD, Bollinger Bands)
  - Multiple timeframe momentum indicators
  - Volatility metrics
  - Price-to-moving average relationships
  - Volume analysis

- **LSTM Neural Network Architecture**
  - Multi-layer LSTM configuration
  - Dropout regularization to prevent overfitting
  - Hyperparameter tuning via Bayesian optimization

- **Comprehensive Evaluation**
  - Multiple error metrics (MSE, RMSE, MAE)
  - Visualization of prediction accuracy
  - Trading strategy simulation
  - Performance analysis

## Sample Results

The model was trained on Apple (AAPL) stock data and achieved the following results:

- **Scaled Metrics**:
  - MSE: 0.092163
  - RMSE: 0.303584
  - MAE: 0.290453

- **Unscaled Metrics**:
  - MSE: 3045.297624
  - RMSE: 55.184215
  - MAE: 52.797309

- **5-Day Forecast (example)**:

```
            Forecasted_Close
Date                        
2023-06-12         95.782661
2023-06-13         99.128990
2023-06-14        101.371292
2023-06-15         99.898003
2023-06-16         92.359787
```
![future_values_prediction](https://github.com/user-attachments/assets/638c3d4d-22a8-4215-9fe9-b4b8c923a718)
![training_mae](https://github.com/user-attachments/assets/eb9641cd-6651-4f2e-89fc-fae3b4a33563)
![training_history](https://github.com/user-attachments/assets/e558c20f-fb4f-4c6c-b4fc-8dd39795dd45)
![correlation_matrix](https://github.com/user-attachments/assets/f098bc3a-2358-4129-bf93-9b5bf5543a20)


## Model Architecture

```
Model: "functional_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_layer_1 (InputLayer)           │ (None, 60, 29)              │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_2 (LSTM)                        │ (None, 60, 64)              │          24,064 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_2 (Dropout)                  │ (None, 60, 64)              │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_3 (LSTM)                        │ (None, 32)                  │          12,416 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_3 (Dropout)                  │ (None, 32)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 5)                   │             165 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 109,937 (429.45 KB)
 Trainable params: 36,645 (143.14 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 73,292 (286.30 KB)
```

## Getting Started

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- pandas
- numpy
- matplotlib
- scikit-learn
- keras-tuner

### Usage

1. Place your stock data CSV file in the project directory
2. Update the file path in the script if necessary
3. Run the prediction model:

```bash
python comprehensive_LSTM.py
```

### Data Format

The model expects CSV data with the following columns:
- Date
- Open
- High
- Low
- Close
- Volume

Additional columns like "Symbol" and "YTD Gains" can be included but aren't required.

## Hyperparameter Tuning

The model uses Bayesian optimization to find optimal hyperparameters:

- LSTM layer sizes
- Dropout rates
- Learning rate
- Layer configuration

Best hyperparameters from the example run:
- lstm_units_1: 64
- dropout_rate_1: 0.2
- lstm_units_2: 32
- dropout_rate_2: 0.2
- learning_rate: 0.004188

## Limitations and Disclaimer

This model is for educational and research purposes only. Stock market prediction is inherently difficult and uncertain. The model doesn't account for:

- Unexpected market events
- Company-specific news
- Macroeconomic changes
- Market sentiment

Real trading requires additional analysis, risk management, and should not rely solely on algorithmic predictions.

## Future Improvements

- Implement ensemble methods combining multiple models
- Add sentiment analysis from news and social media
- Include macroeconomic indicators
- Develop more sophisticated trading strategies
- Improve prediction confidence intervals

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Keras and TensorFlow documentation
- Various finance and technical analysis resources
- Open-source deep learning community
