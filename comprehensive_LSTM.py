import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import keras_tuner as kt
import seaborn as sns
import math
from datetime import datetime, timedelta

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------
# 1. LOAD & PREPARE THE DATA
# ---------------------------
def load_data(file_path):
    """Load and prepare the stock data."""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

# Load the data
df = load_data('AAPL.csv')

# ---------------------------
# 2. FEATURE ENGINEERING
# ---------------------------
def add_technical_indicators(df):
    """Add technical indicators to the dataframe."""
    # Calculate moving averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Calculate price momentum
    df['Returns'] = df['Close'].pct_change()
    df['Returns_5d'] = df['Close'].pct_change(periods=5)
    df['Returns_20d'] = df['Close'].pct_change(periods=20)
    
    # Calculate volatility
    df['Volatility_5d'] = df['Returns'].rolling(window=5).std()
    df['Volatility_20d'] = df['Returns'].rolling(window=20).std()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # Price distance from moving averages
    df['Price_SMA5_Ratio'] = df['Close'] / df['SMA_5']
    df['Price_SMA20_Ratio'] = df['Close'] / df['SMA_20']
    df['Price_SMA50_Ratio'] = df['Close'] / df['SMA_50']
    
    # Volume indicators
    df['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
    
    # Day of week, month, quarter (cyclical features)
    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    
    # Drop rows with NaN values (due to rolling calculations)
    df.dropna(inplace=True)
    
    return df

# Add technical indicators
df = add_technical_indicators(df)

# Feature correlation analysis
plt.figure(figsize=(16, 14))
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', 
            fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Drop highly correlated features (optional - uncomment if needed)
# corr_matrix = df.corr().abs()
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
# df = df.drop(to_drop, axis=1)

# ---------------------------
# 3. FEATURE SELECTION & SCALING
# ---------------------------
# Select features for model training
def select_features(df):
    # Exclude the target variable (Close price) from features
    # Also exclude Date-based features and non-numeric columns for LSTM
    exclude_columns = ['Day_of_Week', 'Month', 'Quarter']
    
    # Identify and exclude non-numeric columns (like 'Symbol')
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    exclude_columns.extend(non_numeric_columns)
    
    # Select only numeric columns for features
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    
    # Create a copy of the selected features
    data = df[feature_columns].copy()
    
    return data, feature_columns

data, feature_columns = select_features(df)
print(f"Selected features: {feature_columns}")
print(f"Data shape: {data.shape}")

# ---------------------------
# 4. SCALE THE FEATURES
# ---------------------------
def scale_features(data, feature_columns):
    """Scale each feature independently."""
    scalers = {}
    scaled_data = np.zeros((len(data), len(feature_columns)))
    
    for i, column in enumerate(feature_columns):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data[:, i] = scaler.fit_transform(data[column].values.reshape(-1, 1)).flatten()
        scalers[column] = scaler
    
    return scaled_data, scalers

scaled_data, scalers = scale_features(data, feature_columns)

# ---------------------------
# 5. CREATE DATASETS FOR TRAINING
# ---------------------------
def create_multi_step_dataset(dataset, time_step, future_days, target_col_idx):
    """
    Create a multi-step dataset for time series forecasting.
    
    Parameters:
    - dataset: The scaled dataset.
    - time_step: Number of past time steps to use for prediction (lookback window).
    - future_days: Number of future time steps to predict.
    - target_col_idx: Index of the target column to predict (e.g., Close price).
    
    Returns:
    - X: Input sequences (samples, time_step, features).
    - y: Target sequences (samples, future_days).
    """
    X, y = [], []
    
    for i in range(len(dataset) - time_step - future_days + 1):
        # Input sequence (all features)
        X.append(dataset[i:i+time_step, :])
        
        # Target sequence (future values of the target column)
        y.append(dataset[i+time_step:i+time_step+future_days, target_col_idx])
    
    return np.array(X), np.array(y)

# Parameters
time_step = 60           # 60 days lookback window
future_days = 5          # Predict 5 days ahead
target_col_idx = 3       # Index of Close price in feature_columns

# Find the index of 'Close' in feature_columns
try:
    target_col_idx = feature_columns.index('Close')
except ValueError:
    print("'Close' not found in feature_columns, using default index")

# Create the dataset
X, y = create_multi_step_dataset(scaled_data, time_step, future_days, target_col_idx)
print(f"X shape: {X.shape}")  # (samples, time_step, features)
print(f"y shape: {y.shape}")  # (samples, future_days)

# ---------------------------
# 6. TRAIN-TEST-VALIDATION SPLIT
# ---------------------------
def train_test_val_split(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split the data into training, validation, and test sets respecting time order."""
    n = len(X)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    # Training set
    X_train, y_train = X[:train_size], y[:train_size]
    
    # Validation set
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    
    # Test set
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# Split the data
X_train, y_train, X_val, y_val, X_test, y_test = train_test_val_split(X, y)
print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Validation set: {X_val.shape}, {y_val.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")

# ---------------------------
# 7. DEFINE MODEL BUILDING FUNCTION WITH KERAS TUNER
# ---------------------------
def build_model(hp):
    """Build a model with hyperparameters to be tuned."""
    inputs = Input(shape=(time_step, X_train.shape[2]))
    
    # First LSTM layer - simplified for initial testing
    lstm_units_1 = hp.Int('lstm_units_1', min_value=32, max_value=128, step=32)
    dropout_rate_1 = hp.Float('dropout_rate_1', min_value=0.1, max_value=0.5, step=0.1)
    
    # Simplified model - removed bidirectional option for initial testing
    x = LSTM(lstm_units_1, return_sequences=True)(inputs)
    x = Dropout(dropout_rate_1)(x)
    
    # Second LSTM layer
    lstm_units_2 = hp.Int('lstm_units_2', min_value=32, max_value=128, step=32)
    dropout_rate_2 = hp.Float('dropout_rate_2', min_value=0.1, max_value=0.5, step=0.1)
    
    x = LSTM(lstm_units_2, return_sequences=False)(x)
    x = Dropout(dropout_rate_2)(x)
    
    # Output layer - direct connection for simplicity
    outputs = Dense(future_days)(x)
    
    # Compile the model
    model = Model(inputs=inputs, outputs=outputs)
    
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# ---------------------------
# 8. HYPERPARAMETER TUNING
# ---------------------------
def run_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    """Run hyperparameter tuning and return the best model."""
    # For quicker testing, you might want to reduce max_trials or use RandomSearch
    tuner = kt.BayesianOptimization(
        build_model,
        objective='val_loss',
        max_trials=10,  # Reduced for faster execution; increase for better results
        directory='hyperparam_tuning',
        project_name='lstm_stock_prediction',
        overwrite=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Search for the best hyperparameters
    tuner.search(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Get the best hyperparameters
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best Hyperparameters:")
    for param, value in best_hp.values.items():
        print(f"{param}: {value}")
    
    # Build the best model
    best_model = tuner.hypermodel.build(best_hp)
    
    return best_model, best_hp

# Run hyperparameter tuning
best_model, best_hp = run_hyperparameter_tuning(X_train, y_train, X_val, y_val)

# ---------------------------
# 9. TRAIN THE FINAL MODEL WITH BEST HYPERPARAMETERS
# ---------------------------
def train_final_model(model, X_train, y_train, X_val, y_val):
    """Train the final model with the best hyperparameters."""
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return model, history

# Train the final model
final_model, history = train_final_model(best_model, X_train, y_train, X_val, y_val)

# Save the model
final_model.save('best_stock_prediction_model.h5')

# ---------------------------
# 10. PLOT TRAINING HISTORY
# ---------------------------
def plot_training_history(history):
    """Plot the training and validation loss."""
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    plt.close()
    
    # Plot MAE if available
    if 'mae' in history.history:
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_mae.png')
        plt.close()

# Plot training history
plot_training_history(history)

# ---------------------------
# 11. EVALUATE THE MODEL
# ---------------------------
def evaluate_model(model, X_test, y_test, scalers, feature_columns, target_col_idx):
    """Evaluate the model on the test set."""
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get the scaler for the target column
        target_column = feature_columns[target_col_idx]
        target_scaler = scalers[target_column]
        
        # Calculate metrics on scaled data
        mse = mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
    
        print("Evaluation Metrics (Scaled):")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
    
        # Plot predictions for the last test sample
        plt.figure(figsize=(12, 6))
        plt.plot(y_test[-1], marker='o', label='True Future Values (scaled)')
        plt.plot(y_pred[-1], marker='x', label='Predicted Future Values (scaled)')
        plt.title('Future Values Prediction (Scaled) - Last Test Sample')
        plt.xlabel('Days Ahead')
        plt.ylabel('Scaled Value')
        plt.legend()
        plt.grid(True)
        plt.savefig('future_values_prediction.png')
        plt.close()
    
        # Unscale the predictions
        y_test_unscaled = np.array([target_scaler.inverse_transform(y.reshape(-1, 1)).flatten() for y in y_test])
        y_pred_unscaled = np.array([target_scaler.inverse_transform(y.reshape(-1, 1)).flatten() for y in y_pred])
        
        # Calculate metrics on unscaled data
        mse_unscaled = mean_squared_error(y_test_unscaled, y_pred_unscaled)
        rmse_unscaled = math.sqrt(mse_unscaled)
        mae_unscaled = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
        
        print("\nEvaluation Metrics (Unscaled):")
        print(f"MSE: {mse_unscaled:.6f}")
        print(f"RMSE: {rmse_unscaled:.6f}")
        print(f"MAE: {mae_unscaled:.6f}")
    
    except Exception as e:
        print(f"Error in model evaluation: {e}")
        return np.array([]), np.array([])
    
    return y_test_unscaled, y_pred_unscaled

# Evaluate the model
try:
    y_test_unscaled, y_pred_unscaled = evaluate_model(final_model, X_test, y_test, scalers, feature_columns, target_col_idx)
except Exception as e:
    print(f"Warning: Could not evaluate model: {e}")
    y_test_unscaled, y_pred_unscaled = np.array([]), np.array([])

# ---------------------------
# 12. FORECAST FUTURE PRICES
# ---------------------------
def forecast_future_prices(model, df, scaled_data, time_step, future_days, scalers, feature_columns, target_col_idx):
    """Forecast future stock prices."""
    try:
        # Get the last time_step days of data for the prediction
        last_sequence = scaled_data[-time_step:].reshape(1, time_step, len(feature_columns))
        
        # Predict future values
        predicted_scaled = model.predict(last_sequence)[0]
        
        # Get the scaler for the target column
        target_column = feature_columns[target_col_idx]
        target_scaler = scalers[target_column]
        
        # Unscale the predictions
        predicted_unscaled = target_scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).flatten()
        
        # Get the last known date and price
        last_date = df.index[-1]
        last_price = df['Close'][-1]
        
        # Generate future dates (business days)
        future_dates = []
        next_date = last_date
        for _ in range(future_days):
            next_date = next_business_day(next_date)
            future_dates.append(next_date)
        
        # Create a DataFrame with the forecast
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecasted_Close': predicted_unscaled
        })
        forecast_df.set_index('Date', inplace=True)
        
        # Plot historical prices with forecast
        plt.figure(figsize=(16, 8))
        
        # Plot historical close prices
        plt.plot(df.index[-250:], df['Close'][-250:], label='Historical Close Price')
        
        # Plot forecasted prices
        plt.plot(forecast_df.index, forecast_df['Forecasted_Close'], 'r-o', label='Forecasted Close Price')
        
        plt.title('Historical Close Price with Forecasted Future Prices')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('price_forecast.png')
        plt.close()
    

    
    except Exception as e:
        print(f"Error in forecasting future prices: {e}")
        # Return empty dataframe in case of error
        return pd.DataFrame()
    
    return forecast_df

def next_business_day(date):
    """Get the next business day (excluding weekends)."""
    next_day = date + timedelta(days=1)
    # If it's a weekend, move to Monday
    if next_day.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
        next_day += timedelta(days=7 - next_day.weekday())
    return next_day

# Forecast future prices
try:
    forecast_df = forecast_future_prices(
        final_model, df, scaled_data, time_step, future_days, 
        scalers, feature_columns, target_col_idx
    )
except Exception as e:
    print(f"Warning: Could not forecast future prices: {e}")
    forecast_df = pd.DataFrame()

print("Future Price Forecast:")
print(forecast_df)

# ---------------------------
# 13. TRADING STRATEGY EVALUATION
# ---------------------------
def evaluate_trading_strategy(y_test_unscaled, y_pred_unscaled, df, test_indices):
    """Evaluate a simple trading strategy based on model predictions."""
    try:
        trading_results = []
        
        for i in range(len(y_test_unscaled)):
            true_next_day = y_test_unscaled[i][0]
            pred_next_day = y_pred_unscaled[i][0]
            current_price = df['Close'].iloc[test_indices[i]]
            
            # Strategy: Buy if model predicts price will go up, sell if model predicts price will go down
            if pred_next_day > current_price:
                position = 'Buy'
            else:
                position = 'Sell'
            
            # Actual outcome
            if true_next_day > current_price:
                actual_movement = 'Up'
                if position == 'Buy':
                    result = 'Win'
                else:
                    result = 'Loss'
            else:
                actual_movement = 'Down'
                if position == 'Sell':
                    result = 'Win'
                else:
                    result = 'Loss'
            
            # Calculate profit/loss percentage
            pnl_pct = ((true_next_day - current_price) / current_price) * 100
            if position == 'Sell':
                pnl_pct = -pnl_pct
            
            trading_results.append({
                'Date': df.index[test_indices[i]],
                'Current_Price': current_price,
                'True_Next_Day': true_next_day,
                'Pred_Next_Day': pred_next_day,
                'Position': position,
                'Actual_Movement': actual_movement,
                'Result': result,
                'PnL_Pct': pnl_pct
            })
        
        # Create DataFrame
        strategy_df = pd.DataFrame(trading_results)
        
        # Calculate strategy performance
        total_trades = len(strategy_df)
        winning_trades = len(strategy_df[strategy_df['Result'] == 'Win'])
        win_rate = winning_trades / total_trades * 100
        
        cumulative_return = strategy_df['PnL_Pct'].sum()
        average_return = strategy_df['PnL_Pct'].mean()
        
        # Print strategy results
        print("\nTrading Strategy Evaluation:")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Cumulative Return: {cumulative_return:.2f}%")
        print(f"Average Return per Trade: {average_return:.2f}%")
        
        # Plot cumulative returns
        plt.figure(figsize=(12, 6))
        plt.plot(strategy_df['Date'], strategy_df['PnL_Pct'].cumsum(), label='Cumulative Return (%)')
        plt.title('Trading Strategy Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('trading_strategy_returns.png')
        plt.close()
    
    except Exception as e:
        print(f"Error in trading strategy evaluation: {e}")
        return pd.DataFrame()
    
    return strategy_df

try:
    # Calculate test indices
    test_indices = list(range(len(X) - len(X_test), len(X)))

    # Evaluate trading strategy (optional - uncomment if needed)
    # strategy_df = evaluate_trading_strategy(y_test_unscaled, y_pred_unscaled, df, test_indices)
except Exception as e:
    print(f"Warning: Could not evaluate trading strategy: {e}")

# ---------------------------
# 14. SUMMARY & RECOMMENDATIONS
# ---------------------------
def generate_summary():
    """Generate a summary of the model and predictions."""
    try:
        print("\n====== MODEL SUMMARY ======")
        print("Model Architecture:")
        final_model.summary()
        
        print("\nBest Hyperparameters:")
        for param, value in best_hp.values.items():
            print(f"{param}: {value}")
        
        print("\nFuture Price Forecast:")
        if not forecast_df.empty:
            print(forecast_df)
            
            print("\nRecommendations:")
            last_price = df['Close'][-1]
            future_prices = forecast_df['Forecasted_Close'].values
            
            # Short-term trend
            if future_prices[0] > last_price:
                short_term = "Bullish (Up)"
            else:
                short_term = "Bearish (Down)"
            
            # Overall trend
            if future_prices[-1] > last_price:
                overall_trend = "Bullish (Up)"
            else:
                overall_trend = "Bearish (Down)"
            
            print(f"Current Price: ${last_price:.2f}")
            print(f"Short-term Trend (1 day): {short_term}")
            print(f"Overall Trend ({future_days} days): {overall_trend}")
            
            # Action recommendation
            if overall_trend == "Bullish (Up)":
                print("Recommended Action: Consider Buy/Hold positions")
            else:
                print("Recommended Action: Consider Sell/Short positions")
        else:
            print("No forecast data available.")
        
        print("\nCAUTION: This model is for educational purposes only. Real trading requires additional analysis and risk management.")
    except Exception as e:
        print(f"Error generating summary: {e}")

# Generate summary
try:
    generate_summary()
except Exception as e:
    print(f"Error in summary generation: {e}")
    
print("\nProcess completed. Check the generated visualizations and metrics above.")