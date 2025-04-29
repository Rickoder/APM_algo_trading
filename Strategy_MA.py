import pandas as pd 
import numpy as np 
import os
import useful_function as uf
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt

import sklearn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv1D, GRU, Dense, Dropout, BatchNormalization, Conv1D, GRU, Dense, Reshape, Input, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler

import matplotlib as mpl
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)



# To plot pretty figures

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
plt.rc('figure', figsize=(5.0, 3.0))


df = pd.read_csv('DF_data_cleaned.csv', parse_dates=['Date'], index_col='Date')


ma_windows_values = [5, 10, 20, 50, 70, 100, 150, 200, 250, 300, 400]
rsi_window_values = [5, 10, 15, 20, 30, 50]

def df_creator(stock, ma_windows, rsi_windows): 

    data = {
             #"Prices": stock.values, 
             "Returns": stock.shift(1).pct_change().dropna(),
             #"log_Returns": np.log(stock.shift(1)).diff().dropna(),
              "Volatility": stock.pct_change().shift(1).rolling(window=30).std() * np.sqrt(252),
              "day_of_week": stock.index.dayofweek,
              "is_month_end": stock.index.is_month_start.astype(int),
              "is_month_start": stock.index.is_month_start.astype(int)
             }

    for window in ma_windows:
        data[f'ema_{window}'] = stock.shift(1).ewm(span=window, adjust=False).mean()
        data[f'sma_{window}'] = stock.shift(1).rolling(window=window).mean()


        
    delta = stock.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    for window in rsi_windows:
        avg_gain = gain.shift(1).rolling(window=window).mean()
        avg_loss = loss.shift(1).rolling(window=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        data[f'rsi_{window}'] = rsi

    df = pd.DataFrame(data).dropna()

    scaler = MinMaxScaler() 
    scaled_data = scaler.fit_transform(df)  

    
    df['return_direction'] = (df['Returns'] > 0).astype(int)
    df = df.drop('Returns', axis=1)
    return scaled_data, df["return_direction"].dropna().values


# X_train_all = []
# y_train_all = []

# X_test_all = []
# y_test_all = []

# timesteps = 20
# batch_size = 32

# for col in df.columns:
#     scaled_data, labels = df_creator(df[col], ma_windows=ma_windows_values, rsi_windows=rsi_window_values)
#     print(f"\n📊 {col} — scaled_data shape: {scaled_data.shape}, labels shape: {labels.shape}")

    
#     labels = labels[-len(scaled_data):]

#     # Train/test split per stock
#     split_idx = int(len(scaled_data) * 0.8)

#     X_stock_train = scaled_data[:split_idx]
#     y_stock_train = labels[:split_idx]

#     X_stock_test = scaled_data[split_idx:]
#     y_stock_test = labels[split_idx:]

#     print(f"Train samples: {len(X_stock_train)}, Test samples: {len(X_stock_test)}")

#     # Create time series generators
#     train_generator = TimeseriesGenerator(X_stock_train, y_stock_train,
#                                           length=timesteps, batch_size=batch_size)

#     test_generator = TimeseriesGenerator(X_stock_test, y_stock_test,
#                                          length=timesteps, batch_size=batch_size)

#     print(f"Train generator for {col} has {len(train_generator)} batches")

  

# scaled_data, returns_only = df_creator(df['AAPL_adjclose'], ma_windows_values, rsi_window_values)
scaled_data, labels = df_creator(df['AAPL_adjclose'], ma_windows_values, rsi_window_values)

labels = labels[-len(scaled_data):]

timesteps = 30 # hyperparmeter to adjust later 
batch_size = 16

# Split data into training and testing sets (80% train, 20% test)
split_idx = int(len(scaled_data) * 0.8)

X_train = scaled_data[:split_idx]
y_train = labels[:split_idx]  # SAME LENGTH as X_train

X_test = scaled_data[split_idx:]
y_test = labels[split_idx:]


train_generator = TimeseriesGenerator(X_train, y_train,
                                      length=timesteps, batch_size=batch_size)

test_generator = TimeseriesGenerator(X_test, y_test,
                                     length=timesteps, batch_size=batch_size)




# Let's examine the first batch
for X_batch, y_batch in train_generator:
    print("X_batch shape:", X_batch.shape)  # Shape of the input sequences
    # Shape of the corresponding target values
    print("y_batch shape:", y_batch.shape)

    # Display the first sequence and target
    print("\nFirst sequence (X):")
    print(X_batch[0])  # First sequence in the batch

    print("\nFirst target (y):")
    print(y_batch[0])  # First target value corresponding to the first sequence

    # Break after the first batch to avoid printing all batches
    break

n_features = scaled_data.shape[1]

def build_model(input_shape):
    model = Sequential()
    
    model.add(Input(shape=input_shape))  # e.g., (timesteps, n_features)
    
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(GRU(32, return_sequences=False))
    model.add(Dropout(0.3))
    
    model.add(Dense(32, activation='relu'))2
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy', 'Precision', 'Recall', 'AUC']
    )
    
    return model




modello = build_model((timesteps, n_features))

# Print model summary
modello.summary()


history = modello.fit(
    train_generator, validation_data=test_generator, epochs=20, verbose=1)