import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


class LSTMDataPreparation:
    """Prepare sequence data for LSTM/GRU/CNN models"""

    @staticmethod
    def create_sequences(data, seq_length):
        """
        Create sequences for time-series models
        
        Args:
            data (np.array): 1D array of prices
            seq_length (int): Length of each sequence
            
        Returns:
            tuple: (X, y) where X is [samples, seq_length, 1]
        """
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    @staticmethod
    def prepare_lstm_data(train_df, test_df, seq_length=60):
        """
        Full preparation pipeline for LSTM data
        
        Args:
            train_df, test_df (pd.DataFrame): Data with Close column
            seq_length (int): Sequence length
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test, scaler)
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Fit scaler on training data
        train_scaled = scaler.fit_transform(train_df["Close"].values.reshape(-1, 1))
        test_scaled = scaler.transform(test_df["Close"].values.reshape(-1, 1))
        
        # Create sequences for training
        X_train, y_train = LSTMDataPreparation.create_sequences(train_scaled, seq_length)
        
        # For test, combine last seq_length training points with test
        full_data = np.concatenate([train_scaled, test_scaled])
        test_input = full_data[len(train_scaled) - seq_length:]
        X_test, y_test = LSTMDataPreparation.create_sequences(test_input, seq_length)
        
        # Reshape for LSTM [samples, time_steps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        print(f"\n[LSTM Data Preparation]")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  y_train shape: {y_train.shape}")
        print(f"  X_test shape: {X_test.shape}")
        print(f"  y_test shape: {y_test.shape}")
        
        return X_train, y_train, X_test, y_test, scaler


class LSTMModel:
    """LSTM (Long Short-Term Memory) neural network"""

    @staticmethod
    def build_lstm(seq_length, units1=50, units2=50, units3=25):
        """
        Build stacked LSTM architecture
        
        Args:
            seq_length (int): Sequence length
            units1, units2, units3 (int): Units for each layer
            
        Returns:
            Sequential: Uncompiled model
        """
        model = Sequential([
            LSTM(units1, return_sequences=True, input_shape=(seq_length, 1)),
            Dropout(0.2),
            LSTM(units2, return_sequences=False),
            Dropout(0.2),
            Dense(units3),
            Dense(1)
        ])
        
        return model

    @staticmethod
    def compile_and_train(model, X_train, y_train, config):
        """
        Compile and train LSTM
        
        Args:
            model: Sequential model
            X_train, y_train: Training data
            config (dict): LSTM config (epochs, batch_size, etc.)
            
        Returns:
            Sequential: Trained model
        """
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="mean_squared_error"
        )
        
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
        
        print("\n[LSTM] Training...")
        model.fit(
            X_train, y_train,
            batch_size=config.get("batch_size", 32),
            epochs=config.get("epochs", 20),
            validation_split=config.get("validation_split", 0.1),
            callbacks=[early_stop],
            verbose=1
        )
        
        return model

    @staticmethod
    def predict_lstm(model, X_test, scaler):
        """
        Make predictions and inverse transform
        
        Args:
            model: Trained model
            X_test: Test sequences
            scaler: MinMaxScaler for inverse transform
            
        Returns:
            np.array: Predictions in original scale
        """
        print("\n[LSTM] Predicting...")
        preds_scaled = model.predict(X_test)
        preds = scaler.inverse_transform(preds_scaled)
        
        return preds.ravel()

    @staticmethod
    def train_lstm_complete(train_df, test_df, config):
        """Complete LSTM pipeline"""
        X_train, y_train, X_test, y_test, scaler = LSTMDataPreparation.prepare_lstm_data(
            train_df, test_df,
            seq_length=config.get("seq_length", 60)
        )
        
        model = LSTMModel.build_lstm(config.get("seq_length", 60))
        model = LSTMModel.compile_and_train(model, X_train, y_train, config)
        predictions = LSTMModel.predict_lstm(model, X_test, scaler)
        
        return predictions


class GRUModel:
    """GRU (Gated Recurrent Unit) - faster than LSTM"""

    @staticmethod
    def build_gru(seq_length, units1=50, units2=50, units3=25):
        """Build stacked GRU architecture"""
        model = Sequential([
            GRU(units1, return_sequences=True, input_shape=(seq_length, 1)),
            Dropout(0.2),
            GRU(units2, return_sequences=False),
            Dropout(0.2),
            Dense(units3),
            Dense(1)
        ])
        
        return model

    @staticmethod
    def train_gru_complete(train_df, test_df, config):
        """Complete GRU pipeline"""
        X_train, y_train, X_test, y_test, scaler = LSTMDataPreparation.prepare_lstm_data(
            train_df, test_df,
            seq_length=config.get("seq_length", 60)
        )
        
        model = GRUModel.build_gru(config.get("seq_length", 60))
        model = LSTMModel.compile_and_train(model, X_train, y_train, config)
        predictions = LSTMModel.predict_lstm(model, X_test, scaler)
        
        return predictions


class CNNModel:
    """1D CNN for temporal pattern detection"""

    @staticmethod
    def build_cnn(seq_length):
        """Build 1D CNN architecture"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation="relu",
                   input_shape=(seq_length, 1)),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation="relu"),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(50, activation="relu"),
            Dense(1)
        ])
        
        return model

    @staticmethod
    def train_cnn_complete(train_df, test_df, config):
        """Complete CNN pipeline"""
        X_train, y_train, X_test, y_test, scaler = LSTMDataPreparation.prepare_lstm_data(
            train_df, test_df,
            seq_length=config.get("seq_length", 60)
        )
        
        model = CNNModel.build_cnn(config.get("seq_length", 60))
        model = LSTMModel.compile_and_train(model, X_train, y_train, config)
        predictions = LSTMModel.predict_lstm(model, X_test, scaler)
        
        return predictions