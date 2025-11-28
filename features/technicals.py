import numpy as np
import pandas as pd


class TechnicalIndicators:
    """Compute technical indicators for feature engineering"""

    @staticmethod
    def add_moving_averages(df, short_window=10, long_window=50, ema_window=10):
        """Simple Moving Average (SMA) and Exponential Moving Average (EMA)"""
        df = df.copy()
        df["SMA_10"] = df["Close"].rolling(window=short_window).mean()
        df["SMA_50"] = df["Close"].rolling(window=long_window).mean()
        df["EMA_10"] = df["Close"].ewm(span=ema_window, adjust=False).mean()
        return df

    @staticmethod
    def add_rsi(df, period=14):
        """
        Relative Strength Index (RSI)
        Measures momentum: 0-30 = oversold, 70-100 = overbought
        """
        df = df.copy()
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def add_volatility(df, period=20):
        """Rolling standard deviation as volatility measure"""
        df = df.copy()
        df["Volatility"] = df["Close"].rolling(window=period).std()
        return df

    @staticmethod
    def add_macd(df, fast=12, slow=26, signal=9):
        """
        MACD (Moving Average Convergence Divergence)
        Helps identify trend changes and momentum
        """
        df = df.copy()
        exp1 = df["Close"].ewm(span=fast, adjust=False).mean()
        exp2 = df["Close"].ewm(span=slow, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
        df["MACD_Diff"] = df["MACD"] - df["MACD_Signal"]
        return df

    @staticmethod
    def add_bollinger_bands(df, period=20, num_std=2):
        """
        Bollinger Bands: Upper/Lower bands around moving average
        Helps identify overbought/oversold conditions
        """
        df = df.copy()
        df["BB_Middle"] = df["Close"].rolling(window=period).mean()
        bb_std = df["Close"].rolling(window=period).std()
        df["BB_Upper"] = df["BB_Middle"] + (num_std * bb_std)
        df["BB_Lower"] = df["BB_Middle"] - (num_std * bb_std)
        df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]
        return df

    @staticmethod
    def add_price_ratios(df):
        """Price-to-moving average ratios for normalization"""
        df = df.copy()
        df["Price_SMA_Ratio"] = df["Close"] / df["SMA_50"]
        df["Price_EMA_Ratio"] = df["Close"] / df["EMA_10"]
        return df

    @staticmethod
    def add_volume_indicators(df, period=20):
        """Volume-based indicators"""
        df = df.copy()
        
        # Volume Moving Average
        df["Volume_MA"] = df["Volume"].rolling(window=period).mean()
        
        # Volume Ratio
        df["Volume_Ratio"] = df["Volume"] / df["Volume_MA"]
        
        return df

    @staticmethod
    def add_atr(df, period=14):
        """Average True Range - volatility measure"""
        df = df.copy()
        high_low = df["High"] - df["Low"]
        high_close = abs(df["High"] - df["Close"].shift())
        low_close = abs(df["Low"] - df["Close"].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR"] = true_range.rolling(window=period).mean()
        return df

    @staticmethod
    def add_all_indicators(df, config):
        """
        Apply all technical indicators using config
        
        Args:
            df (pd.DataFrame): OHLCV data
            config (dict): Configuration from config.py
            
        Returns:
            pd.DataFrame: Data with all indicators
        """
        params = config["technical_indicators"]
        
        print("\n[TechnicalIndicators] Computing...")
        
        df = TechnicalIndicators.add_moving_averages(
            df,
            short_window=params["sma_short"],
            long_window=params["sma_long"],
            ema_window=params["ema_short"]
        )
        print("  ✓ Moving averages")
        
        df = TechnicalIndicators.add_rsi(df, period=params["rsi_period"])
        print("  ✓ RSI")
        
        df = TechnicalIndicators.add_volatility(df, period=params["volatility_period"])
        print("  ✓ Volatility")
        
        df = TechnicalIndicators.add_macd(
            df,
            fast=params["macd_fast"],
            slow=params["macd_slow"],
            signal=params["macd_signal"]
        )
        print("  ✓ MACD")
        
        df = TechnicalIndicators.add_bollinger_bands(
            df,
            period=params["bb_period"],
            num_std=params["bb_std"]
        )
        print("  ✓ Bollinger Bands")
        
        df = TechnicalIndicators.add_price_ratios(df)
        print("  ✓ Price ratios")
        
        df = TechnicalIndicators.add_volume_indicators(df)
        print("  ✓ Volume indicators")
        
        df = TechnicalIndicators.add_atr(df)
        print("  ✓ ATR")
        
        # Remove NAs from indicator computation
        df = df.dropna()
        print(f"  After NA removal: {len(df)} rows")
        
        return df


class FeatureNormalization:
    """Normalize/scale features for ML models"""

    @staticmethod
    def prepare_ml_features(df, feature_cols, target_col="Close", lag=5):
        """
        Prepare feature matrix with lagged target features
        
        Args:
            df (pd.DataFrame): Data with all features
            feature_cols (list): Technical indicator columns to use
            target_col (str): Target column name
            lag (int): Number of lags to create
            
        Returns:
            tuple: (X, y) ready for sklearn/XGBoost
        """
        df_ml = df.copy()
        
        # Create lagged close prices as additional features
        for i in range(1, lag + 1):
            df_ml[f"Close_Lag_{i}"] = df_ml[target_col].shift(i)
        
        df_ml = df_ml.dropna()
        
        all_features = feature_cols + [f"Close_Lag_{i}" for i in range(1, lag + 1)]
        X = df_ml[all_features]
        y = df_ml[target_col]
        
        print(f"\n[FeaturePreparation]")
        print(f"  Features: {len(all_features)} total")
        print(f"  Samples: {len(X)}")
        
        return X, y

    @staticmethod
    def scale_features(X_train, X_test, feature_cols=None):
        """
        MinMax scaling for ML models
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            feature_cols (list): Columns to scale (default: all)
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled, scaler)
        """
        from sklearn.preprocessing import MinMaxScaler
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        if feature_cols is None:
            feature_cols = X_train.columns.tolist()
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[feature_cols] = scaler.fit_transform(X_train[feature_cols])
        X_test_scaled[feature_cols] = scaler.transform(X_test[feature_cols])
        
        print(f"[Scaling] Applied MinMaxScaler to {len(feature_cols)} features")
        
        return X_train_scaled, X_test_scaled, scaler