# =====================
# DATA SETTINGS
# =====================
DATA_CONFIG = {
    "ticker": "AAPL",
    "start_date": "2015-01-01",
    "end_date": "2024-12-31",
    "train_ratio": 0.8,
}

# =====================
# FEATURE ENGINEERING
# =====================
FEATURE_CONFIG = {
    "technical_indicators": {
        "sma_short": 10,
        "sma_long": 50,
        "ema_short": 10,
        "rsi_period": 14,
        "volatility_period": 20,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bb_period": 20,
        "bb_std": 2,
    },
    "lag_features": 5,
    "feature_cols": [
        "SMA_10", "SMA_50", "EMA_10", "RSI",
        "Volatility", "MACD", "MACD_Signal",
        "Volume", "Price_SMA_Ratio"
    ]
}

# =====================
# MARKET REGIME
# =====================
REGIME_CONFIG = {
    "rolling_window": 60,  # for rolling returns
    "volatility_split": "median",  # or percentile value
}

# =====================
# MODEL HYPERPARAMETERS
# =====================
MODEL_CONFIG = {
    "arima": {
        "enabled": True,
        "max_p": 5,
        "max_d": 2,
        "max_q": 5,
    },
    "prophet": {
        "enabled": True,
        "daily_seasonality": True,
        "yearly_seasonality": True,
    },
    "random_forest": {
        "enabled": True,
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5]
        },
        "cv_folds": 3,
    },
    "xgboost": {
        "enabled": True,
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.1, 0.2]
        },
        "cv_folds": 3,
    },
    "svr": {
        "enabled": True,
        "param_grid": {
            "kernel": ["rbf", "poly"],
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"]
        },
        "cv_folds": 3,
    },
    "linear_regression": {
        "enabled": True,
    },
    "lstm": {
        "enabled": True,
        "seq_length": 60,
        "batch_size": 32,
        "epochs": 20,
        "validation_split": 0.1,
        "architecture": {
            "layers": [
                {"type": "LSTM", "units": 50, "return_sequences": True, "dropout": 0.2},
                {"type": "LSTM", "units": 50, "return_sequences": False, "dropout": 0.2},
                {"type": "Dense", "units": 25},
                {"type": "Dense", "units": 1},
            ]
        }
    },
    "gru": {
        "enabled": True,
        "seq_length": 60,
        "batch_size": 32,
        "epochs": 20,
        "validation_split": 0.1,
    },
    "cnn": {
        "enabled": True,
        "seq_length": 60,
        "batch_size": 32,
        "epochs": 20,
        "validation_split": 0.1,
    }
}

# =====================
# EVALUATION
# =====================
EVAL_CONFIG = {
    "metrics": ["RMSE", "MAE", "MAPE"],
    "cv_type": "time_series",
    "cv_folds": 5,
    "eval_by_regime": True,  # Evaluate per market regime
}

# =====================
# OUTPUT
# =====================
OUTPUT_CONFIG = {
    "save_results": True,
    "results_dir": "./results",
    "models_dir": "./models",
    "data_dir": "./data",
    "export_csv": True,
    "export_plots": True,
}