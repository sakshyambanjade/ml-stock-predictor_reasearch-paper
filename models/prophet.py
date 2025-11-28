import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from prophet import Prophet


class ProphetModel:
    """Facebook Prophet for time-series with seasonality"""

    @staticmethod
    def prepare_data(train_df, test_df):
        """
        Prepare data in Prophet format: 'ds' (date) and 'y' (target)
        
        Args:
            train_df, test_df (pd.DataFrame): Must have 'date' and 'Close' columns
            
        Returns:
            tuple: (train_prophet, test_prophet)
        """
        train_prophet = train_df[["date", "Close"]].rename(
            columns={"date": "ds", "Close": "y"}
        ).copy()
        
        test_prophet = test_df[["date", "Close"]].rename(
            columns={"date": "ds", "Close": "y"}
        ).copy()
        
        print(f"\n[Prophet] Data prepared")
        print(f"  Train: {len(train_prophet)} rows")
        print(f"  Test: {len(test_prophet)} rows")
        
        return train_prophet, test_prophet

    @staticmethod
    def fit_prophet(train_data, config):
        """
        Fit Prophet model
        
        Args:
            train_data (pd.DataFrame): Training data with 'ds' and 'y'
            config (dict): Prophet config from config.py
            
        Returns:
            Prophet: Fitted model
        """
        print("\n[Prophet] Fitting model...")
        
        model = Prophet(
            daily_seasonality=config.get("daily_seasonality", True),
            yearly_seasonality=config.get("yearly_seasonality", True),
            interval_width=0.95
        )
        
        model.fit(train_data)
        
        return model

    @staticmethod
    def add_regressors(model, train_data, feature_cols):
        """
        Add external regressors (optional)
        
        Args:
            model (Prophet): Prophet model
            train_data (pd.DataFrame): Training data with features
            feature_cols (list): Additional feature columns
            
        Returns:
            Prophet: Model with regressors
        """
        for col in feature_cols:
            if col in train_data.columns:
                model.add_regressor(col)
        
        return model

    @staticmethod
    def predict_prophet(model, test_data):
        """
        Make predictions
        
        Args:
            model (Prophet): Fitted model
            test_data (pd.DataFrame): Test data with 'ds' column
            
        Returns:
            np.array: Predictions (yhat)
        """
        print(f"\n[Prophet] Forecasting {len(test_data)} steps...")
        
        future = test_data[["ds"]].copy()
        forecast = model.predict(future)
        
        # Extract point predictions
        predictions = forecast["yhat"].values
        
        return predictions

    @staticmethod
    def train_and_predict(train_df, test_df, config):
        """
        Complete Prophet pipeline
        
        Args:
            train_df, test_df (pd.DataFrame): Data with 'date' and 'Close'
            config (dict): Prophet config
            
        Returns:
            np.array: Predictions
        """
        train_prophet, test_prophet = ProphetModel.prepare_data(train_df, test_df)
        model = ProphetModel.fit_prophet(train_prophet, config)
        predictions = ProphetModel.predict_prophet(model, test_prophet)
        
        return predictions