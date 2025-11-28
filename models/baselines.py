import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class BaselineModels:
    """Simple baseline models for comparison"""

    @staticmethod
    def train_linear_regression(X_train, y_train, X_test):
        """
        Linear Regression baseline
        
        Args:
            X_train, y_train: Training data
            X_test: Test features
            
        Returns:
            tuple: (predictions, model)
        """
        print("\n[LinearRegression] Training baseline model...")
        
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        
        print(f"  R² score (train): {lr.score(X_train, y_train):.4f}")
        
        predictions = lr.predict(X_test)
        
        return predictions, lr

    @staticmethod
    def persistence_baseline(test_series):
        """
        Naive persistence: tomorrow's price = today's price
        Useful to see if more complex models beat this simple approach
        
        Args:
            test_series (pd.Series): Test close prices
            
        Returns:
            np.array: Predictions (shifted by 1)
        """
        print("\n[PersistenceBaseline] Today's price = tomorrow's prediction")
        
        persistence_pred = test_series.shift(1).dropna().values
        
        return persistence_pred

    @staticmethod
    def seasonal_naive(test_series, season_length=252):
        """
        Seasonal naive: price n days ago = today's prediction
        For stocks, 252 trading days ≈ 1 year
        
        Args:
            test_series (pd.Series): Test close prices
            season_length (int): Seasonal period
            
        Returns:
            np.array: Predictions
        """
        print(f"\n[SeasonalNaive] Prediction = price from {season_length} days ago")
        
        seasonal_pred = test_series.shift(season_length).dropna().values
        
        return seasonal_pred