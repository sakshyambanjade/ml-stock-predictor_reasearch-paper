import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel:
    """ARIMA for time-series forecasting"""

    @staticmethod
    def auto_tune_arima(train_series, max_p=5, max_d=2, max_q=5):
        """
        Auto-tune ARIMA parameters (p,d,q) using pmdarima
        
        Args:
            train_series (pd.Series): Training close prices
            max_p, max_d, max_q: Maximum values to search
            
        Returns:
            tuple: Best (p, d, q) order
        """
        print("\n[ARIMA] Auto-tuning parameters (p,d,q)...")
        
        auto_model = pm.auto_arima(
            train_series,
            start_p=0,
            start_q=0,
            max_p=max_p,
            max_d=max_d,
            max_q=max_q,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False
        )
        
        best_order = auto_model.order
        print(f"  Best order (p,d,q): {best_order}")
        print(f"  AIC: {auto_model.aic():.2f}")
        print(f"  BIC: {auto_model.bic():.2f}")
        
        return best_order

    @staticmethod
    def fit_arima(train_series, order):
        """
        Fit ARIMA model with given order
        
        Args:
            train_series (pd.Series): Training data
            order (tuple): (p, d, q)
            
        Returns:
            model: Fitted ARIMA model
        """
        print(f"\n[ARIMA] Fitting with order {order}...")
        
        model = ARIMA(train_series, order=order)
        fitted = model.fit()
        
        print(f"  AIC: {fitted.aic:.2f}")
        print(f"  BIC: {fitted.bic:.2f}")
        
        return fitted

    @staticmethod
    def predict_arima(model, test_series):
        """
        Make predictions
        
        Args:
            model: Fitted ARIMA model
            test_series (pd.Series): Test data (for length reference)
            
        Returns:
            np.array: Predictions
        """
        print(f"\n[ARIMA] Forecasting {len(test_series)} steps...")
        
        predictions = model.forecast(steps=len(test_series))
        
        return np.array(predictions)

    @staticmethod
    def train_and_predict(train_series, test_series, config):
        """
        Complete ARIMA pipeline
        
        Args:
            train_series (pd.Series): Training data
            test_series (pd.Series): Test data
            config (dict): ARIMA config from config.py
            
        Returns:
            np.array: Predictions
        """
        order = ARIMAModel.auto_tune_arima(
            train_series,
            max_p=config.get("max_p", 5),
            max_d=config.get("max_d", 2),
            max_q=config.get("max_q", 5)
        )
        
        model = ARIMAModel.fit_arima(train_series, order)
        predictions = ARIMAModel.predict_arima(model, test_series)
        
        return predictions