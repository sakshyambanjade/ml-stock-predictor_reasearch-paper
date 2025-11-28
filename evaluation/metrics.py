import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


class ModelEvaluation:
    """Compute evaluation metrics"""

    @staticmethod
    def calculate_metrics(y_true, y_pred, model_name):
        """
        Calculate RMSE, MAE, MAPE
        
        Args:
            y_true, y_pred: Actual vs predicted values
            model_name (str): Model name for printing
            
        Returns:
            dict: Metrics dictionary
        """
        # Align lengths
        min_len = min(len(y_true), len(y_pred))
        y_true = np.array(y_true[:min_len]).flatten()
        y_pred = np.array(y_pred[:min_len]).flatten()
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        
        print(f"\n[{model_name}]")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  MAPE: {mape:.4f}")
        
        return {
            "Model": model_name,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape
        }

    @staticmethod
    def directional_accuracy(y_true, y_pred):
        """
        % of times model correctly predicted price direction (up/down)
        
        Args:
            y_true, y_pred: Actual vs predicted
            
        Returns:
            float: Directional accuracy (0-1)
        """
        true_diff = np.diff(y_true)
        pred_diff = np.diff(y_pred)
        
        correct = np.sum((true_diff > 0) == (pred_diff > 0))
        accuracy = correct / len(true_diff)
        
        return accuracy

    @staticmethod
    def evaluate_by_regime(test_df, y_pred, model_name, regime_col="Market_Regime"):
        """
        Evaluate model performance per market regime
        
        Args:
            test_df (pd.DataFrame): Test data with regime labels
            y_pred (np.array): Predictions
            model_name (str): Model name
            regime_col (str): Regime column name
            
        Returns:
            list: Results per regime
        """
        print(f"\n[{model_name}] Performance by {regime_col}")
        
        regime_results = []
        
        for regime in test_df[regime_col].unique():
            regime_mask = test_df[regime_col] == regime
            indices = np.where(regime_mask.values)[0]
            
            if len(indices) > 0:
                y_true_regime = test_df["Close"].iloc[indices].values
                
                # Match prediction indices
                y_pred_regime = y_pred[indices[:len(y_pred)]]
                
                if len(y_pred_regime) > 0:
                    rmse = np.sqrt(mean_squared_error(
                        y_true_regime[:len(y_pred_regime)],
                        y_pred_regime
                    ))
                    mae = mean_absolute_error(
                        y_true_regime[:len(y_pred_regime)],
                        y_pred_regime
                    )
                    
                    regime_results.append({
                        "Model": model_name,
                        "Regime": regime,
                        "Samples": len(y_pred_regime),
                        "RMSE": rmse,
                        "MAE": mae
                    })
                    
                    print(f"  {regime}: RMSE={rmse:.4f}, MAE={mae:.4f}, n={len(y_pred_regime)}")
        
        return regime_results


class TimeSeriesCrossValidation:
    """Time-series specific cross-validation"""

    @staticmethod
    def walk_forward_validation(X, y, model_class, initial_window=100, step=50):
        """
        Walk-forward (rolling) cross-validation for time-series
        
        Args:
            X, y: Features and target
            model_class: Model class with fit/predict
            initial_window (int): Initial training window
            step (int): Step size for rolling
            
        Returns:
            dict: CV results
        """
        predictions = []
        actuals = []
        
        for i in range(initial_window, len(X), step):
            # Split
            X_train, y_train = X[:i], y[:i]
            X_test_start = i
            X_test_end = min(i + step, len(X))
            X_test, y_test = X[X_test_start:X_test_end], y[X_test_start:X_test_end]
            
            # Train and predict
            model = model_class()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            
            predictions.extend(pred)
            actuals.extend(y_test)
        
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        
        return {
            "rmse": rmse,
            "predictions": np.array(predictions),
            "actuals": np.array(actuals)
        }