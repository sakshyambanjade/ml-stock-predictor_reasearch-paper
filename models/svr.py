import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


class SVRModel:
    """Support Vector Regression for stock price prediction"""

    @staticmethod
    def scale_data(X_train, y_train, X_test):
        """
        SVR requires feature scaling
        
        Args:
            X_train, y_train, X_test: Training and test data
            
        Returns:
            tuple: (X_train_scaled, y_train_scaled, X_test_scaled, scaler_X, scaler_y)
        """
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        
        print("\n[SVR] Features scaled with StandardScaler")
        
        return X_train_scaled, y_train_scaled, X_test_scaled, scaler_X, scaler_y

    @staticmethod
    def hyperparameter_tuning(X_train_scaled, y_train_scaled, param_grid, cv_folds=3):
        """
        GridSearch for best kernel and parameters
        
        Args:
            X_train_scaled, y_train_scaled: Scaled training data
            param_grid (dict): Parameter grid to search
            cv_folds (int): Cross-validation folds
            
        Returns:
            sklearn.svm.SVR: Best model
        """
        print("\n[SVR] Hyperparameter tuning...")
        
        svr = SVR()
        grid_search = GridSearchCV(
            svr,
            param_grid,
            cv=cv_folds,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train_scaled, y_train_scaled)
        
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Best CV RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
        
        return grid_search.best_estimator_

    @staticmethod
    def train_svr(X_train, y_train, X_test, config):
        """
        Complete SVR pipeline
        
        Args:
            X_train, y_train: Training data
            X_test: Test features
            config (dict): SVR config
            
        Returns:
            tuple: (predictions, best_model, scaler_y)
        """
        # Scale data
        X_train_scaled, y_train_scaled, X_test_scaled, scaler_X, scaler_y = SVRModel.scale_data(
            X_train, y_train, X_test
        )
        
        # Tune and train
        best_model = SVRModel.hyperparameter_tuning(
            X_train_scaled,
            y_train_scaled,
            config["param_grid"],
            config.get("cv_folds", 3)
        )
        
        # Predict
        print("\n[SVR] Making predictions...")
        predictions_scaled = best_model.predict(X_test_scaled)
        
        # Inverse transform
        predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
        
        return predictions, best_model