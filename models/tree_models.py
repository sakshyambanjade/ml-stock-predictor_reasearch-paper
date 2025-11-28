import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


class RandomForestModel:
    """Random Forest for stock price prediction"""

    @staticmethod
    def hyperparameter_tuning(X_train, y_train, param_grid, cv_folds=3):
        """
        GridSearch for best hyperparameters
        
        Args:
            X_train, y_train: Training features and target
            param_grid (dict): Parameter grid to search
            cv_folds (int): Cross-validation folds
            
        Returns:
            sklearn.ensemble.RandomForestRegressor: Best model
        """
        print("\n[RandomForest] Hyperparameter tuning...")
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=cv_folds,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Best CV RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
        
        return grid_search.best_estimator_

    @staticmethod
    def train_random_forest(X_train, y_train, X_test, config):
        """
        Complete Random Forest pipeline
        
        Args:
            X_train, y_train: Training data
            X_test: Test features
            config (dict): Random Forest config
            
        Returns:
            np.array: Predictions
        """
        best_model = RandomForestModel.hyperparameter_tuning(
            X_train,
            y_train,
            config["param_grid"],
            config.get("cv_folds", 3)
        )
        
        print("\n[RandomForest] Making predictions...")
        predictions = best_model.predict(X_test)
        
        return predictions, best_model


class XGBoostModel:
    """XGBoost for stock price prediction"""

    @staticmethod
    def hyperparameter_tuning(X_train, y_train, param_grid, cv_folds=3):
        """
        GridSearch for best hyperparameters
        
        Args:
            X_train, y_train: Training features and target
            param_grid (dict): Parameter grid to search
            cv_folds (int): Cross-validation folds
            
        Returns:
            xgboost.XGBRegressor: Best model
        """
        print("\n[XGBoost] Hyperparameter tuning...")
        
        xgb = XGBRegressor(
            random_state=42,
            objective="reg:squarederror",
            n_jobs=-1,
            verbosity=0
        )
        
        grid_search = GridSearchCV(
            xgb,
            param_grid,
            cv=cv_folds,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Best CV RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
        
        return grid_search.best_estimator_

    @staticmethod
    def train_xgboost(X_train, y_train, X_test, config):
        """
        Complete XGBoost pipeline
        
        Args:
            X_train, y_train: Training data
            X_test: Test features
            config (dict): XGBoost config
            
        Returns:
            tuple: (predictions, best_model)
        """
        best_model = XGBoostModel.hyperparameter_tuning(
            X_train,
            y_train,
            config["param_grid"],
            config.get("cv_folds", 3)
        )
        
        print("\n[XGBoost] Making predictions...")
        predictions = best_model.predict(X_test)
        
        return predictions, best_model

    @staticmethod
    def get_feature_importance(model, feature_names, top_n=10):
        """Get feature importance scores"""
        importance = model.feature_importances_
        indices = np.argsort(importance)[-top_n:]
        
        print(f"\n[XGBoost] Top {top_n} Important Features:")
        for i, idx in enumerate(reversed(indices)):
            print(f"  {i+1}. {feature_names[idx]}: {importance[idx]:.4f}")
        
        return importance