import warnings
warnings.filterwarnings("ignore")

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Import config
from config import (
    DATA_CONFIG, FEATURE_CONFIG, REGIME_CONFIG,
    MODEL_CONFIG, EVAL_CONFIG, OUTPUT_CONFIG
)

# Import modules
from data.loader import DataLoader, DataPreprocessor
from features.technicals import TechnicalIndicators, FeatureNormalization
from features.regimes import MarketRegimeAnalyzer

from models.arima import ARIMAModel
from models.prophet import ProphetModel
from models.tree_models import RandomForestModel, XGBoostModel
from models.svr import SVRModel
from models.baselines import BaselineModels
from models.neural_nets import LSTMModel, GRUModel, CNNModel

from evaluation.metrics import ModelEvaluation


def create_output_dirs():
    """Create output directories"""
    for dir_path in [
        OUTPUT_CONFIG["results_dir"],
        OUTPUT_CONFIG["models_dir"],
        OUTPUT_CONFIG["data_dir"]
    ]:
        Path(dir_path).mkdir(exist_ok=True)


def main():
    print("=" * 80)
    print("ML-BASED STOCK PREDICTION PIPELINE")
    print("=" * 80)

    create_output_dirs()

    # ========================================================
    # STEP 1: DATA COLLECTION & PREPROCESSING
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 1: DATA COLLECTION & PREPROCESSING")
    print("=" * 80)

    loader = DataLoader(data_dir=OUTPUT_CONFIG["data_dir"])

    df = loader.fetch_stock_data(
        ticker=DATA_CONFIG["ticker"],
        start_date=DATA_CONFIG["start_date"],
        end_date=DATA_CONFIG["end_date"],
        use_cache=True,
        fallback_csv="AAPL_test_data.csv"
    )

    loader.validate_data(df)

    # Preprocess
    df = DataPreprocessor.preprocess(df)

    # ✅ Safe stationarity check (FIXED)
    if len(df) > 0:
        is_stationary = DataPreprocessor.test_stationarity(
            df["Close"],
            name=f"{DATA_CONFIG['ticker']} Close Price"
        )
    else:
        print("[WARNING] Data is empty - skipping stationarity test")
        is_stationary = False

    # ========================================================
    # STEP 2: FEATURE ENGINEERING
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 80)

    df = TechnicalIndicators.add_all_indicators(df, FEATURE_CONFIG)

    # ========================================================
    # STEP 3: MARKET REGIME DEFINITION
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 3: MARKET REGIME DEFINITION")
    print("=" * 80)

    df = MarketRegimeAnalyzer.add_all_regimes(df, REGIME_CONFIG)
    MarketRegimeAnalyzer.regime_summary(df)

    # ========================================================
    # STEP 4: TRAIN/TEST SPLIT
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 4: TRAIN/TEST SPLIT")
    print("=" * 80)

    train_df, test_df = DataPreprocessor.time_series_split(
        df,
        train_ratio=DATA_CONFIG["train_ratio"]
    )

    # Prepare ML features
    feature_cols = FEATURE_CONFIG["feature_cols"]

    X_train, y_train = FeatureNormalization.prepare_ml_features(
        train_df,
        feature_cols,
        target_col="Close",
        lag=FEATURE_CONFIG["lag_features"]
    )

    X_test, y_test = FeatureNormalization.prepare_ml_features(
        test_df,
        feature_cols,
        target_col="Close",
        lag=FEATURE_CONFIG["lag_features"]
    )

    # ========================================================
    # STEP 5: TRAIN MODELS
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 5: TRAINING MODELS")
    print("=" * 80)

    results = []

    # --- ARIMA ---
    if MODEL_CONFIG["arima"]["enabled"]:
        try:
            print("\n[1/9] ARIMA...")
            arima_pred = ARIMAModel.train_and_predict(
                train_df["Close"],
                test_df["Close"],
                MODEL_CONFIG["arima"]
            )
            results.append(ModelEvaluation.calculate_metrics(
                test_df["Close"].values,
                arima_pred,
                "ARIMA"
            ))
        except Exception as e:
            print(f"  ERROR: {e}")

    # --- Prophet ---
    if MODEL_CONFIG["prophet"]["enabled"]:
        try:
            print("\n[2/9] Prophet...")
            prophet_pred = ProphetModel.train_and_predict(
                train_df,
                test_df,
                MODEL_CONFIG["prophet"]
            )
            results.append(ModelEvaluation.calculate_metrics(
                test_df["Close"].values,
                prophet_pred,
                "Prophet"
            ))
        except Exception as e:
            print(f"  ERROR: {e}")

    # --- Random Forest ---
    if MODEL_CONFIG["random_forest"]["enabled"]:
        try:
            print("\n[3/9] Random Forest...")
            rf_pred, _ = RandomForestModel.train_random_forest(
                X_train, y_train, X_test,
                MODEL_CONFIG["random_forest"]
            )
            results.append(ModelEvaluation.calculate_metrics(
                y_test.values,
                rf_pred,
                "Random Forest"
            ))
        except Exception as e:
            print(f"  ERROR: {e}")

    # --- XGBoost ---
    if MODEL_CONFIG["xgboost"]["enabled"]:
        try:
            print("\n[4/9] XGBoost...")
            xgb_pred, _ = XGBoostModel.train_xgboost(
                X_train, y_train, X_test,
                MODEL_CONFIG["xgboost"]
            )
            results.append(ModelEvaluation.calculate_metrics(
                y_test.values,
                xgb_pred,
                "XGBoost"
            ))
        except Exception as e:
            print(f"  ERROR: {e}")

    # --- SVR ---
    if MODEL_CONFIG["svr"]["enabled"]:
        try:
            print("\n[5/9] SVR...")
            svr_pred, _ = SVRModel.train_svr(
                X_train, y_train, X_test,
                MODEL_CONFIG["svr"]
            )
            results.append(ModelEvaluation.calculate_metrics(
                y_test.values,
                svr_pred,
                "SVR"
            ))
        except Exception as e:
            print(f"  ERROR: {e}")

    # --- Linear Regression (Baseline) ---
    if MODEL_CONFIG["linear_regression"]["enabled"]:
        try:
            print("\n[6/9] Linear Regression (Baseline)...")
            lr_pred, _ = BaselineModels.train_linear_regression(
                X_train, y_train, X_test
            )
            results.append(ModelEvaluation.calculate_metrics(
                y_test.values,
                lr_pred,
                "Linear Regression"
            ))
        except Exception as e:
            print(f"  ERROR: {e}")

    # --- LSTM ---
    if MODEL_CONFIG["lstm"]["enabled"]:
        try:
            print("\n[7/9] LSTM...")
            lstm_pred = LSTMModel.train_lstm_complete(
                train_df, test_df,
                MODEL_CONFIG["lstm"]
            )
            results.append(ModelEvaluation.calculate_metrics(
                test_df["Close"].values[-len(lstm_pred):],
                lstm_pred,
                "LSTM"
            ))
        except Exception as e:
            print(f"  ERROR: {e}")

    # --- GRU ---
    if MODEL_CONFIG["gru"]["enabled"]:
        try:
            print("\n[8/9] GRU...")
            gru_pred = GRUModel.train_gru_complete(
                train_df, test_df,
                MODEL_CONFIG["gru"]
            )
            results.append(ModelEvaluation.calculate_metrics(
                test_df["Close"].values[-len(gru_pred):],
                gru_pred,
                "GRU"
            ))
        except Exception as e:
            print(f"  ERROR: {e}")

    # --- CNN ---
    if MODEL_CONFIG["cnn"]["enabled"]:
        try:
            print("\n[9/9] CNN...")
            cnn_pred = CNNModel.train_cnn_complete(
                train_df, test_df,
                MODEL_CONFIG["cnn"]
            )
            results.append(ModelEvaluation.calculate_metrics(
                test_df["Close"].values[-len(cnn_pred):],
                cnn_pred,
                "CNN"
            ))
        except Exception as e:
            print(f"  ERROR: {e}")

    # ========================================================
    # STEP 6: RESULTS & EVALUATION
    # ========================================================
    print("\n" + "=" * 80)
    print("STEP 6: RESULTS & EVALUATION")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("RMSE")

    print("\n" + "=" * 80)
    print("FINAL RESULTS - RANKED BY RMSE")
    print("=" * 80)
    print(results_df.to_string(index=False))

    if OUTPUT_CONFIG["export_csv"]:
        results_file = Path(OUTPUT_CONFIG["results_dir"]) / f"{DATA_CONFIG['ticker']}_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\n✓ Results saved to: {results_file}")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
