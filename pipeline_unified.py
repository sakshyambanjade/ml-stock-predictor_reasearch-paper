import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, confusion_matrix
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

class MultiStockPipeline:
    def __init__(self, tickers=['AAPL', 'MSFT', 'GOOGL', 'TSLA'], ridge_alpha=10):
        self.tickers = tickers
        self.ridge_alpha = ridge_alpha
        self.results = {}
        self.regime_results = {}
        
    def generate_synthetic_data(self, ticker, initial_price=100, drift=0.001, vol=0.02):
        """Generate synthetic stock data"""
        np.random.seed(42 + hash(ticker) % 100)
        start_date = datetime(2015, 1, 1)
        dates = [start_date + timedelta(days=x) for x in range(0, 2500, 1) 
                 if (start_date + timedelta(days=x)).weekday() < 5]
        
        prices = []
        open_p = initial_price
        
        for i in range(len(dates)):
            daily_change = np.random.normal(drift, vol)
            close_p = open_p * (1 + daily_change)
            high_p = max(open_p, close_p) * (1 + abs(np.random.normal(0, 0.01)))
            low_p = min(open_p, close_p) * (1 - abs(np.random.normal(0, 0.01)))
            volume = np.random.randint(50000000, 100000000)
            
            prices.append({
                'ticker': ticker,
                'date': dates[i],
                'Open': open_p,
                'High': high_p,
                'Low': low_p,
                'Close': close_p,
                'Volume': volume,
            })
            open_p = close_p
        
        return pd.DataFrame(prices)
    
    def compute_features(self, df):
        """Compute all technical features"""
        df = df.copy()
        
        # Returns
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["return"] = df["Close"].pct_change()
        
        # Moving Averages
        df["SMA_10"] = df["Close"].rolling(window=10).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
        
        # RSI
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        
        # Volatility
        df["Volatility"] = df["Close"].rolling(window=20).std()
        
        # MACD
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        
        # Volume
        df["Volume_MA"] = df["Volume"].rolling(window=20).mean()
        df["Volume_Ratio"] = df["Volume"] / df["Volume_MA"]
        df["Price_SMA_Ratio"] = df["Close"] / df["SMA_50"]
        
        # Lag features
        for i in range(1, 6):
            df[f"Close_Lag_{i}"] = df["Close"].shift(i)
        
        # Market regimes
        df["rolling_return"] = df["Close"].pct_change(60)
        df["Market_Regime"] = np.where(df["rolling_return"] > 0, "Bull", "Bear")
        
        vol_median = df["Volatility"].median()
        df["Volatility_Regime"] = np.where(df["Volatility"] > vol_median, "High_Vol", "Low_Vol")
        df["Combined_Regime"] = df["Market_Regime"] + "_" + df["Volatility_Regime"]
        
        # True regime for validation
        df["Price_60d_MA"] = df["Close"].rolling(window=60).mean()
        df["Market_Regime_True"] = np.where(df["Close"] > df["Price_60d_MA"], "Bull", "Bear")
        df["Volatility_Regime_True"] = np.where(df["Volatility"] > vol_median, "High_Vol", "Low_Vol")
        
        return df.dropna()
    
    def train_predict_ridge(self, df):
        """Train Ridge model and make predictions"""
        feature_cols = [
            "SMA_10", "SMA_50", "EMA_10", "RSI",
            "Volatility", "MACD", "MACD_Signal",
            "Volume_Ratio", "Price_SMA_Ratio",
            "Close_Lag_1", "Close_Lag_2", "Close_Lag_3", "Close_Lag_4", "Close_Lag_5"
        ]
        
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        X_train = train_df[feature_cols]
        y_train = train_df["Close"]
        X_test = test_df[feature_cols]
        y_test = test_df["Close"]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        ridge = Ridge(alpha=self.ridge_alpha)
        ridge.fit(X_train_scaled, y_train)
        ridge_pred = ridge.predict(X_test_scaled)
        
        # Metrics
        ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
        ridge_mae = mean_absolute_error(y_test, ridge_pred)
        ridge_mape = np.mean(np.abs((y_test - ridge_pred) / y_test)) * 100
        
        return {
            'model': ridge,
            'scaler': scaler,
            'rmse': ridge_rmse,
            'mae': ridge_mae,
            'mape': ridge_mape,
            'predictions': ridge_pred,
            'y_test': y_test.values,
            'test_df': test_df
        }
    
    def train_predict_naive(self, df):
        """Naive baseline: predict today = yesterday"""
        split_idx = int(len(df) * 0.8)
        test_df = df.iloc[split_idx:].copy()
        
        y_test = test_df["Close"].values
        naive_pred = test_df["Close"].shift(1).dropna().values
        y_test_aligned = y_test[1:]
        
        naive_rmse = np.sqrt(mean_squared_error(y_test_aligned, naive_pred))
        naive_mae = mean_absolute_error(y_test_aligned, naive_pred)
        naive_mape = np.mean(np.abs((y_test_aligned - naive_pred) / y_test_aligned)) * 100
        
        return {
            'rmse': naive_rmse,
            'mae': naive_mae,
            'mape': naive_mape,
            'predictions': naive_pred
        }
    
    def evaluate_regime_classification(self, test_df):
        """Evaluate regime classification accuracy"""
        market_accuracy = accuracy_score(test_df["Market_Regime_True"], test_df["Market_Regime"])
        market_f1 = f1_score(test_df["Market_Regime_True"], test_df["Market_Regime"], 
                            pos_label="Bull", zero_division=0)
        
        vol_accuracy = accuracy_score(test_df["Volatility_Regime_True"], test_df["Volatility_Regime"])
        vol_f1 = f1_score(test_df["Volatility_Regime_True"], test_df["Volatility_Regime"], 
                         pos_label="High_Vol", zero_division=0)
        
        bull_pct = (test_df['Market_Regime_True'] == 'Bull').sum() / len(test_df) * 100
        high_vol_pct = (test_df['Volatility_Regime_True'] == 'High_Vol').sum() / len(test_df) * 100
        
        return {
            'market_accuracy': market_accuracy,
            'market_f1': market_f1,
            'vol_accuracy': vol_accuracy,
            'vol_f1': vol_f1,
            'bull_pct': bull_pct,
            'high_vol_pct': high_vol_pct
        }
    
    def run(self):
        """Execute full pipeline for all stocks"""
        print("="*80)
        print("MULTI-STOCK ML PREDICTION PIPELINE")
        print("="*80)
        
        # Stock configs
        configs = {
            'AAPL': {'initial_price': 100, 'drift': 0.0008, 'vol': 0.020},
            'MSFT': {'initial_price': 150, 'drift': 0.0010, 'vol': 0.018},
            'GOOGL': {'initial_price': 800, 'drift': 0.0012, 'vol': 0.022},
            'TSLA': {'initial_price': 200, 'drift': 0.0015, 'vol': 0.035}
        }
        
        results_ridge = []
        results_naive = []
        regime_data = []
        
        for ticker in self.tickers:
            print(f"\n{'='*80}")
            print(f"Processing: {ticker}")
            print(f"{'='*80}")
            
            # Generate data
            config = configs.get(ticker, {'initial_price': 100, 'drift': 0.001, 'vol': 0.02})
            df = self.generate_synthetic_data(ticker, **config)
            
            # Features
            df = self.compute_features(df)
            
            # Ridge model
            ridge_results = self.train_predict_ridge(df)
            print(f"\n[Ridge] RMSE: {ridge_results['rmse']:.4f}, MAE: {ridge_results['mae']:.4f}, MAPE: {ridge_results['mape']:.2f}%")
            
            results_ridge.append({
                'Stock': ticker,
                'Model': 'Ridge',
                'RMSE': ridge_results['rmse'],
                'MAE': ridge_results['mae'],
                'MAPE': ridge_results['mape']
            })
            
            # Naive baseline
            naive_results = self.train_predict_naive(df)
            print(f"[Naive] RMSE: {naive_results['rmse']:.4f}, MAE: {naive_results['mae']:.4f}, MAPE: {naive_results['mape']:.2f}%")
            
            improvement = (naive_results['rmse'] - ridge_results['rmse']) / naive_results['rmse'] * 100
            print(f"[Improvement] Ridge beats Naive by {improvement:+.1f}%")
            
            results_naive.append({
                'Stock': ticker,
                'Model': 'Naive',
                'RMSE': naive_results['rmse'],
                'MAE': naive_results['mae'],
                'MAPE': naive_results['mape']
            })
            
            # Regime classification
            regime_stats = self.evaluate_regime_classification(ridge_results['test_df'])
            print(f"\n[Regime Classification]")
            print(f"  Market Regime Accuracy: {regime_stats['market_accuracy']:.1%} (F1: {regime_stats['market_f1']:.3f})")
            print(f"  Volatility Regime Accuracy: {regime_stats['vol_accuracy']:.1%} (F1: {regime_stats['vol_f1']:.3f})")
            print(f"  Bull: {regime_stats['bull_pct']:.1f}%, High_Vol: {regime_stats['high_vol_pct']:.1f}%")
            
            regime_data.append({
                'Stock': ticker,
                'Market_Accuracy': regime_stats['market_accuracy'],
                'Market_F1': regime_stats['market_f1'],
                'Vol_Accuracy': regime_stats['vol_accuracy'],
                'Vol_F1': regime_stats['vol_f1'],
                'Bull_Pct': regime_stats['bull_pct'],
                'High_Vol_Pct': regime_stats['high_vol_pct']
            })
            
            self.results[ticker] = ridge_results
            self.regime_results[ticker] = regime_stats
        
        # Summary tables
        print("\n" + "="*80)
        print("FINAL RESULTS SUMMARY")
        print("="*80)
        
        df_ridge = pd.DataFrame(results_ridge)
        df_naive = pd.DataFrame(results_naive)
        df_comparison = pd.concat([df_ridge, df_naive], ignore_index=True)
        
        print("\n📊 MODEL COMPARISON (Ridge vs Naive):")
        print(df_comparison.to_string(index=False))
        
        print("\n📊 REGIME CLASSIFICATION ACCURACY:")
        df_regime = pd.DataFrame(regime_data)
        print(df_regime.to_string(index=False))
        
        print("\n📊 RIDGE IMPROVEMENT OVER NAIVE:")
        for ticker in self.tickers:
            ridge_rmse = df_ridge[df_ridge['Stock'] == ticker]['RMSE'].values[0]
            naive_rmse = df_naive[df_naive['Stock'] == ticker]['RMSE'].values[0]
            improvement = (naive_rmse - ridge_rmse) / naive_rmse * 100
            print(f"   {ticker}: {improvement:+.1f}% improvement")
        
        return {
            'ridge_results': df_ridge,
            'naive_results': df_naive,
            'regime_results': df_regime,
            'comparison': df_comparison
        }


if __name__ == "__main__":
    pipeline = MultiStockPipeline(tickers=['AAPL', 'MSFT', 'GOOGL', 'TSLA'], ridge_alpha=10)
    results = pipeline.run()
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE")
    print("="*80)
    print("\nResults available in:")
    print("  - results['ridge_results']")
    print("  - results['naive_results']")
    print("  - results['regime_results']")
    print("  - results['comparison']")