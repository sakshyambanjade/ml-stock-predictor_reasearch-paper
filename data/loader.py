import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path


class DataLoader:
    """Handles downloading and caching stock data"""

    def __init__(self, data_dir="./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def fetch_stock_data(self, ticker, start_date, end_date, use_cache=False, fallback_csv=None):
        """
        Fetch historical stock data from Yahoo Finance
        Falls back to local CSV if download fails
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            use_cache (bool): Load from CSV if exists
            fallback_csv (str): Path to fallback CSV file if download fails
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        cache_file = self.data_dir / f"{ticker}_{start_date}_{end_date}.csv"
        
        # Try cache first
        if use_cache and cache_file.exists():
            print(f"[DataLoader] Loading from cache: {cache_file}")
            df = pd.read_csv(cache_file)
            return df

        # Try download
        print(f"[DataLoader] Downloading {ticker} ({start_date} to {end_date})")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            # Check if download actually worked (not empty)
            if df is None or df.empty or len(df) == 0:
                raise ValueError("Downloaded data is empty")
            
            df.reset_index(inplace=True)
            if 'Date' in df.columns:
                df.rename(columns={"Date": "date"}, inplace=True)
            
            # Save to cache
            df.to_csv(cache_file, index=False)
            print(f"[DataLoader] Saved to {cache_file}")
            return df
            
        except Exception as e:
            print(f"[DataLoader] Download failed: {str(e)}")
            
            # Try fallback CSV
            if fallback_csv and Path(fallback_csv).exists():
                print(f"[DataLoader] Using fallback CSV: {fallback_csv}")
                df = pd.read_csv(fallback_csv)
                # Standardize column names
                if 'Date' in df.columns:
                    df.rename(columns={"Date": "date"}, inplace=True)
                print(f"[DataLoader] Loaded {len(df)} rows from {fallback_csv}")
                return df
            else:
                raise ValueError(f"Download failed AND no fallback CSV available. Error: {str(e)}")

    def validate_data(self, df):
        """Check for data quality issues"""
        print("\n[DataLoader] Validating data...")
        print(f"  Shape: {df.shape}")
        
        if 'date' in df.columns:
            print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        
        print(f"  Missing values: {df.isnull().sum().sum()}")
        print(f"  Duplicates: {df.duplicated().sum()}")
        
        if df.isnull().sum().sum() > 0:
            print("  WARNING: Missing values detected")
        if df.duplicated().sum() > 0:
            print("  WARNING: Duplicates detected")


class DataPreprocessor:
    """Handles preprocessing: NA removal, returns, splitting"""

    @staticmethod
    def preprocess(df):
        """
        Core preprocessing steps:
        - Drop NAs
        - Compute log returns
        - Compute simple returns
        
        Args:
            df (pd.DataFrame): Raw OHLCV data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        df = df.copy()
        
        # Remove NaNs
        df = df.dropna()
        print(f"[Preprocessor] After NA removal: {len(df)} rows")
        
        if len(df) == 0:
            print("[Preprocessor] WARNING: All data removed during NA cleanup!")
            return df
        
        # Log returns (for time-series models)
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        
        # Simple returns (for ML features)
        df["return"] = df["Close"].pct_change()
        
        # Remove first row (has NaN in returns)
        df = df.dropna()
        print(f"[Preprocessor] After returns calculation: {len(df)} rows")
        
        return df

    @staticmethod
    def test_stationarity(series, name="Series", critical_value=0.05):
        """
        Augmented Dickey-Fuller test for stationarity
        
        Args:
            series (pd.Series): Time series to test
            name (str): Series name for printing
            critical_value (float): Significance level
            
        Returns:
            bool: True if stationary
        """
        from statsmodels.tsa.stattools import adfuller
        
        # Check if series has data
        clean_series = series.dropna()
        if len(clean_series) < 3:
            print(f"\n[ADF Test] {name}")
            print(f"  WARNING: Series too short ({len(clean_series)} rows) - skipping stationarity test")
            return False
        
        try:
            result = adfuller(clean_series, autolag='AIC')
            p_val = result[1]
            adf_stat = result[0]
            
            is_stationary = p_val < critical_value
            status = "✓ Stationary" if is_stationary else "✗ Non-stationary (need differencing)"
            
            print(f"\n[ADF Test] {name}")
            print(f"  ADF Statistic: {adf_stat:.6f}")
            print(f"  p-value: {p_val:.6f}")
            print(f"  Status: {status}")
            print(f"  Crit values: {result[4]}")
            
            return is_stationary
        except Exception as e:
            print(f"\n[ADF Test] {name}")
            print(f"  ERROR: {str(e)}")
            return False

    @staticmethod
    def difference_series(series, order=1):
        """Difference a series to make it stationary"""
        diff = series.diff(order).dropna()
        print(f"[Preprocessor] Differenced series (order={order}): {len(diff)} rows")
        return diff

    @staticmethod
    def time_series_split(df, train_ratio=0.8):
        """
        Split data respecting time order (crucial for time-series!)
        
        Args:
            df (pd.DataFrame): Full dataset
            train_ratio (float): Proportion for training
            
        Returns:
            tuple: (train_df, test_df)
        """
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        print(f"\n[TimeSeriesSplit]")
        print(f"  Total: {len(df)} rows")
        
        if 'date' in train_df.columns:
            print(f"  Train: {len(train_df)} rows ({train_df['date'].min()} to {train_df['date'].max()})")
            print(f"  Test: {len(test_df)} rows ({test_df['date'].min()} to {test_df['date'].max()})")
        else:
            print(f"  Train: {len(train_df)} rows")
            print(f"  Test: {len(test_df)} rows")
        
        return train_df, test_df
