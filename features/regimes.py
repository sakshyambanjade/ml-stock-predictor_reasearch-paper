import numpy as np
import pandas as pd


class MarketRegimeAnalyzer:
    """Segment data into market regimes for evaluation across conditions"""

    @staticmethod
    def define_bull_bear(df, window=60):
        """
        Define bull/bear markets based on rolling returns
        
        Args:
            df (pd.DataFrame): Data with Close prices
            window (int): Rolling window for returns
            
        Returns:
            pd.DataFrame: Data with Market_Regime column
        """
        df = df.copy()
        
        # Calculate rolling returns
        df["rolling_return"] = df["Close"].pct_change(window)
        
        # Label regimes
        df["Market_Regime"] = np.where(df["rolling_return"] > 0, "Bull", "Bear")
        
        print(f"\n[MarketRegime] Bull/Bear (window={window})")
        print(df["Market_Regime"].value_counts())
        
        return df

    @staticmethod
    def define_volatility_regimes(df, split_method="median", percentile=50):
        """
        Define high/low volatility regimes
        
        Args:
            df (pd.DataFrame): Data with Volatility column
            split_method (str): "median" or "percentile"
            percentile (int): Percentile for split (if using percentile)
            
        Returns:
            pd.DataFrame: Data with Volatility_Regime column
        """
        df = df.copy()
        
        if split_method == "median":
            threshold = df["Volatility"].median()
            print(f"\n[VolatilityRegime] Using median = {threshold:.6f}")
        else:
            threshold = df["Volatility"].quantile(percentile / 100)
            print(f"\n[VolatilityRegime] Using {percentile}th percentile = {threshold:.6f}")
        
        df["Volatility_Regime"] = np.where(
            df["Volatility"] > threshold,
            "High_Vol",
            "Low_Vol"
        )
        
        print(df["Volatility_Regime"].value_counts())
        
        return df

    @staticmethod
    def combine_regimes(df):
        """Combine Market_Regime and Volatility_Regime"""
        df = df.copy()
        df["Combined_Regime"] = df["Market_Regime"] + "_" + df["Volatility_Regime"]
        
        print(f"\n[CombinedRegime]")
        print(df["Combined_Regime"].value_counts())
        
        return df

    @staticmethod
    def get_regime_data(df, regime_col, regime_value):
        """Extract data for a specific regime"""
        regime_df = df[df[regime_col] == regime_value].copy()
        print(f"\n[GetRegimeData] {regime_col}={regime_value}: {len(regime_df)} rows")
        return regime_df

    @staticmethod
    def add_all_regimes(df, config):
        """
        Apply all regime definitions using config
        
        Args:
            df (pd.DataFrame): Data with Volatility column
            config (dict): Configuration from config.py
            
        Returns:
            pd.DataFrame: Data with all regime columns
        """
        window = config["rolling_window"]
        split_method = config["volatility_split"]
        
        df = MarketRegimeAnalyzer.define_bull_bear(df, window=window)
        df = MarketRegimeAnalyzer.define_volatility_regimes(df, split_method=split_method)
        df = MarketRegimeAnalyzer.combine_regimes(df)
        
        return df

    @staticmethod
    def regime_summary(df):
        """Print summary of regime distribution"""
        print("\n" + "=" * 60)
        print("MARKET REGIME SUMMARY")
        print("=" * 60)
        
        print("\nMarket Trend:")
        print(df["Market_Regime"].value_counts())
        print(f"  Bull: {(df['Market_Regime'] == 'Bull').sum() / len(df) * 100:.1f}%")
        print(f"  Bear: {(df['Market_Regime'] == 'Bear').sum() / len(df) * 100:.1f}%")
        
        print("\nVolatility Regime:")
        print(df["Volatility_Regime"].value_counts())
        print(f"  High: {(df['Volatility_Regime'] == 'High_Vol').sum() / len(df) * 100:.1f}%")
        print(f"  Low: {(df['Volatility_Regime'] == 'Low_Vol').sum() / len(df) * 100:.1f}%")
        
        print("\nCombined Regimes:")
        print(df["Combined_Regime"].value_counts())