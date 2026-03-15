"""
Data Processor Module
====================

Handles data preprocessing and feature engineering:
- Fundamental data processing
- Price data processing
- Feature engineering for ML models
- Data quality checks and cleaning
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DataProcessor:
    """Data processor for fundamental and price data."""

    def __init__(self, data_dir: str = "./data"):
        """
        Initialize data processor.

        Args:
            data_dir: Base directory for data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def process_fundamental_data(self, raw_fundamentals_path: str,
                               processed_path: str = None) -> pd.DataFrame:
        """
        Process raw fundamental data into ML-ready format.

        Args:
            raw_fundamentals_path: Path to raw fundamental data
            processed_path: Path to save processed data (optional)

        Returns:
            Processed fundamental data DataFrame
        """
        logger.info(f"Processing fundamental data from {raw_fundamentals_path}")

        # Load raw data
        try:
            df = pd.read_csv(raw_fundamentals_path)
        except FileNotFoundError:
            logger.error(f"File not found: {raw_fundamentals_path}")
            return pd.DataFrame()
        except Exception as load_err:
            logger.error(f"Error loading {raw_fundamentals_path}: {load_err}")
            return pd.DataFrame()
            
        if len(df) == 0:
            logger.warning("Empty fundamental data provided")
            return df
            
        logger.info(f"Loaded {len(df)} raw fundamental records")

        # Basic data cleaning
        df = self._clean_fundamental_data(df)

        # Feature engineering
        df = self._engineer_fundamental_features(df)

        # Handle missing values
        df = self._handle_missing_values(df)

        # Save processed data
        if processed_path:
            processed_path = Path(processed_path)
            processed_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(processed_path, index=False)
            logger.info(f"Saved processed data to {processed_path}")

        logger.info(f"Processed {len(df)} fundamental records")
        return df

    def _clean_fundamental_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean fundamental data. Supports both FMP and Yahoo Finance formats."""
        if len(df) == 0:
            logger.warning("Empty fundamental data provided")
            return df
            
        # Normalize column names for consistency
        if 'symbol' in df.columns and 'tic' not in df.columns:
            df['tic'] = df['symbol']
        if 'gvkey' not in df.columns:
            df['gvkey'] = df.get('tic', df.get('symbol', 'UNKNOWN'))
        if 'date' in df.columns and 'datadate' not in df.columns:
            df['datadate'] = df['date']
            
        # Remove duplicates
        df = df.drop_duplicates(subset=['gvkey', 'datadate'])

        # Convert date column
        df['datadate'] = pd.to_datetime(df['datadate'])

        # Normalize sector column name (gsector -> sector)
        if 'gsector' in df.columns and 'sector' not in df.columns:
            df['sector'] = df['gsector']

        # Handle price columns based on format
        # FMP format: prccd + ajexdi
        # Yahoo format: adj_close_q or could be just Close
        if 'prccd' in df.columns and 'ajexdi' in df.columns:
            # FMP Compustat format
            df = df[df['prccd'] > 0]  # Valid prices
            df = df[df['ajexdi'] > 0]  # Valid adjustment factors
            df['adj_price'] = df['prccd'] / df['ajexdi']
        elif 'adj_close_q' in df.columns and df['adj_close_q'].notna().sum() > 0:
            # Yahoo Finance format from get_fundamental_data
            df = df[df['adj_close_q'] > 0]
            df['adj_price'] = df['adj_close_q']
        elif 'adj_close_q' in df.columns:
            # Yahoo Finance format but adj_close_q might be all NaN - that's ok
            df['adj_price'] = np.nan
            logger.debug("No adj_close_q values available, using NaN for prices")
        elif 'Close' in df.columns:
            # Alternative format
            df = df[df['Close'] > 0]
            df['adj_price'] = df['Close']
        else:
            # No price data, but that's ok for fundamentals
            logger.warning("No price column found in fundamental data")
            df['adj_price'] = np.nan

        return df

    def _engineer_fundamental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer fundamental features for ML models."""
        # Basic profitability ratios
        if 'revenue' in df.columns and 'net_income' in df.columns:
            df['profit_margin'] = df['net_income'] / df['revenue']

        # Growth rates (quarterly)
        df = df.sort_values(['gvkey', 'datadate'])
        if 'adj_price' in df.columns and df['adj_price'].notna().sum() > 0:
            df['price_growth_qtr'] = df.groupby('gvkey')['adj_price'].pct_change()
            df['price_volatility_4q'] = df.groupby('gvkey')['adj_price'].rolling(4).std().reset_index(0, drop=True)

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in fundamental data."""
        if len(df) == 0:
            return df
            
        # Drop columns with too many missing values
        missing_threshold = 0.5
        missing_ratios = df.isnull().mean()
        columns_to_drop = missing_ratios[missing_ratios > missing_threshold].index
        df = df.drop(columns=columns_to_drop)

        logger.info(f"Dropped {len(columns_to_drop)} columns with >{missing_threshold*100}% missing values")

        # Fill remaining missing values with median by sector if sector column exists and has data
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if 'sector' in df.columns and df['sector'].notna().sum() > 0:
            try:
                # Only fill for sectors that exist
                df[numeric_columns] = df.groupby('sector', observed=True)[numeric_columns].transform(
                    lambda x: x.fillna(x.median())
                )
            except Exception as sect_err:
                logger.warning(f"Failed to fill by sector: {sect_err}, using global median")
                df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        else:
            # No sector info, use global median
            logger.debug("Using global median to fill missing values")
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

        return df

    def process_price_data(self, raw_prices_path: str,
                          processed_path: str = None) -> pd.DataFrame:
        """
        Process raw price data into ML-ready format.

        Args:
            raw_prices_path: Path to raw price data
            processed_path: Path to save processed data (optional)

        Returns:
            Processed price data DataFrame
        """
        logger.info(f"Processing price data from {raw_prices_path}")

        # Load raw data
        try:
            df = pd.read_csv(raw_prices_path)
        except FileNotFoundError:
            logger.error(f"File not found: {raw_prices_path}")
            return pd.DataFrame()
        except Exception as load_err:
            logger.error(f"Error loading {raw_prices_path}: {load_err}")
            return pd.DataFrame()
            
        if len(df) == 0:
            logger.warning("Empty price data provided")
            return df
            
        logger.info(f"Loaded {len(df)} raw price records")

        # Basic data cleaning
        df = self._clean_price_data(df)

        # Feature engineering
        df = self._engineer_price_features(df)

        # Save processed data
        if processed_path:
            processed_path = Path(processed_path)
            processed_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(processed_path, index=False)
            logger.info(f"Saved processed data to {processed_path}")

        logger.info(f"Processed {len(df)} price records")
        return df

    def _clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean price data. Supports both FMP and Yahoo Finance formats."""
        if len(df) == 0:
            logger.warning("Empty price data provided")
            return df
            
        # Normalize column names
        if 'symbol' in df.columns and 'tic' not in df.columns:
            df['tic'] = df['symbol']
        if 'gvkey' not in df.columns:
            df['gvkey'] = df.get('tic', df.get('symbol', 'UNKNOWN'))
        if 'date' in df.columns and 'datadate' not in df.columns:
            df['datadate'] = df['date']
            
        # Remove duplicates
        df = df.drop_duplicates(subset=['gvkey', 'datadate'])

        # Convert date column
        df['datadate'] = pd.to_datetime(df['datadate'])

        # Handle different price data formats
        if 'prccd' in df.columns and 'ajexdi' in df.columns:
            # FMP Compustat format
            df = df[df['prccd'] > 0]  # Valid prices
            df = df[df['ajexdi'] > 0]  # Valid adjustment factors
            df['adj_close'] = df['prccd'] / df['ajexdi']
            df['adj_open'] = df['prcod'] / df['ajexdi'] if 'prcod' in df.columns else df['adj_close']
            df['adj_high'] = df['prchd'] / df['ajexdi'] if 'prchd' in df.columns else df['adj_close']
            df['adj_low'] = df['prcld'] / df['ajexdi'] if 'prcld' in df.columns else df['adj_close']
        elif 'Adj Close' in df.columns:
            # Yahoo Finance format
            df = df[df['Adj Close'] > 0]
            df['adj_close'] = df['Adj Close']
            df['adj_open'] = df['Open'] if 'Open' in df.columns else df['adj_close']
            df['adj_high'] = df['High'] if 'High' in df.columns else df['adj_close']
            df['adj_low'] = df['Low'] if 'Low' in df.columns else df['adj_close']
        elif 'Close' in df.columns:
            # Alternative format
            df = df[df['Close'] > 0]
            df['adj_close'] = df['Close']
            df['adj_open'] = df['Open'] if 'Open' in df.columns else df['adj_close']
            df['adj_high'] = df['High'] if 'High' in df.columns else df['adj_close']
            df['adj_low'] = df['Low'] if 'Low' in df.columns else df['adj_close']
        else:
            logger.warning("No recognizable price columns found")
            df['adj_close'] = np.nan

        return df

    def _engineer_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer price-based features."""
        # Daily returns
        df = df.sort_values(['gvkey', 'datadate'])
        if 'adj_close' in df.columns and df['adj_close'].notna().sum() > 0:
            df['daily_return'] = df.groupby('gvkey')['adj_close'].pct_change()

            # Technical indicators
            df = self._add_technical_indicators(df)

            # Volatility measures
            if 'daily_return' in df.columns and df['daily_return'].notna().sum() > 0:
                df['volatility_20d'] = df.groupby('gvkey')['daily_return'].rolling(20).std().reset_index(0, drop=True)
                df['volatility_60d'] = df.groupby('gvkey')['daily_return'].rolling(60).std().reset_index(0, drop=True)

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data."""
        if len(df) == 0:
            logger.warning("Cannot calculate technical indicators: empty DataFrame")
            return df
        
        try:
            # Check minimum per-group data
            group_sizes = df.groupby('gvkey').size()
            min_group_size = group_sizes.min() if len(group_sizes) > 0 else 0
            
            # Simple moving averages
            for period in [5, 10, 20, 50, 200]:
                if min_group_size >= period:
                    try:
                        df[f'sma_{period}'] = df.groupby('gvkey')['adj_close'].rolling(period, min_periods=1).mean().reset_index(0, drop=True)
                    except Exception as sma_err:
                        logger.debug(f"Failed to calculate SMA {period}: {sma_err}")
                else:
                    logger.debug(f"Insufficient data for SMA {period} (min group size: {min_group_size})")

            # RSI (Relative Strength Index) - needs at least 14 data points per group
            if min_group_size >= 14:
                df = self._calculate_rsi(df)
            else:
                logger.debug(f"Insufficient data for RSI calculation (min group size: {min_group_size}, need 14)")

            # MACD - needs at least 26 data points per group
            if min_group_size >= 26:
                df = self._calculate_macd(df)
            else:
                logger.debug(f"Insufficient data for MACD calculation (min group size: {min_group_size}, need 26)")
        except Exception as e:
            logger.warning(f"Error in _add_technical_indicators: {e}")
        
        return df

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate RSI indicator."""
        try:
            def rsi_calc(group):
                if len(group) < period:
                    return pd.Series(np.nan, index=group.index)
                delta = group['adj_close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi

            df['rsi_14'] = df.groupby('gvkey').apply(rsi_calc, include_groups=False).reset_index(level=0, drop=True)
        except Exception as e:
            logger.warning(f"Failed to calculate RSI: {e}")
        return df

    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicator."""
        try:
            def macd_calc(group):
                if len(group) < 26:
                    return pd.DataFrame({
                        'macd': np.nan,
                        'signal': np.nan
                    }, index=group.index)
                ema_12 = group['adj_close'].ewm(span=12).mean()
                ema_26 = group['adj_close'].ewm(span=26).mean()
                macd = ema_12 - ema_26
                signal = macd.ewm(span=9).mean()
                return pd.DataFrame({
                    'macd': macd,
                    'signal': signal
                }, index=group.index)

            macd_results = df.groupby('gvkey').apply(macd_calc, include_groups=False).reset_index(level=0, drop=True)
            df['macd'] = macd_results['macd']
            df['macd_signal'] = macd_results['signal']
        except Exception as e:
            logger.warning(f"Failed to calculate MACD: {e}")
            df['macd'] = np.nan
            df['macd_signal'] = np.nan
        return df

    def create_ml_dataset(self, fundamentals_path: str, prices_path: str,
                         target_period: int = 63) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create ML-ready dataset by combining fundamentals and price data.

        Args:
            fundamentals_path: Path to processed fundamental data
            prices_path: Path to processed price data
            target_period: Days to look ahead for target variable

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info("Creating ML dataset...")

        # Load processed data
        fundamentals = pd.read_csv(fundamentals_path)
        prices = pd.read_csv(prices_path)

        logger.info(f"Loaded fundamentals: {len(fundamentals)} rows, prices: {len(prices)} rows")

        # Normalize date column names
        if 'date' in fundamentals.columns and 'datadate' not in fundamentals.columns:
            fundamentals['datadate'] = fundamentals['date']
        if 'date' in prices.columns and 'datadate' not in prices.columns:
            prices['datadate'] = prices['date']
        
        # Ensure we have required columns
        if 'gvkey' not in prices.columns:
            prices['gvkey'] = prices.get('symbol', prices.get('tic', 'UNKNOWN'))
        if 'gvkey' not in fundamentals.columns:
            fundamentals['gvkey'] = fundamentals.get('symbol', fundamentals.get('tic', 'UNKNOWN'))

        # Convert dates
        fundamentals['datadate'] = pd.to_datetime(fundamentals['datadate'])
        prices['datadate'] = pd.to_datetime(prices['datadate'])

        logger.info(f"Fundamentals dates: {fundamentals['datadate'].min()} to {fundamentals['datadate'].max()}")
        logger.info(f"Prices dates: {prices['datadate'].min()} to {prices['datadate'].max()}")

        # Ensure we have adj_close column
        if 'adj_close' not in prices.columns:
            prices['adj_close'] = prices.get('Close', prices.get('Adj Close'))

        if len(prices) == 0 or prices['adj_close'].isna().all():
            logger.error("No valid price data found")
            return pd.DataFrame(), pd.Series()

        # Create target variable (future returns)
        prices = prices.sort_values(['gvkey', 'datadate'])
        
        # Adjust target_period if dataset is too small
        # For small datasets, use next-day returns instead of 63-day
        min_data_points = prices.groupby('gvkey').size().min()
        actual_target_period = min(target_period, max(1, min_data_points - 5))
        
        if actual_target_period != target_period:
            logger.info(f"Adjusted target_period from {target_period} to {actual_target_period} (min group size: {min_data_points})")
        
        prices['future_return'] = prices.groupby('gvkey')['adj_close'].shift(-actual_target_period) / prices['adj_close'] - 1
        prices = prices.dropna(subset=['future_return'])

        logger.info(f"Prices with future_return: {len(prices)} rows (target_period: {actual_target_period})")

        if len(prices) == 0:
            logger.error("No prices with valid future_return after shift")
            return pd.DataFrame(), pd.Series()

        # Get latest fundamentals for each gvkey
        # Since fundamentals from Yahoo are point-in-time (latest info), 
        # we take the latest row per gvkey
        fundamentals_latest = fundamentals.sort_values('datadate').groupby('gvkey').tail(1)
        logger.info(f"Latest fundamentals per gvkey: {len(fundamentals_latest)} rows")

        # Merge: prices LEFT JOIN fundamentals (keep all prices, match with latest fundamental)
        merged = prices.merge(
            fundamentals_latest,
            on='gvkey',
            how='left',
            suffixes=('_price', '_fund')
        )

        logger.info(f"Merged data: {len(merged)} rows")

        if len(merged) == 0:
            logger.error("Merge resulted in empty dataset")
            return pd.DataFrame(), pd.Series()

        # Select features - exclude non-numeric and special columns
        feature_columns = [col for col in merged.columns
                          if col not in ['gvkey', 'datadate_price', 'datadate_fund', 'future_return', 
                                        'tic', 'symbol', 'date', 'adj_close']]
        
        # Keep only numeric features
        feature_columns = [col for col in feature_columns if merged[col].dtype in [np.float64, np.int64]]
        
        logger.info(f"Feature columns before cleanup: {len(feature_columns)}")

        # Clean dataset - drop NaN targets ONLY
        merged = merged.dropna(subset=['future_return'])
        logger.info(f"After removing null targets: {len(merged)} rows")

        if len(merged) == 0:
            logger.warning("No data with valid future_return")
            return pd.DataFrame(), pd.Series()

        # Replace infinite values with NaN
        merged = merged.replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaN values in features with median (not dropping entire rows)
        for col in feature_columns:
            if col in merged.columns and merged[col].dtype in [np.float64, np.int64]:
                if merged[col].isna().sum() > 0:
                    median_val = merged[col].median()
                    if pd.notna(median_val):
                        merged[col] = merged[col].fillna(median_val)
                    else:
                        # All NaN, drop this feature
                        feature_columns.remove(col)
        
        # Final cleanup - ensure target is not NaN
        merged = merged.dropna(subset=['future_return'])
        
        logger.info(f"After cleanup: {len(merged)} rows, {len(feature_columns)} features")

        if len(merged) == 0 or len(feature_columns) == 0:
            logger.warning(f"Empty dataset after cleanup (rows: {len(merged)}, features: {len(feature_columns)})")
            return pd.DataFrame(), pd.Series()

        X = merged[feature_columns]
        y = merged['future_return']

        logger.info(f"Created ML dataset with {len(X)} samples and {len(feature_columns)} features")

        return X, y

    def split_by_sector(self, df: pd.DataFrame, sector_column: str = 'sector',
                       output_dir: str = "./data/processed/sectors") -> Dict[str, pd.DataFrame]:
        """
        Split data by sector for sector-neutral strategies.

        Args:
            df: Input DataFrame
            sector_column: Column name for sector information
            output_dir: Directory to save sector files

        Returns:
            Dictionary of sector DataFrames
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sector_data = {}
        for sector, group in df.groupby(sector_column):
            sector_file = output_dir / f"sector_{sector}.csv"
            group.to_csv(sector_file, index=False)
            sector_data[sector] = group
            logger.info(f"Saved sector {sector} with {len(group)} records")

        return sector_data


# Convenience functions
def process_fundamentals(input_path: str, output_path: str = None) -> pd.DataFrame:
    """Process fundamental data."""
    processor = DataProcessor()
    return processor.process_fundamental_data(input_path, output_path)


def process_prices(input_path: str, output_path: str = None) -> pd.DataFrame:
    """Process price data."""
    processor = DataProcessor()
    return processor.process_price_data(input_path, output_path)


def create_ml_dataset(fundamentals_path: str, prices_path: str,
                     target_period: int = 63) -> Tuple[pd.DataFrame, pd.Series]:
    """Create ML-ready dataset."""
    processor = DataProcessor()
    return processor.create_ml_dataset(fundamentals_path, prices_path, target_period)


def main():
    """
    Main entry point for data processing from CLI.
    Processes raw fundamental and price data into ML-ready format.
    """
    from pathlib import Path
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting data processing...")
    
    # Default paths
    fundamentals_path = './data/fundamentals.csv'
    prices_path = './data/prices.csv'
    output_dir = Path('./data/processed')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Process fundamental data
        if Path(fundamentals_path).exists():
            logger.info(f"Processing fundamentals from {fundamentals_path}")
            fundamentals = process_fundamentals(
                fundamentals_path,
                output_dir / "fundamentals_processed.csv"
            )
            logger.info(f"✓ Processed {len(fundamentals)} fundamental records")
        else:
            logger.warning(f"Fundamental data not found: {fundamentals_path}")
            logger.info("To process fundamental data, please provide data files first.")
            logger.info("You can use the Streamlit dashboard to fetch data:")
            logger.info("  python src/main.py dashboard")
            return
        
        # Process price data
        if Path(prices_path).exists():
            logger.info(f"Processing prices from {prices_path}")
            prices = process_prices(
                prices_path,
                output_dir / "prices_processed.csv"
            )
            logger.info(f"✓ Processed {len(prices)} price records")
        else:
            logger.warning(f"Price data not found: {prices_path}")
        
        # Try to create ML dataset if both files exist
        fund_path = output_dir / "fundamentals_processed.csv"
        prices_path_obj = output_dir / "prices_processed.csv"
        
        if fund_path.exists() and prices_path_obj.exists():
            logger.info("Creating ML dataset...")
            X, y = create_ml_dataset(str(fund_path), str(prices_path_obj))
            logger.info(f"✓ Created ML dataset with {len(X)} samples and {X.shape[1]} features")
            
            # Save ML dataset
            X.to_csv(output_dir / "ml_features.csv", index=False)
            y.to_csv(output_dir / "ml_targets.csv", index=False)
            logger.info(f"✓ Saved ML dataset to {output_dir}")
        
        logger.info("Data processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Process sample data
    try:
        fundamentals = process_fundamentals("./data/fundamentals.csv")
        prices = process_prices("./data/prices.csv")

        # Create ML dataset
        X, y = create_ml_dataset("./data/fundamentals.csv", "./data/prices.csv")
        print(f"Created dataset with {len(X)} samples")

    except FileNotFoundError as e:
        print(f"Sample data not found: {e}")
        print("Run wrds_fetcher.py first to generate sample data")
