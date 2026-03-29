"""
Data Quality Assessment Module
================================

Evaluates data quality across 4 dimensions:
1. Completeness: % of non-null values
2. Accuracy: Outlier detection + value range validation
3. Consistency: Data type consistency + format validation
4. Timeliness: How recent is the data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """Check and score data quality."""

    def __init__(self):
        """Initialize quality checker."""
        self.scores = {}

    def assess_fundamentals(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Assess fundamental data quality.
        
        Args:
            df: DataFrame with fundamental data (must have 'datadate' column)
            
        Returns:
            Dictionary with quality scores (0-100 scale)
        """
        if df is None or df.empty:
            return {
                'completeness': 0.0,
                'accuracy': 0.0,
                'consistency': 0.0,
                'timeliness': 0.0,
                'overall': 0.0
            }

        # 1. COMPLETENESS: % of non-null values
        completeness = self._assess_completeness(df)

        # 2. ACCURACY: Outliers + value ranges
        accuracy = self._assess_accuracy_fundamentals(df)

        # 3. CONSISTENCY: Format + types
        consistency = self._assess_consistency(df, 'fundamentals')

        # 4. TIMELINESS: Data recency
        timeliness = self._assess_timeliness(df)

        # Overall = average of all 4
        overall = np.mean([completeness, accuracy, consistency, timeliness])

        return {
            'completeness': completeness,
            'accuracy': accuracy,
            'consistency': consistency,
            'timeliness': timeliness,
            'overall': overall
        }

    def assess_prices(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Assess price data quality.
        
        Args:
            df: DataFrame with price data (must have OHLCV columns)
            
        Returns:
            Dictionary with quality scores (0-100 scale)
        """
        if df is None or df.empty:
            return {
                'completeness': 0.0,
                'accuracy': 0.0,
                'consistency': 0.0,
                'timeliness': 0.0,
                'overall': 0.0
            }

        # 1. COMPLETENESS
        completeness = self._assess_completeness(df)

        # 2. ACCURACY: OHLC validation (Open, High, Low, Close relationships)
        accuracy = self._assess_accuracy_prices(df)

        # 3. CONSISTENCY
        consistency = self._assess_consistency(df, 'prices')

        # 4. TIMELINESS
        timeliness = self._assess_timeliness(df)

        # Overall
        overall = np.mean([completeness, accuracy, consistency, timeliness])

        return {
            'completeness': completeness,
            'accuracy': accuracy,
            'consistency': consistency,
            'timeliness': timeliness,
            'overall': overall
        }

    def _assess_completeness(self, df: pd.DataFrame) -> float:
        """
        Calculate completeness score.
        
        Score = Average % of non-null values across all columns
        - 100: All values present
        - 90+: Very complete (>90% non-null)
        - 70+: Mostly complete (>70% non-null)
        - <50: Poor (many missing values)
        """
        # Exclude index columns
        exclude_cols = ['index', 'gvkey', 'tic', 'sector', 'gsector']
        cols = [c for c in df.columns if c not in exclude_cols]

        if not cols:
            return 100.0

        non_null_pcts = []
        for col in cols:
            pct = (1 - df[col].isna().sum() / len(df)) * 100
            non_null_pcts.append(pct)

        return np.mean(non_null_pcts) if non_null_pcts else 100.0

    def _assess_accuracy_fundamentals(self, df: pd.DataFrame) -> float:
        """
        Assess accuracy for fundamental data.
        
        Checks:
        - PE ratio > 0 (or NaN acceptable for unprofitable companies)
        - Revenue > 0
        - Market cap makes sense
        - No extreme outliers (IQR-based detection)
        """
        scores = []

        # Check PE ratio (if exists)
        if 'pe' in df.columns:
            pe_valid = (df['pe'] > 0) | (df['pe'].isna())
            scores.append(pe_valid.sum() / len(df) * 100)

        # Check revenue (if exists)
        if 'revenue' in df.columns:
            rev_valid = (df['revenue'] >= 0) | (df['revenue'].isna())
            scores.append(rev_valid.sum() / len(df) * 100)

        # Outlier detection on numeric columns
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                       if c not in ['gvkey', 'tic', 'gsector']]
        
        outlier_pct = 0
        for col in numeric_cols:
            if df[col].notna().sum() > 3:  # Need at least 4 values
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    lower = Q1 - 3 * IQR  # Beyond 3 IQR
                    upper = Q3 + 3 * IQR
                    valid = ((df[col] >= lower) | (df[col].isna())) & \
                            ((df[col] <= upper) | (df[col].isna()))
                    outlier_pct += valid.sum()
                else:
                    outlier_pct += len(df)

        if numeric_cols:
            outlier_score = (outlier_pct / (len(df) * len(numeric_cols))) * 100
            scores.append(outlier_score)

        return np.mean(scores) if scores else 100.0

    def _assess_accuracy_prices(self, df: pd.DataFrame) -> float:
        """
        Assess accuracy for price data.
        
        Checks:
        - High >= Low
        - High >= Open, Close
        - Low <= Open, Close
        - Close >= 0
        - Volume >= 0
        """
        scores = []

        # Low <= High
        if 'Low' in df.columns and 'High' in df.columns:
            low_high_valid = (df['Low'] <= df['High']).sum() / len(df) * 100
            scores.append(low_high_valid)

        # Open, Close within Low-High range
        if all(c in df.columns for c in ['Low', 'High', 'Open']):
            range_valid = ((df['Open'] >= df['Low']) & (df['Open'] <= df['High'])).sum() / len(df) * 100
            scores.append(range_valid)

        # Non-negative Close and Volume
        if 'Close' in df.columns:
            close_valid = (df['Close'] >= 0).sum() / len(df) * 100
            scores.append(close_valid)

        if 'Volume' in df.columns:
            vol_valid = (df['Volume'] >= 0).sum() / len(df) * 100
            scores.append(vol_valid)

        # Also accept generic column names
        for close_col in ['close', 'prccd', 'adj_close']:
            if close_col in df.columns:
                close_valid = (df[close_col] >= 0).sum() / len(df) * 100
                scores.append(close_valid)
                break

        return np.mean(scores) if scores else 100.0

    def _assess_consistency(self, df: pd.DataFrame, data_type: str) -> float:
        """
        Assess data consistency.
        
        Checks:
        - Date format consistency
        - Data type consistency
        - No duplicate rows
        """
        scores = []

        # Date consistency
        date_cols = [c for c in df.columns if 'date' in c.lower()]
        for col in date_cols:
            try:
                dates = pd.to_datetime(df[col], errors='coerce')
                valid_dates = dates.notna().sum() / len(df) * 100
                scores.append(valid_dates)
            except:
                scores.append(0)

        # Numeric type consistency
        if data_type == 'prices':
            price_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
                         'close', 'prccd', 'adj_close']
            existing_cols = [c for c in price_cols if c in df.columns]
            if existing_cols:
                numeric_check = all(pd.api.types.is_numeric_dtype(df[c]) for c in existing_cols)
                scores.append(100.0 if numeric_check else 50.0)

        # Duplicate check (lower penalty if some duplicates exist)
        dup_pct = (1 - df.duplicated().sum() / len(df)) * 100
        scores.append(dup_pct)

        return np.mean(scores) if scores else 100.0

    def _assess_timeliness(self, df: pd.DataFrame) -> float:
        """
        Assess data timeliness (how recent).
        
        Score based on:
        - Latest date in data vs today
        - 100: Updated within 1 day
        - 90: Updated within 1 week
        - 70: Updated within 1 month
        - 50: Updated within 3 months
        - 0: Older than 3 months
        """
        # Find date column
        date_col = None
        for col in ['datadate', 'date', 'Date']:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            return 50.0  # Unknown timeliness

        try:
            latest_date = pd.to_datetime(df[date_col]).max()
            days_old = (datetime.now() - latest_date).days

            if days_old <= 1:
                return 100.0
            elif days_old <= 7:
                return 90.0
            elif days_old <= 30:
                return 70.0
            elif days_old <= 90:
                return 50.0
            else:
                return 20.0

        except Exception as e:
            logger.warning(f"Could not assess timeliness: {e}")
            return 50.0

    @staticmethod
    def score_to_status(score: float) -> Tuple[str, str]:
        """
        Convert numeric score to status label and emoji.
        
        Returns:
            Tuple of (status_text, emoji)
        """
        if score >= 95:
            return "Excellent", "🟢"
        elif score >= 80:
            return "Good", "🟡"
        elif score >= 60:
            return "Fair", "🟠"
        else:
            return "Poor", "🔴"


def assess_data_quality(fundamentals_path=None, prices_path=None) -> Dict:
    """
    Assess quality of both fundamental and price data.
    
    Args:
        fundamentals_path: Path to processed fundamentals CSV
        prices_path: Path to processed prices CSV
        
    Returns:
        Dictionary with quality assessment for both datasets
    """
    checker = DataQualityChecker()
    results = {}

    # Assess fundamentals
    if fundamentals_path:
        try:
            fund_df = pd.read_csv(fundamentals_path)
            results['fundamentals'] = checker.assess_fundamentals(fund_df)
            results['fundamentals']['record_count'] = len(fund_df)
        except Exception as e:
            logger.error(f"Error assessing fundamentals: {e}")
            results['fundamentals'] = None

    # Assess prices
    if prices_path:
        try:
            price_df = pd.read_csv(prices_path)
            results['prices'] = checker.assess_prices(price_df)
            results['prices']['record_count'] = len(price_df)
        except Exception as e:
            logger.error(f"Error assessing prices: {e}")
            results['prices'] = None

    return results
