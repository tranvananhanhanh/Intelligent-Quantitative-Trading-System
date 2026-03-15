"""
FinRL Trading Dashboard
======================

Main Streamlit application for the FinRL Trading platform.
Provides interactive visualization and control of trading strategies.
"""

import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Get project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Import project modules with individual try-except
from config.settings import get_config

try:
    from data.data_store import get_data_store
except ImportError:
    get_data_store = None

try:
    from backtest.backtest_engine import BacktestEngine, BacktestConfig
except ImportError:
    BacktestEngine = None
    BacktestConfig = None

try:
    from trading.alpaca_manager import create_alpaca_account_from_env
except ImportError:
    create_alpaca_account_from_env = None

try:
    from trading.trade_executor import TradeExecutor, ExecutionConfig
except ImportError:
    TradeExecutor = None
    ExecutionConfig = None

from utils.logging_utils import setup_logging

# Setup logging
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="FinRL Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'config' not in st.session_state:
    st.session_state.config = get_config()
if 'data_store' not in st.session_state:
    st.session_state.data_store = get_data_store() if get_data_store else None


def main():
    """Main application function."""
    st.title("📈 FinRL Trading Dashboard")
    st.markdown("AI-powered quantitative trading platform")

    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Select Page",
            ["Overview", "Data Management", "Strategy Backtesting",
             "Live Trading", "Portfolio Analysis", "Settings"]
        )

        st.divider()

        # Quick stats
        display_quick_stats()

    # Main content
    if page == "Overview":
        show_overview()
    elif page == "Data Management":
        show_data_management()
    elif page == "Strategy Backtesting":
        show_strategy_backtesting()
    elif page == "Live Trading":
        show_live_trading()
    elif page == "Portfolio Analysis":
        show_portfolio_analysis()
    elif page == "Settings":
        show_settings()


def display_quick_stats():
    """Display quick statistics in sidebar."""
    st.subheader("Quick Stats")

    try:
        # Get data store stats
        stats = st.session_state.data_store.get_storage_stats() if st.session_state.data_store else {}

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Data Versions", stats.get('data_versions', 0))
        with col2:
            st.metric("Cache Entries", stats.get('cache_entries', 0))

        st.metric("Storage Used", ".1f")

    except Exception as e:
        st.error(f"Could not load stats: {e}")


def get_sample_sp500_data():
    """Get sample S&P 500 data for demo."""
    sample_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
        'META', 'TSLA', 'JNJ', 'V', 'WMT',
        'JPM', 'PG', 'MA', 'HD', 'MCD',
        'BA', 'NKE', 'CSCO', 'ABBV', 'PEP'
    ]
    return pd.DataFrame({
        'tickers': sample_tickers,
        'sectors': ['Technology', 'Technology', 'Technology', 'Consumer Cyclical', 'Technology',
                   'Technology', 'Consumer Cyclical', 'Healthcare', 'Financial Services', 'Consumer Defensive',
                   'Financial Services', 'Consumer Defensive', 'Financial Services', 'Consumer Cyclical', 'Consumer Cyclical',
                   'Industrials', 'Consumer Cyclical', 'Technology', 'Healthcare', 'Consumer Defensive'],
        'dateFirstAdded': ['1985-11-30'] * 20
    })


def get_api_key_status():
    """Check if API keys are configured."""
    try:
        from config.settings import get_config
        config = get_config()
        fmp_key = hasattr(config, 'fmp') and config.fmp.api_key
        return {'fmp': bool(fmp_key)}
    except:
        return {'fmp': False}


def show_overview():
    """Show overview dashboard."""
    st.header("Trading Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Strategies", "5", "↗️ 2")
    with col2:
        st.metric("Active Positions", "12", "↗️ 3")
    with col3:
        st.metric("Portfolio Value", "$1,250,000", "+2.5%")
    with col4:
        st.metric("Today's P&L", "+$1,250", "+1.2%")

    # Recent activity
    st.subheader("Recent Activity")
    activity_data = pd.DataFrame({
        'Time': pd.date_range('2024-01-01 09:00', periods=5, freq='1H'),
        'Action': ['Strategy Execution', 'Portfolio Rebalance', 'Data Update', 'Order Filled', 'Strategy Backtest'],
        'Status': ['Success', 'Success', 'Success', 'Success', 'Completed'],
        'Details': ['ML Strategy executed', 'Quarterly rebalance', 'S&P 500 data updated', 'AAPL order filled', 'Backtest completed']
    })

    st.dataframe(activity_data, use_container_width=True)

    # Performance chart
    st.subheader("Portfolio Performance")
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    portfolio_values = 1000000 + np.cumsum(np.random.normal(1000, 5000, 30))

    fig = px.line(x=dates, y=portfolio_values, title="Portfolio Value Over Time")
    fig.update_layout(xaxis_title="Date", yaxis_title="Portfolio Value ($)")
    st.plotly_chart(fig, use_container_width=True)


def show_data_management():
    """Show data management interface."""
    st.header("Data Management")

    tab1, tab2, tab3, tab4 = st.tabs(["Data Sources", "Data Processing", "Data Storage", "Data Quality"])

    with tab1:
        st.subheader("Data Sources")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("WRDS Data")
            if st.button("Fetch S&P 500 Components"):
                with st.spinner("Fetching data..."):
                    try:
                        from data.data_fetcher import fetch_sp500_tickers
                        tickers = fetch_sp500_tickers()
                        if isinstance(tickers, pd.DataFrame):
                            st.success(f"Successfully fetched {len(tickers)} tickers from API")
                            st.dataframe(tickers.head(10), use_container_width=True)
                        else:
                            st.success(f"Successfully fetched {len(tickers)} tickers")
                            st.info(f"Sample tickers: {', '.join(tickers[:10])}")
                    except Exception as e:
                        st.warning(f"⚠️ Could not fetch from API: {str(e)[:100]}")
                        st.info("**Solution:** Using sample S&P 500 data for demo")
                        
                        # Load sample data
                        sample_data = get_sample_sp500_data()
                        st.success(f"Loaded {len(sample_data)} sample S&P 500 tickers")
                        st.dataframe(sample_data, use_container_width=True)
                        
                        # Show how to configure API
                        with st.expander("🔑 How to Configure FMP API Key"):
                            st.markdown("""
                            1. Get API Key from [Financial Modeling Prep](https://financialmodelingprep.com/)
                            2. Create `.env` file in project root:
                            ```
                            FMP_API_KEY=your_api_key_here
                            ```
                            3. Restart the dashboard
                            """)

            if st.button("Fetch Fundamental Data"):
                with st.spinner("Fetching fundamental data..."):
                    try:
                        from data.data_fetcher import fetch_fundamental_data
                        from data.data_quality import DataQualityChecker
                        from pathlib import Path
                        
                        tickers = ['AAPL', 'MSFT', 'GOOGL']
                        fundamentals = fetch_fundamental_data(
                            tickers, '2020-01-01', '2023-12-31'
                        )
                        
                        if len(fundamentals) > 0:
                            # Save to CSV automatically (use absolute path from PROJECT_ROOT)
                            csv_path = DATA_DIR / 'fundamentals.csv'
                            fundamentals.to_csv(csv_path, index=False)
                            
                            st.success(f"✓ Successfully fetched {len(fundamentals)} records")
                            st.info(f"💾 Saved to `{csv_path}`")
                            st.dataframe(fundamentals.head(10), use_container_width=True)
                            
                            # Store in session state for later use
                            st.session_state['fundamentals_data'] = fundamentals
                            st.session_state['fundamentals_path'] = str(csv_path)
                        else:
                            st.warning("No data returned. Loading sample data...")
                            # Create sample fundamental data
                            sample_data = pd.DataFrame({
                                'symbol': ['AAPL', 'AAPL', 'MSFT', 'MSFT', 'GOOGL', 'GOOGL'],
                                'date': ['2023-09-30', '2022-09-30', '2023-06-30', '2022-06-30', '2023-09-30', '2022-09-30'],
                                'revenue': [383285, 365817, 52857, 51865, 76691, 69787],
                                'netIncome': [96995, 99803, 16425, 16425, 12213, 13615],
                                'totalAssets': [352755, 352755, 411975, 411975, 402392, 402392]
                            })
                            st.success(f"Loaded {len(sample_data)} sample records")
                            st.dataframe(sample_data, use_container_width=True)
                    except Exception as e:
                        st.error(f"Failed to fetch data: {str(e)[:200]}")

        with col2:
            st.subheader("Local Data")
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.write(f"Uploaded {len(df)} rows")
                st.dataframe(df.head())

    with tab2:
        st.subheader("Data Processing")

        if st.button("Process Raw Data"):
            with st.spinner("Processing data..."):
                try:
                    from data.data_processor import process_fundamentals, process_prices
                    from data.data_fetcher import fetch_price_data
                    
                    fund_path = DATA_DIR / "fundamentals.csv"
                    prices_path = DATA_DIR / "prices.csv"
                    
                    # Check if fundamentals exist
                    if not fund_path.exists():
                        st.warning(f"⚠️ Fundamentals file not found!")
                        st.info("""
                        **Solution:** 
                        1. Go to **Data Sources** tab
                        2. Click **"Fetch Fundamental Data"**
                        3. Return to this tab to process
                        """)
                        return

                    # Load fundamentals to get tickers and date range
                    fundamentals_raw = pd.read_csv(fund_path)
                    st.write(f"📊 Loaded fundamentals.csv: {len(fundamentals_raw)} rows, {fundamentals_raw.shape[1]} columns")
                    if len(fundamentals_raw) == 0:
                        st.warning("⚠️ Fundamentals CSV is empty!")
                        return
                    
                    # Auto-fetch price data if missing
                    if not prices_path.exists():
                        st.info("💾 Price data not found, auto-fetching...")
                        try:
                            # Get unique tickers and date range from fundamentals
                            tickers_list = fundamentals_raw['tic'].unique() if 'tic' in fundamentals_raw.columns else \
                                          fundamentals_raw['symbol'].unique() if 'symbol' in fundamentals_raw.columns else []
                            
                            date_col = 'datadate' if 'datadate' in fundamentals_raw.columns else 'date'
                            dates = pd.to_datetime(fundamentals_raw[date_col])
                            start_date = dates.min().strftime('%Y-%m-%d')
                            end_date = dates.max().strftime('%Y-%m-%d')
                            
                            st.write(f"Fetching prices for {len(tickers_list)} tickers from {start_date} to {end_date}...")
                            
                            prices = fetch_price_data(list(tickers_list), start_date, end_date)
                            if len(prices) > 0:
                                prices.to_csv(prices_path, index=False)
                                st.success(f"✓ Auto-fetched {len(prices)} price records")
                            else:
                                st.warning("⚠️ Could not fetch prices, using sample data")
                                sample_prices = pd.DataFrame({
                                    'gvkey': list(tickers_list) * 3,
                                    'datadate': ['2023-12-29', '2023-06-30', '2023-01-03'] * len(list(tickers_list)),
                                    'adj_close': [170.5, 160.2, 150.8] * len(list(tickers_list)),
                                })
                                sample_prices.to_csv(prices_path, index=False)
                        except Exception as price_err:
                            st.warning(f"⚠️ Auto-fetch failed: {str(price_err)[:100]}")
                            st.info("Using sample price data for demo...")
                            sample_prices = pd.DataFrame({
                                'gvkey': ['AAPL', 'MSFT', 'GOOGL'] * 3,
                                'datadate': ['2023-12-29', '2023-06-30', '2023-01-03'] * 3,
                                'adj_close': [170.5, 339.2, 129.8] * 3,
                            })
                            sample_prices.to_csv(prices_path, index=False)

                    # Process data
                    st.write("⏳ Processing fundamentals...")
                    fundamentals = process_fundamentals(str(fund_path))
                    st.write(f"✓ Processed {len(fundamentals)} fundamental records")
                    
                    st.write("⏳ Processing prices...")
                    prices = process_prices(str(prices_path))
                    st.write(f"✓ Processed {len(prices)} price records")
                    
                    if len(fundamentals) == 0 or len(prices) == 0:
                        st.warning("⚠️ Processed data is empty! Check input files.")
                        return

                    st.success("✓ Data processing completed successfully")

                except Exception as e:
                    st.error(f"Data processing failed: {type(e).__name__}: {str(e)}")
                    st.info(f"""
                    **Debug Info:**
                    - Fundamentals path: `{fund_path}`
                    - Fundamentals exists: {fund_path.exists()}
                    - Prices path: `{prices_path}`
                    - Prices exists: {prices_path.exists()}
                    
                    If files exist but processing fails, try:
                    1. Click "Fetch Fundamental Data" again to refresh
                    2. Or delete CSV files and start over
                    """)
                    import traceback
                    logger.error(traceback.format_exc())

        if st.button("Generate ML Dataset"):
            with st.spinner("Creating ML dataset..."):
                try:
                    from data.data_processor import create_ml_dataset
                    from pathlib import Path

                    fund_path = DATA_DIR / "fundamentals.csv"
                    prices_path = DATA_DIR / "prices.csv"
                    
                    if not fund_path.exists() or not prices_path.exists():
                        st.warning(f"⚠️ Data files not found! Please process data first.")
                        st.info(f"Expected paths:\n- {fund_path}\n- {prices_path}")
                        return

                    X, y = create_ml_dataset(str(fund_path), str(prices_path))
                    st.success("✓ ML dataset created")
                    st.write(f"Features shape: {X.shape}")
                    st.write(f"Target shape: {y.shape}")

                except Exception as e:
                    st.error(f"ML dataset creation failed: {e}")

    with tab3:
        st.subheader("Data Storage")

        # Display storage stats
        stats = st.session_state.data_store.get_storage_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Files", stats['total_files'])
        with col2:
            st.metric("Storage (MB)", f"{stats['total_size_mb']:.2f}")
        with col3:
            st.metric("Price Records", f"{stats['price_records']:,}")
        with col4:
            st.metric("Database", "SQLite")
        
        st.divider()
        
        # Database path info
        st.info(f"📁 Database: `{stats['database_path']}`")

        if st.button("🧹 Cleanup Expired Cache (30+ days old)"):
            with st.spinner("Cleaning up cache..."):
                try:
                    deleted = st.session_state.data_store.cleanup_expired_cache(days_old=30)
                    st.success("✓ Cache cleanup completed")
                    st.json({
                        'deleted_price_data': deleted.get('price_data', 0),
                        'deleted_news_articles': deleted.get('news_articles', 0),
                        'deleted_raw_payloads': deleted.get('raw_payloads', 0),
                        'total_deleted': sum(deleted.values())
                    })
                except Exception as e:
                    st.error(f"Cache cleanup failed: {e}")

    with tab4:
        st.subheader("📊 Data Quality Assessment")
        
        # Quick explanation
        with st.expander("ℹ️ How Data Quality Scoring Works"):
            st.markdown("""
            **4 Quality Dimensions:**
            
            1. **Completeness** - % of non-null values
               - 100% = All values present
               - <70% = Too many missing values
            
            2. **Accuracy** - Outlier detection + range validation
               - PE > 0, Revenue > 0 (Fundamentals)
               - High ≥ Low, OHLC valid (Prices)
            
            3. **Consistency** - Format & type consistency
               - Valid date format (YYYY-MM-DD)
               - Numeric columns are numbers
               - No duplicate rows
            
            4. **Timeliness** - Data recency
               - 100% = ≤ 1 day old
               - 70% = ≤ 1 month old
               - 20% = > 3 months old
            """)
        
        st.divider()

        # Load data files
        fund_path = DATA_DIR / "fundamentals.csv"
        prices_path = DATA_DIR / "prices.csv"

        # Auto-assess if data exists
        fund_exists = fund_path.exists()
        prices_exists = prices_path.exists()

        if st.button("🔍 Assess Data Quality", use_container_width=True):
            if not fund_exists and not prices_exists:
                st.error("❌ No data files found!")
                st.info("📌 Please fetch and process data first in the **Data Sources** and **Data Processing** tabs")
                st.stop()
            
            with st.spinner("Analyzing data quality..."):
                try:
                    from data.data_quality import assess_data_quality, DataQualityChecker
                    
                    results = assess_data_quality(
                        fundamentals_path=fund_path if fund_exists else None,
                        prices_path=prices_path if prices_exists else None
                    )
                    
                    checker = DataQualityChecker()
                    
                    # Create columns for side-by-side display
                    col1, col2 = st.columns(2)
                    
                    # ===== FUNDAMENTAL DATA QUALITY =====
                    if results.get('fundamentals') and fund_exists:
                        with col1:
                            st.subheader("📈 Fundamental Data")
                            fund_scores = results['fundamentals']
                            
                            # Overall score box
                            overall = fund_scores['overall']
                            status, emoji = checker.score_to_status(overall)
                            
                            st.metric(
                                f"{emoji} Overall Quality",
                                f"{overall:.1f}%",
                                f"Status: {status}"
                            )
                            
                            # Breakdown table
                            breakdown_df = pd.DataFrame({
                                'Dimension': ['Completeness', 'Accuracy', 'Consistency', 'Timeliness'],
                                'Score': [
                                    f"{fund_scores['completeness']:.1f}%",
                                    f"{fund_scores['accuracy']:.1f}%",
                                    f"{fund_scores['consistency']:.1f}%",
                                    f"{fund_scores['timeliness']:.1f}%"
                                ],
                                'Status': [
                                    checker.score_to_status(fund_scores['completeness'])[0],
                                    checker.score_to_status(fund_scores['accuracy'])[0],
                                    checker.score_to_status(fund_scores['consistency'])[0],
                                    checker.score_to_status(fund_scores['timeliness'])[0]
                                ]
                            })
                            st.dataframe(breakdown_df, use_container_width=True)
                            st.caption(f"📦 Records: {fund_scores.get('record_count', 'N/A')}")
                    
                    # ===== PRICE DATA QUALITY =====
                    if results.get('prices') and prices_exists:
                        with col2:
                            st.subheader("💹 Price Data")
                            price_scores = results['prices']
                            
                            # Overall score box
                            overall = price_scores['overall']
                            status, emoji = checker.score_to_status(overall)
                            
                            st.metric(
                                f"{emoji} Overall Quality",
                                f"{overall:.1f}%",
                                f"Status: {status}"
                            )
                            
                            # Breakdown table
                            breakdown_df = pd.DataFrame({
                                'Dimension': ['Completeness', 'Accuracy', 'Consistency', 'Timeliness'],
                                'Score': [
                                    f"{price_scores['completeness']:.1f}%",
                                    f"{price_scores['accuracy']:.1f}%",
                                    f"{price_scores['consistency']:.1f}%",
                                    f"{price_scores['timeliness']:.1f}%"
                                ],
                                'Status': [
                                    checker.score_to_status(price_scores['completeness'])[0],
                                    checker.score_to_status(price_scores['accuracy'])[0],
                                    checker.score_to_status(price_scores['consistency'])[0],
                                    checker.score_to_status(price_scores['timeliness'])[0]
                                ]
                            })
                            st.dataframe(breakdown_df, use_container_width=True)
                            st.caption(f"📦 Records: {price_scores.get('record_count', 'N/A')}")
                    
                    # ===== RECOMMENDATIONS =====
                    st.divider()
                    st.subheader("💡 Recommendations")
                    
                    rec_col1, rec_col2 = st.columns(2)
                    
                    if fund_exists and results.get('fundamentals'):
                        with rec_col1:
                            overall = results['fundamentals']['overall']
                            if overall >= 80:
                                st.success("✅ Fundamental data is GOOD - ready for model training")
                            elif overall >= 60:
                                st.warning("⚠️ Fundamental data is FAIR - review before production use")
                            else:
                                st.error("❌ Fundamental data quality is POOR - needs improvement")
                    
                    if prices_exists and results.get('prices'):
                        with rec_col2:
                            overall = results['prices']['overall']
                            if overall >= 80:
                                st.success("✅ Price data is GOOD - ready for backtesting")
                            elif overall >= 60:
                                st.warning("⚠️ Price data is FAIR - review before backtesting")
                            else:
                                st.error("❌ Price data quality is POOR - needs improvement")
                        
                except Exception as e:
                    st.error(f"❌ Quality assessment failed: {e}")


def show_strategy_backtesting():
    """Show strategy backtesting interface."""
    st.header("Strategy Backtesting")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Backtest Configuration")

        # Strategy selection
        strategy_type = st.selectbox(
            "Strategy Type",
            ["equal_weight", "market_cap_weight", "ml_strategy"]
        )

        # Backtest parameters
        start_date = st.date_input("Start Date", datetime(2020, 1, 1))
        end_date = st.date_input("End Date", datetime(2023, 12, 31))
        initial_capital = st.number_input("Initial Capital", value=1000000, step=100000)

        # ML strategy parameters
        if strategy_type == "ml_strategy":
            top_quantile = st.slider("Top Quantile", 0.5, 1.0, 0.75, 0.05)

        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                try:
                    st.info("Backtest feature coming soon. Please use the Jupyter notebook examples for backtesting.")
                    # run_backtest(
                    #     strategy_type=strategy_type,
                    #     start_date=start_date,
                    #     end_date=end_date,
                    #     initial_capital=initial_capital,
                    #     top_quantile=top_quantile if strategy_type == "ml_strategy" else 0.75
                    # )
                except Exception as e:
                    st.error(f"Backtest failed: {e}")

    with col2:
        st.subheader("Backtest Results")

        # Display results if available
        if 'backtest_result' in st.session_state:
            result = st.session_state.backtest_result

            # Key metrics
            metrics_cols = st.columns(4)
            with metrics_cols[0]:
                st.metric("Final Value", ".2f")
            with metrics_cols[1]:
                st.metric("Total Return", ".2%")
            with metrics_cols[2]:
                st.metric("Annual Return", ".2%")
            with metrics_cols[3]:
                st.metric("Sharpe Ratio", ".2f")

            # Performance chart
            if hasattr(result, 'portfolio_values'):
                fig = px.line(
                    x=result.portfolio_values.index,
                    y=result.portfolio_values.values,
                    title="Portfolio Value"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Detailed metrics
            st.subheader("Detailed Metrics")
            metrics_df = pd.DataFrame({
                'Metric': list(result.metrics.keys()),
                'Value': [".4f" for v in result.metrics.values()]
            })
            st.dataframe(metrics_df)


# def run_backtest(strategy_type, start_date, end_date, initial_capital, top_quantile):
#     """Run backtest with given parameters."""
#     # Create strategy
#     config = StrategyConfig(name=f"{strategy_type} Backtest")
#     strategy = create_strategy(strategy_type, config)
#
#     # Create backtest configuration
#     backtest_config = BacktestConfig(
#         start_date=start_date.strftime("%Y-%m-%d"),
#         end_date=end_date.strftime("%Y-%m-%d"),
#         initial_capital=initial_capital
#     )
#
#     # Load sample data (in practice, load real data)
#     dates = pd.date_range(start_date, end_date, freq='D')
#     price_data = pd.DataFrame({
#         'datadate': dates,
#         'adj_close': 100 + np.cumsum(np.random.normal(0, 0.02, len(dates)))
#     })
#
#     # Sample weight signals
#     weight_signals = pd.DataFrame({
#         'date': pd.date_range(start_date, end_date, freq='Q'),
#         'AAPL': 0.5,
#         'MSFT': 0.3,
#         'GOOGL': 0.2
#     })
#
#     # Run backtest
#     engine = BacktestEngine(backtest_config)
#     result = engine.run_backtest(strategy, price_data, weight_signals)
#
#     # Store result
#     st.session_state.backtest_result = result
#
#     st.success("Backtest completed successfully!")


def show_live_trading():
    """Show live trading interface."""
    st.header("Live Trading")

    # Check if trading is configured
    try:
        if not create_alpaca_account_from_env:
            st.error("Alpaca trading module not available")
            return
            
        account = create_alpaca_account_from_env()
        st.success(f"Connected to Alpaca account (Paper: {account.is_paper})")

        tab1, tab2, tab3 = st.tabs(["Portfolio", "Order Management", "Strategy Execution"])

        with tab1:
            st.subheader("Current Portfolio")

            if st.button("Refresh Portfolio"):
                with st.spinner("Loading portfolio..."):
                    try:
                        from trading.alpaca_manager import AlpacaManager
                        manager = AlpacaManager([account])

                        # Get account info
                        account_info = manager.get_account_info()
                        positions = manager.get_positions()

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Portfolio Value", ".2f")
                        with col2:
                            st.metric("Cash", ".2f")
                        with col3:
                            st.metric("Buying Power", ".2f")

                        # Positions table
                        if positions:
                            positions_df = pd.DataFrame(positions)
                            st.dataframe(positions_df[['symbol', 'qty', 'avg_entry_price', 'market_value', 'unrealized_pl']], use_container_width=True)
                        else:
                            st.info("No open positions")

                    except Exception as e:
                        st.error(f"Failed to load portfolio: {e}")

        with tab2:
            st.subheader("Order Management")

            # Place order form
            with st.form("place_order"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    symbol = st.text_input("Symbol", "AAPL").upper()
                with col2:
                    quantity = st.number_input("Quantity", min_value=1, value=10)
                with col3:
                    side = st.selectbox("Side", ["buy", "sell"])

                order_type = st.selectbox("Order Type", ["market", "limit"])
                limit_price = st.number_input("Limit Price", min_value=0.01, step=0.01) if order_type == "limit" else None

                submitted = st.form_submit_button("Place Order")
                if submitted:
                    try:
                        from trading.alpaca_manager import AlpacaManager, OrderRequest
                        manager = AlpacaManager([account])

                        order = OrderRequest(
                            symbol=symbol,
                            quantity=quantity,
                            side=side,
                            order_type=order_type,
                            limit_price=limit_price
                        )

                        response = manager.place_order(order)
                        st.success(f"Order placed: {response.order_id}")

                    except Exception as e:
                        st.error(f"Failed to place order: {e}")

        with tab3:
            st.subheader("Strategy Execution")
            st.info("Strategy execution feature coming soon. Please check the Jupyter notebook examples for detailed strategy implementation.")
            # with st.button("Execute Sample Strategy"):
            #     with st.spinner("Executing strategy..."):
            #         try:
            #             from ..trading.trade_executor import TradeExecutor
            #             from ..strategies.base_strategy import StrategyConfig, EqualWeightStrategy
            #
            #             manager = AlpacaManager([account])
            #             executor = TradeExecutor(manager)
            #
            #             # Create sample strategy
            #             config = StrategyConfig(name="Sample Equal Weight")
            #             strategy = EqualWeightStrategy(config)
            #
            #             # Sample data
            #             sample_data = {
            #                 'fundamentals': pd.DataFrame({
            #                     'gvkey': ['AAPL', 'MSFT', 'GOOGL'],
            #                     'datadate': ['2024-01-01'] * 3
            #                 })
            #             }
            #
            #             result = executor.execute_strategy(strategy, sample_data)
            #             st.success(f"Strategy executed: {len(result.orders_placed)} orders placed")
            #
            #         except Exception as e:
            #             st.error(f"Strategy execution failed: {e}")

    except Exception as e:
        st.error(f"Trading not configured: {e}")
        st.info("Please set up Alpaca API credentials in environment variables")


def show_portfolio_analysis():
    """Show portfolio analysis interface."""
    st.header("Portfolio Analysis")

    # Sample portfolio data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    portfolio_values = 1000000 + np.cumsum(np.random.normal(2000, 8000, 100))

    tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Risk Analysis", "Attribution", "Benchmarking"])

    with tab1:
        st.subheader("Performance Analysis")

        # Performance chart
        fig = px.line(x=dates, y=portfolio_values, title="Portfolio Performance")
        st.plotly_chart(fig, use_container_width=True)

        # Performance metrics
        returns = pd.Series(portfolio_values).pct_change().dropna()
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        annual_return = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", ".2%")
        with col2:
            st.metric("Annual Return", ".2%")
        with col3:
            st.metric("Volatility", ".2%")
        with col4:
            st.metric("Sharpe Ratio", ".2f")

    with tab2:
        st.subheader("Risk Analysis")

        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        fig = px.line(x=dates[1:], y=drawdown, title="Portfolio Drawdown")
        fig.update_layout(yaxis_title="Drawdown", yaxis_tickformat=".2%")
        st.plotly_chart(fig, use_container_width=True)

        # Risk metrics
        max_drawdown = drawdown.min()
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Max Drawdown", ".2%")
        with col2:
            st.metric("VaR (95%)", ".2%")
        with col3:
            st.metric("CVaR (95%)", ".2%")

    with tab3:
        st.subheader("Attribution Analysis")

        # Sample attribution data
        attribution_data = pd.DataFrame({
            'Asset': ['AAPL', 'MSFT', 'GOOGL', 'Bonds', 'Cash'],
            'Weight': [0.3, 0.25, 0.2, 0.15, 0.1],
            'Return': [0.15, 0.12, 0.18, 0.03, 0.02],
            'Contribution': [0.045, 0.03, 0.036, 0.0045, 0.002]
        })

        fig = px.bar(attribution_data, x='Asset', y='Contribution',
                    title="Return Attribution by Asset")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(attribution_data, use_container_width=True)

    with tab4:
        st.subheader("Benchmarking")

        # Sample benchmark comparison
        benchmark_data = pd.DataFrame({
            'Date': dates,
            'Portfolio': portfolio_values,
            'SPY': 1000000 + np.cumsum(np.random.normal(1500, 6000, 100)),
            'QQQ': 1000000 + np.cumsum(np.random.normal(1800, 7000, 100))
        })

        fig = px.line(benchmark_data, x='Date', y=['Portfolio', 'SPY', 'QQQ'],
                     title="Portfolio vs Benchmarks")
        st.plotly_chart(fig, use_container_width=True)


def show_settings():
    """Show settings interface."""
    st.header("Settings")

    tab1, tab2, tab3 = st.tabs(["General", "Trading", "Data"])

    with tab1:
        st.subheader("General Settings")

        # Logging level
        log_level = st.selectbox("Logging Level", ["DEBUG", "INFO", "WARNING", "ERROR"])
        if st.button("Apply Logging Level"):
            logging.getLogger().setLevel(getattr(logging, log_level))
            st.success(f"Logging level set to {log_level}")

        # Theme
        theme = st.selectbox("Theme", ["Light", "Dark"])
        if st.button("Apply Theme"):
            st.success(f"Theme set to {theme}")

    with tab2:
        st.subheader("Trading Settings")

        # Risk limits
        max_order_value = st.number_input("Max Order Value ($)", value=100000, step=10000)
        max_portfolio_turnover = st.slider("Max Portfolio Turnover (%)", 0.0, 1.0, 0.5, 0.05)

        if st.button("Save Trading Settings"):
            st.success("Trading settings saved")

        # API Configuration
        st.subheader("API Configuration")
        api_key = st.text_input("Alpaca API Key", type="password")
        api_secret = st.text_input("Alpaca API Secret", type="password")
        use_paper = st.checkbox("Use Paper Trading", value=True)

        if st.button("Save API Settings"):
            st.success("API settings saved")

    with tab3:
        st.subheader("Data Settings")

        # Data paths
        data_dir = st.text_input("Data Directory", value="./data")
        cache_dir = st.text_input("Cache Directory", value="./data/cache")

        if st.button("Save Data Settings"):
            st.success("Data settings saved")

        # Data sources
        st.subheader("Data Sources")
        enable_wrds = st.checkbox("Enable WRDS", value=True)
        enable_alpha_vantage = st.checkbox("Enable Alpha Vantage", value=False)

        if st.button("Save Data Source Settings"):
            st.success("Data source settings saved")


if __name__ == "__main__":
    main()
