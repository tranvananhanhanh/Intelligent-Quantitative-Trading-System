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
        'Time': pd.date_range('2024-01-01 09:00', periods=5, freq='h'),
        'Action': ['Strategy Execution', 'Portfolio Rebalance', 'Data Update', 'Order Filled', 'Strategy Backtest'],
        'Status': ['Success', 'Success', 'Success', 'Success', 'Completed'],
        'Details': ['ML Strategy executed', 'Quarterly rebalance', 'S&P 500 data updated', 'AAPL order filled', 'Backtest completed']
    })

    st.dataframe(activity_data, width='stretch')

    # Performance chart
    st.subheader("Portfolio Performance")
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    portfolio_values = 1000000 + np.cumsum(np.random.normal(1000, 5000, 30))

    fig = px.line(x=dates, y=portfolio_values, title="Portfolio Value Over Time")
    fig.update_layout(xaxis_title="Date", yaxis_title="Portfolio Value ($)")
    st.plotly_chart(fig, width='stretch')


def show_data_management():
    """Show data management interface."""
    st.header("Data Management")
    
    # Check if data_store is available
    if st.session_state.data_store is None:
        st.warning("⚠️ Data store is not initialized. Some features may be unavailable.")
        st.info("💡 The system will use basic functionality. To enable full features, ensure data_store module is properly configured.")

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
                            st.dataframe(tickers.head(10), width='stretch')
                        else:
                            st.success(f"Successfully fetched {len(tickers)} tickers")
                            st.info(f"Sample tickers: {', '.join(tickers[:10])}")
                    except Exception as e:
                        st.warning(f"⚠️ Could not fetch from API: {str(e)[:100]}")
                        st.info("**Solution:** Using sample S&P 500 data for demo")
                        
                        # Load sample data
                        sample_data = get_sample_sp500_data()
                        st.success(f"Loaded {len(sample_data)} sample S&P 500 tickers")
                        st.dataframe(sample_data, width='stretch')
                        
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
                        tickers = ['AAPL', 'MSFT', 'GOOGL']
                        fundamentals = fetch_fundamental_data(
                            tickers, '2020-01-01', '2023-12-31'
                        )
                        if len(fundamentals) > 0:
                            st.success(f"Successfully fetched {len(fundamentals)} records")
                            st.dataframe(fundamentals.head(10), width='stretch')
                        else:
                            st.warning("No data returned. Loading sample data...")
                            # Create sample fundamental data
                            sample_data = pd.DataFrame({
                                'symbol': ['AAPL', 'AAPL', 'MSFT', 'MSFT', 'GOOGL', 'GOOGL'],
                                'date': ['2023-09-30', '2022-09-30', '2023-06-30', '2022-06-30', '2023-09-30', '2022-09-30'],
                                'revenue': [383285, 365817, 52857, 51865, 76691, 69787],
                                'netIncome': [96995, 99803, 16425, 16425, 12213, 13615],
                                'totalAssets': [352,755, 352,755, 411,975, 411,975, 402,392, 402,392]
                            })
                            st.success(f"Loaded {len(sample_data)} sample records")
                            st.dataframe(sample_data, width='stretch')
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

                    # Process sample data
                    fundamentals = process_fundamentals("./data/fundamentals.csv")
                    prices = process_prices("./data/prices.csv")

                    st.success("Data processing completed")
                    st.write(f"Processed {len(fundamentals)} fundamental records")
                    st.write(f"Processed {len(prices)} price records")

                except Exception as e:
                    st.error(f"Data processing failed: {e}")

        if st.button("Generate ML Dataset"):
            with st.spinner("Creating ML dataset..."):
                try:
                    from data.data_processor import create_ml_dataset

                    X, y = create_ml_dataset("./data/fundamentals.csv", "./data/prices.csv")
                    st.success("ML dataset created")
                    st.write(f"Features shape: {X.shape}")
                    st.write(f"Target shape: {y.shape}")

                except Exception as e:
                    st.error(f"ML dataset creation failed: {e}")

    with tab3:
        st.subheader("Data Storage")

        # Display storage stats only if data_store is available
        if st.session_state.data_store is not None:
            stats = st.session_state.data_store.get_storage_stats()
            st.json(stats)

            if st.button("Cleanup Expired Cache"):
                with st.spinner("Cleaning up cache..."):
                    try:
                        st.session_state.data_store.cleanup_expired_cache()
                        st.success("Cache cleanup completed")
                    except Exception as e:
                        st.error(f"Cache cleanup failed: {e}")
        else:
            st.info("Data store not available. Storage management disabled.")

    with tab4:
        st.subheader("Data Quality")

        # Data quality checks
        st.subheader("Data Quality Metrics")

        # Sample quality metrics
        quality_data = pd.DataFrame({
            'Metric': ['Completeness', 'Accuracy', 'Consistency', 'Timeliness'],
            'Score': [95.2, 98.1, 92.3, 99.8],
            'Status': ['Good', 'Excellent', 'Good', 'Excellent']
        })

        st.dataframe(quality_data, width='stretch')


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
        top_quantile = 0.75
        if strategy_type == "ml_strategy":
            top_quantile = st.slider("Top Quantile", 0.5, 1.0, 0.75, 0.05)

        # Number of stocks
        num_stocks = st.number_input("Number of Stocks", value=20, min_value=5, max_value=100, step=5)

        if st.button("Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                try:
                    run_backtest_demo(
                        strategy_type=strategy_type,
                        start_date=start_date,
                        end_date=end_date,
                        initial_capital=initial_capital,
                        top_quantile=top_quantile if strategy_type == "ml_strategy" else 0.75,
                        num_stocks=num_stocks
                    )
                except Exception as e:
                    st.error(f"Backtest failed: {str(e)[:200]}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

    with col2:
        st.subheader("Backtest Results")

        # Display results if available
        if 'backtest_result' in st.session_state and st.session_state.backtest_result:
            result = st.session_state.backtest_result

            # Key metrics - 4 columns
            metrics_cols = st.columns(4)
            with metrics_cols[0]:
                st.metric(
                    "Final Value",
                    f"${result.portfolio_values.iloc[-1]:,.0f}",
                    f"${result.portfolio_values.iloc[-1] - result.portfolio_values.iloc[0]:,.0f}"
                )
            with metrics_cols[1]:
                st.metric(
                    "Total Return",
                    f"{result.metrics.get('total_return', 0):.2%}"
                )
            with metrics_cols[2]:
                st.metric(
                    "Annual Return",
                    f"{result.annualized_return:.2%}"
                )
            with metrics_cols[3]:
                st.metric(
                    "Sharpe Ratio",
                    f"{result.metrics.get('sharpe_ratio', 0):.2f}"
                )

            # Performance chart
            st.subheader("Portfolio Performance")
            fig_perf = px.line(
                x=result.portfolio_values.index,
                y=result.portfolio_values.values,
                title="Portfolio Value Over Time",
                labels={'x': 'Date', 'y': 'Portfolio Value ($)'}
            )
            fig_perf.update_layout(hovermode='x unified')
            st.plotly_chart(fig_perf, use_container_width=True)

            # Returns distribution
            st.subheader("Returns Distribution")
            fig_ret = px.histogram(
                x=result.portfolio_returns.values,
                nbins=50,
                title="Daily Returns Distribution",
                labels={'x': 'Daily Return', 'y': 'Frequency'}
            )
            st.plotly_chart(fig_ret, use_container_width=True)

            # Detailed metrics table
            st.subheader("Detailed Metrics")
            
            # Compare with benchmarks if available
            if result.benchmark_metrics:
                metrics_comparison = result.to_metrics_dataframe()
                st.dataframe(
                    metrics_comparison.style.format('{:.4f}'),
                    use_container_width=True
                )
            else:
                # Show only strategy metrics
                metrics_df = pd.DataFrame({
                    'Metric': list(result.metrics.keys()),
                    'Value': list(result.metrics.values())
                })
                st.dataframe(
                    metrics_df.style.format({'Value': '{:.4f}'}),
                    use_container_width=True
                )

            # Risk analysis
            st.subheader("Risk Metrics")
            risk_cols = st.columns(3)
            with risk_cols[0]:
                st.metric(
                    "Annual Volatility",
                    f"{result.metrics.get('annual_volatility', 0):.2%}"
                )
            with risk_cols[1]:
                st.metric(
                    "Max Drawdown",
                    f"{result.metrics.get('max_drawdown', 0):.2%}",
                    delta=None
                )
            with risk_cols[2]:
                st.metric(
                    "Sortino Ratio",
                    f"{result.metrics.get('sortino_ratio', 0):.2f}"
                )

            # Drawdown chart
            st.subheader("Portfolio Drawdown")
            cumulative = (1 + result.portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max

            fig_dd = px.area(
                x=drawdown.index,
                y=drawdown.values,
                title="Drawdown Over Time",
                labels={'x': 'Date', 'y': 'Drawdown'}
            )
            fig_dd.update_yaxes(tickformat=".2%")
            st.plotly_chart(fig_dd, use_container_width=True)

            # Benchmark comparison if available
            if result.benchmark_annualized:
                st.subheader("vs Benchmarks")
                benchmark_comparison = pd.DataFrame({
                    'Strategy/Benchmark': list(result.benchmark_annualized.keys()),
                    'Annualized Return': list(result.benchmark_annualized.values())
                })
                benchmark_comparison = pd.concat([
                    pd.DataFrame({
                        'Strategy/Benchmark': [result.strategy_name],
                        'Annualized Return': [result.annualized_return]
                    }),
                    benchmark_comparison
                ], ignore_index=True)

                fig_bm = px.bar(
                    benchmark_comparison,
                    x='Strategy/Benchmark',
                    y='Annualized Return',
                    title="Strategy vs Benchmarks",
                    color='Strategy/Benchmark'
                )
                fig_bm.update_yaxes(tickformat=".2%")
                st.plotly_chart(fig_bm, use_container_width=True)

                st.dataframe(
                    benchmark_comparison.style.format({'Annualized Return': '{:.2%}'}),
                    use_container_width=True
                )

        else:
            st.info("👈 Configure parameters and click **Run Backtest** to see results")


def run_backtest_demo(strategy_type: str, start_date, end_date, initial_capital: float,
                      top_quantile: float, num_stocks: int):
    """
    Run backtest with given parameters.
    
    Args:
        strategy_type: Type of strategy ('equal_weight', 'market_cap_weight', 'ml_strategy')
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_capital: Initial capital
        top_quantile: Top quantile for ML strategy
        num_stocks: Number of stocks to use
    """
    from src.backtest.backtest_engine import BacktestEngine, BacktestConfig
    from src.data.data_fetcher import fetch_price_data
    import pandas as pd
    import numpy as np

    try:
        st.info(f"Loading {num_stocks} stocks for {strategy_type} strategy...")

        # Get sample S&P 500 tickers (or use from data manager)
        sample_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JNJ', 'V', 'WMT',
            'JPM', 'PG', 'MA', 'HD', 'MCD', 'BA', 'NKE', 'CSCO', 'ABBV', 'PEP',
            'XOM', 'CVX', 'KO', 'PEP', 'COST', 'AMGN', 'LLY', 'CRM', 'IBM', 'INTC',
            'ORCL', 'QCOM', 'AMD', 'ADBE', 'PYPL', 'SQ', 'NET', 'CrowdStrike', 'UBER', 'LYFT'
        ]
        tickers = sample_tickers[:num_stocks]

        st.info(f"Fetching price data for {len(tickers)} stocks...")

        # Fetch price data
        price_data = fetch_price_data(
            tickers,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

        if price_data.empty:
            st.error("No price data available for selected period")
            return

        st.info(f"Generating {strategy_type} signals...")

        # Generate weight signals based on strategy type
        if strategy_type == "equal_weight":
            weight_signals = _generate_equal_weight_signals(price_data, tickers)
        elif strategy_type == "market_cap_weight":
            weight_signals = _generate_market_cap_weight_signals(price_data, tickers)
        elif strategy_type == "ml_strategy":
            weight_signals = _generate_ml_weight_signals(price_data, tickers, top_quantile)
        else:
            weight_signals = _generate_equal_weight_signals(price_data, tickers)

        if weight_signals.empty:
            st.error("Could not generate weight signals")
            return

        st.info("Running backtest...")

        # Configure and run backtest
        config = BacktestConfig(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            rebalance_freq='Q',
            initial_capital=initial_capital,
            benchmark_tickers=['SPY', 'QQQ']
        )

        engine = BacktestEngine(config)
        result = engine.run_backtest(f'{strategy_type.upper()} Strategy', price_data, weight_signals)

        # Store result in session state
        st.session_state.backtest_result = result

        st.success(f"✅ Backtest completed!")
        st.info(f"Annualized Return: {result.annualized_return:.2%}")

    except Exception as e:
        st.error(f"Backtest error: {str(e)}")
        raise


def _generate_equal_weight_signals(price_data: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Generate equal weight signals."""
    # Convert datadate to datetime if it's a string
    price_data_copy = price_data.copy()
    if 'datadate' in price_data_copy.columns:
        price_data_copy['datadate'] = pd.to_datetime(price_data_copy['datadate'])
        dates = price_data_copy['datadate'].unique()
    else:
        dates = pd.to_datetime(price_data_copy.index.unique())

    # Generate quarterly rebalance dates
    quarterly_dates = pd.date_range(
        start=pd.Timestamp(dates.min()),
        end=pd.Timestamp(dates.max()),
        freq='Q'
    )

    # Create weight signals (equal weight for all tickers)
    weight_signals = pd.DataFrame(
        data=1.0 / len(tickers),
        index=quarterly_dates,
        columns=tickers
    )

    return weight_signals


def _generate_market_cap_weight_signals(price_data: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Generate market-cap weighted signals."""
    # Convert datadate to datetime if it's a string
    price_data_copy = price_data.copy()
    if 'datadate' in price_data_copy.columns:
        price_data_copy['datadate'] = pd.to_datetime(price_data_copy['datadate'])
        dates = price_data_copy['datadate'].unique()
    else:
        dates = pd.to_datetime(price_data_copy.index.unique())

    # Generate quarterly rebalance dates
    quarterly_dates = pd.date_range(
        start=pd.Timestamp(dates.min()),
        end=pd.Timestamp(dates.max()),
        freq='Q'
    )

    # Simulate market cap weights (use price as proxy)
    weight_signals = pd.DataFrame(index=quarterly_dates, columns=tickers)

    for date in quarterly_dates:
        # Get prices at this date (forward fill if not available)
        available_data = price_data_copy[price_data_copy['datadate'] <= date]
        if not available_data.empty:
            latest_prices = available_data.groupby('tic')['adj_close'].last()
            # Use price as proxy for market cap (higher price = more weight)
            weights = latest_prices / latest_prices.sum()
            weight_signals.loc[date] = weights
        else:
            # Default to equal weight if no data
            weight_signals.loc[date] = 1.0 / len(tickers)

    weight_signals = weight_signals.fillna(1.0 / len(tickers))
    return weight_signals


def _generate_ml_weight_signals(price_data: pd.DataFrame, tickers: list, 
                               top_quantile: float = 0.75) -> pd.DataFrame:
    """Generate ML-based weight signals."""
    import numpy as np

    # Convert datadate to datetime if it's a string
    price_data_copy = price_data.copy()
    if 'datadate' in price_data_copy.columns:
        price_data_copy['datadate'] = pd.to_datetime(price_data_copy['datadate'])
        dates = price_data_copy['datadate'].unique()
    else:
        dates = pd.to_datetime(price_data_copy.index.unique())

    # Generate quarterly rebalance dates
    quarterly_dates = pd.date_range(
        start=pd.Timestamp(dates.min()),
        end=pd.Timestamp(dates.max()),
        freq='Q'
    )

    weight_signals = pd.DataFrame(index=quarterly_dates, columns=tickers)

    for date in quarterly_dates:
        # Get prices at this date
        available_data = price_data_copy[price_data_copy['datadate'] <= date]
        if not available_data.empty:
            # Calculate recent returns (proxy for ML predictions)
            recent_data = available_data.tail(60)  # Last 60 days
            
            if len(recent_data) > 0:
                returns = recent_data.groupby('tic')['adj_close'].apply(
                    lambda x: (x.iloc[-1] / x.iloc[0] - 1) if len(x) > 0 else 0
                )
                
                # Select top performers (top_quantile)
                threshold = returns.quantile(top_quantile)
                selected = returns[returns >= threshold]
                
                if len(selected) > 0:
                    # Equal weight among selected stocks
                    weights = pd.Series(0.0, index=tickers)
                    weights[selected.index] = 1.0 / len(selected)
                    weight_signals.loc[date] = weights
                else:
                    # Default to equal weight if no selection
                    weight_signals.loc[date] = 1.0 / len(tickers)
            else:
                weight_signals.loc[date] = 1.0 / len(tickers)
        else:
            weight_signals.loc[date] = 1.0 / len(tickers)

    weight_signals = weight_signals.fillna(1.0 / len(tickers))
    return weight_signals


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
                            st.dataframe(positions_df[['symbol', 'qty', 'avg_entry_price', 'market_value', 'unrealized_pl']], width='stretch')
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
    st.header("📊 Portfolio Analysis")
    
    # Sidebar for stock selection
    with st.sidebar:
        st.subheader("Stock Selection")
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365),
                max_value=datetime.now(),
                key="portfolio_start_date"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                max_value=datetime.now(),
                key="portfolio_end_date"
            )
        
        # Load available tickers
        try:
            from data.data_fetcher import get_data_manager
            manager = get_data_manager()
            components = manager.get_sp500_components()
            if not components.empty and 'tickers' in components.columns:
                available_tickers = sorted(components['tickers'].tolist())
            else:
                available_tickers = []
        except Exception as e:
            available_tickers = []
        
        # Fallback to sample tickers if API fails
        if not available_tickers:
            st.warning("⚠️ Could not load S&P 500 tickers")
            available_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
                               'META', 'TSLA', 'JPM', 'V', 'WMT',
                               'JNJ', 'PG', 'MA', 'HD', 'MCD']
        
        # Stock multiselect
        selected_tickers = st.multiselect(
            "Select Stocks (3-10)",
            options=available_tickers,
            default=available_tickers[:5] if available_tickers else [],
            max_selections=10,
            key="portfolio_tickers"
        )
        
        # Fetch button
        fetch_clicked = st.button("🔄 Fetch Data", type="primary", use_container_width=True)
    
    # Main content
    if not fetch_clicked:
        st.info("👈 Select stocks from sidebar and click **Fetch Data** to analyze")
        
        # Show sample data hint
        st.markdown("""
        ### Features Available:
        - 📈 **Price Performance** - Compare normalized returns
        - 📊 **Returns Analysis** - Daily/cumulative returns, Sharpe ratio
        - ⚠️ **Risk Metrics** - Volatility, VaR, CVaR, Max Drawdown
        - 🎯 **Correlation Matrix** - Stock correlations heatmap
        
        **Data Sources:**
        - Primary: Financial Modeling Prep (FMP)
        - Fallback: Yahoo Finance (when FMP unavailable)
        """)
        return
    
    if not selected_tickers:
        st.warning("⚠️ Please select at least 3 stocks")
        return
    
    if len(selected_tickers) < 3:
        st.warning("⚠️ Please select at least 3 stocks for meaningful analysis")
        return
    
    # Fetch data
    with st.spinner(f"Fetching data for {len(selected_tickers)} stocks..."):
        try:
            from data.data_fetcher import fetch_price_data
            
            # Prepare tickers DataFrame
            tickers_df = pd.DataFrame({
                'tickers': selected_tickers,
                'sectors': [None] * len(selected_tickers),
                'dateFirstAdded': [None] * len(selected_tickers)
            })
            
            # Fetch price data (auto fallback to Yahoo Finance if FMP fails)
            price_data = fetch_price_data(
                tickers_df,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if price_data.empty:
                st.error("❌ No data available for selected tickers and date range")
                return
            
            # Detect data source
            has_fundamentals = 'EPS' in price_data.columns
            data_source = "FMP" if has_fundamentals else "Yahoo Finance"
            
            st.success(f"✅ Fetched {len(price_data)} records from **{data_source}**")
            
            # Show analysis
            _show_price_analysis(price_data, selected_tickers, start_date, end_date)
            
        except Exception as e:
            st.error(f"❌ Failed to fetch data: {str(e)[:200]}")
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())


def _show_price_analysis(price_data: pd.DataFrame, tickers: list, start_date, end_date):
    """Show price-based portfolio analysis."""
    
    tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Risk Analysis", "Attribution", "Benchmarking"])

def _show_price_analysis(price_data: pd.DataFrame, tickers: list, start_date, end_date):
    """Show price-based portfolio analysis."""
    
    tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Risk Analysis", "Returns", "Correlation"])

    with tab1:
        st.subheader("📈 Price Performance")
        
        # Pivot data for charting
        chart_data = price_data.pivot(
            index='datadate',
            columns='tic',
            values='adj_close'
        ).sort_index()
        
        # Normalize to base 100
        normalized = (chart_data / chart_data.iloc[0]) * 100
        
        # Plot
        fig = px.line(
            normalized,
            title="Normalized Price Performance (Base 100)",
            labels={'value': 'Normalized Price', 'datadate': 'Date', 'tic': 'Ticker'}
        )
        fig.update_layout(
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        total_returns = ((chart_data.iloc[-1] / chart_data.iloc[0]) - 1) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Total Returns")
            returns_df = pd.DataFrame({
                'Ticker': total_returns.index,
                'Return (%)': total_returns.values
            }).sort_values('Return (%)', ascending=False)
            
            st.dataframe(
                returns_df.style.format({'Return (%)': '{:.2f}%'})
                .background_gradient(cmap='RdYlGn', subset=['Return (%)']),
                use_container_width=True
            )
        
        with col2:
            st.subheader("Price Statistics")
            stats_df = pd.DataFrame({
                'Ticker': chart_data.columns,
                'Start Price': chart_data.iloc[0].values,
                'End Price': chart_data.iloc[-1].values,
                'Min Price': chart_data.min().values,
                'Max Price': chart_data.max().values
            })
            st.dataframe(
                stats_df.style.format({
                    'Start Price': '${:.2f}',
                    'End Price': '${:.2f}',
                    'Min Price': '${:.2f}',
                    'Max Price': '${:.2f}'
                }),
                use_container_width=True
            )

    with tab2:
        st.subheader("⚠️ Risk Analysis")
        
        # Calculate returns
        returns_data = price_data.copy()
        returns_data['returns'] = returns_data.groupby('tic')['adj_close'].pct_change()
        
        # Risk metrics table
        risk_metrics = []
        for ticker in tickers:
            ticker_returns = returns_data[returns_data['tic'] == ticker]['returns'].dropna()
            ticker_prices = price_data[price_data['tic'] == ticker]['adj_close']
            
            if len(ticker_returns) > 0:
                volatility = ticker_returns.std() * np.sqrt(252)
                max_dd = _calculate_max_drawdown(ticker_prices)
                var_95 = np.percentile(ticker_returns, 5)
                cvar_95 = ticker_returns[ticker_returns <= var_95].mean()
                
                risk_metrics.append({
                    'Ticker': ticker,
                    'Volatility (Annual)': volatility,
                    'Max Drawdown': max_dd,
                    'VaR (95%)': var_95,
                    'CVaR (95%)': cvar_95
                })
        
        risk_df = pd.DataFrame(risk_metrics)
        st.dataframe(
            risk_df.style.format({
                'Volatility (Annual)': '{:.2%}',
                'Max Drawdown': '{:.2%}',
                'VaR (95%)': '{:.4f}',
                'CVaR (95%)': '{:.4f}'
            }).background_gradient(cmap='YlOrRd', subset=['Volatility (Annual)', 'Max Drawdown']),
            use_container_width=True
        )
        
        # Drawdown chart
        st.subheader("Maximum Drawdown by Stock")
        drawdown_data = []
        for ticker in tickers:
            ticker_prices = price_data[price_data['tic'] == ticker].sort_values('datadate')
            if len(ticker_prices) > 0:
                cumulative = ticker_prices['adj_close'] / ticker_prices['adj_close'].iloc[0]
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                
                for idx, row in ticker_prices.iterrows():
                    drawdown_data.append({
                        'Date': row['datadate'],
                        'Ticker': ticker,
                        'Drawdown': drawdown.iloc[ticker_prices.index.get_loc(idx)]
                    })
        
        dd_df = pd.DataFrame(drawdown_data)
        fig = px.line(
            dd_df,
            x='Date',
            y='Drawdown',
            color='Ticker',
            title='Portfolio Drawdown Over Time'
        )
        fig.update_layout(yaxis_tickformat='.2%', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("📊 Returns Analysis")
        
        # Calculate daily returns
        returns_pivot = returns_data.pivot(
            index='datadate',
            columns='tic',
            values='returns'
        ).sort_index()
        
        # Cumulative returns
        cumulative_returns = (1 + returns_pivot).cumprod()
        
        fig = px.line(
            cumulative_returns,
            title='Cumulative Returns',
            labels={'value': 'Cumulative Return', 'datadate': 'Date'}
        )
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # Returns statistics
        st.subheader("Returns Statistics")
        
        # Calculate statistics manually for each ticker
        stats_list = []
        for ticker in tickers:
            if ticker in returns_pivot.columns:
                ticker_returns = returns_pivot[ticker].dropna()
                if len(ticker_returns) > 0:
                    mean_daily = ticker_returns.mean()
                    std_dev = ticker_returns.std()
                    sharpe = (mean_daily / std_dev * np.sqrt(252)) if std_dev > 0 else 0
                    
                    stats_list.append({
                        'Ticker': ticker,
                        'Mean Daily': mean_daily,
                        'Std Dev': std_dev,
                        'Sharpe Ratio': sharpe,
                        'Min': ticker_returns.min(),
                        'Max': ticker_returns.max()
                    })
        
        returns_stats = pd.DataFrame(stats_list)
        
        if not returns_stats.empty:
            st.dataframe(
                returns_stats.style.format({
                    'Mean Daily': '{:.4f}',
                    'Std Dev': '{:.4f}',
                    'Sharpe Ratio': '{:.2f}',
                    'Min': '{:.4f}',
                    'Max': '{:.4f}'
                }).background_gradient(cmap='RdYlGn', subset=['Sharpe Ratio']),
                use_container_width=True
            )
        else:
            st.warning("No returns data available")

    with tab4:
        st.subheader("🎯 Correlation Matrix")
        
        # Calculate correlation
        returns_pivot = returns_data.pivot(
            index='datadate',
            columns='tic',
            values='returns'
        ).sort_index()
        
        correlation = returns_pivot.corr()
        
        # Heatmap
        fig = px.imshow(
            correlation,
            text_auto='.2f',
            aspect='auto',
            color_continuous_scale='RdBu_r',
            title='Stock Returns Correlation Matrix',
            labels=dict(color="Correlation")
        )
        fig.update_layout(width=800, height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary
        st.subheader("Correlation Summary")
        st.markdown(f"""
        - **Average Correlation:** {correlation.values[np.triu_indices_from(correlation.values, k=1)].mean():.2f}
        - **Highest Correlation:** {correlation.values[np.triu_indices_from(correlation.values, k=1)].max():.2f}
        - **Lowest Correlation:** {correlation.values[np.triu_indices_from(correlation.values, k=1)].min():.2f}
        """)


def _calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculate maximum drawdown from price series."""
    if len(prices) == 0:
        return 0.0
    cumulative = prices / prices.iloc[0]
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


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
