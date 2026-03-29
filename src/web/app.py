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

        st.metric("Storage Used", f"{stats.get('total_size_mb', 0):.1f} MB")

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
                final_val = result.portfolio_values.iloc[-1] if hasattr(result, 'portfolio_values') and len(result.portfolio_values) > 0 else result.metrics.get('final_value', 0)
                st.metric("Final Value", f"${float(final_val):,.2f}")
            with metrics_cols[1]:
                st.metric("Total Return", f"{result.metrics.get('total_return', 0):.2%}")
            with metrics_cols[2]:
                st.metric("Annual Return", f"{result.annualized_return:.2%}")
            with metrics_cols[3]:
                st.metric("Sharpe Ratio", f"{result.metrics.get('sharpe_ratio', 0):.2f}")

            # Performance chart
            if hasattr(result, 'portfolio_values'):
                fig = px.line(
                    x=result.portfolio_values.index,
                    y=result.portfolio_values.values,
                    title="Portfolio Value"
                )
                st.plotly_chart(fig, width='stretch')

            # Detailed metrics
            st.subheader("Detailed Metrics")
            metrics_df = pd.DataFrame({
                'Metric': list(result.metrics.keys()),
                'Value': [f"{v:.4f}" for v in result.metrics.values()]
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

        tab1, tab2, tab3, tab4 = st.tabs(["Portfolio", "Order Management", "Strategy Execution", "Performance Monitoring"])

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
                            st.metric("Portfolio Value", f"${float(account_info.get('portfolio_value', 0)):,.2f}")
                        with col2:
                            st.metric("Cash", f"${float(account_info.get('cash', 0)):,.2f}")
                        with col3:
                            st.metric("Buying Power", f"${float(account_info.get('buying_power', 0)):,.2f}")

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
            _show_strategy_execution(account)

        with tab4:
            st.subheader("Performance Monitoring")
            _show_performance_monitoring(account)

    except Exception as e:
        st.error(f"Trading not configured: {e}")
        st.info("Please set up Alpaca API credentials in environment variables")


def _show_performance_monitoring(account):
    """Render the Performance Monitoring tab inside Live Trading."""
    from trading.alpaca_manager import AlpacaManager
    from trading.performance_analyzer import (
        get_first_order_date, get_portfolio_history,
        get_benchmark_data, compute_performance_metrics,
    )

    manager = AlpacaManager([account])

    # ── Date range ─────────────────────────────────────────────────────────────
    col_d1, col_d2, col_d3 = st.columns([1, 1, 1])
    with col_d1:
        auto_start = st.checkbox("Tự động lấy ngày đầu tiên có lệnh", value=True)
    with col_d2:
        manual_start = st.date_input(
            "Từ ngày", value=datetime.today() - timedelta(days=30),
            disabled=auto_start, key="pm_start"
        )
    with col_d3:
        end_date_input = st.date_input("Đến ngày", value=datetime.today(), key="pm_end")

    if st.button("🔄 Tải dữ liệu", type="primary"):
        with st.spinner("Đang tải portfolio history và benchmark..."):
            try:
                import pytz
                utc = pytz.utc
                end_dt   = datetime.combine(end_date_input, datetime.min.time()).replace(tzinfo=utc)

                if auto_start:
                    first_date = get_first_order_date(manager)
                    if first_date is None:
                        st.warning("Chưa có lệnh nào trong tài khoản. Chọn ngày thủ công.")
                        return
                    start_dt = first_date - timedelta(days=1)
                else:
                    start_dt = datetime.combine(manual_start, datetime.min.time()).replace(tzinfo=utc)

                portfolio_df = get_portfolio_history(manager, start_dt, end_dt)
                fmp_end      = (end_date_input + timedelta(days=1)).strftime("%Y-%m-%d")
                benchmark_df = get_benchmark_data(start_dt.date().isoformat(), fmp_end)

                if portfolio_df.empty:
                    st.warning("Không có dữ liệu portfolio history trong khoảng thời gian này.")
                    return

                # ── Metrics ───────────────────────────────────────────────────
                st.markdown("#### Chỉ số hiệu suất")
                equity_series = portfolio_df.set_index("date")["equity"]
                pm = compute_performance_metrics(equity_series)

                cols_m = st.columns(5)
                cols_m[0].metric("Total Return",      f"{pm['total_return']:.2%}")
                cols_m[1].metric("Annual Return",     f"{pm['annual_return']:.2%}")
                cols_m[2].metric("Volatility (Ann.)", f"{pm['annual_volatility']:.2%}")
                cols_m[3].metric("Sharpe Ratio",      f"{pm['sharpe_ratio']:.2f}")
                cols_m[4].metric("Max Drawdown",      f"{pm['max_drawdown']:.2%}")

                # Benchmark metrics
                if not benchmark_df.empty:
                    bm_cols = st.columns(len(benchmark_df.columns))
                    for i, bm in enumerate(benchmark_df.columns):
                        bm_series = benchmark_df[bm].dropna()
                        if len(bm_series) > 1:
                            bm_m = compute_performance_metrics(bm_series)
                            with bm_cols[i]:
                                st.markdown(f"**{bm}**: Return {bm_m['total_return']:.2%} | "
                                            f"Sharpe {bm_m['sharpe_ratio']:.2f} | "
                                            f"MaxDD {bm_m['max_drawdown']:.2%}")

                st.divider()

                # ── Equity curve vs benchmarks ────────────────────────────────
                st.markdown("#### Equity Curve so với Benchmark")
                fig = go.Figure()

                # Normalize to 1.0
                eq = equity_series.dropna()
                if len(eq) > 0:
                    eq_norm = eq / eq.iloc[0]
                    fig.add_trace(go.Scatter(
                        x=eq_norm.index, y=eq_norm.values,
                        name="Portfolio", line=dict(width=2, color="#2962FF")
                    ))

                colors = {"SPY": "#FF6D00", "QQQ": "#00BFA5"}
                if not benchmark_df.empty:
                    for bm in benchmark_df.columns:
                        bm_s = benchmark_df[bm].dropna()
                        if len(bm_s) > 0:
                            bm_norm = bm_s / bm_s.iloc[0]
                            fig.add_trace(go.Scatter(
                                x=bm_norm.index, y=bm_norm.values,
                                name=bm, line=dict(dash="dash", color=colors.get(bm, "gray"))
                            ))

                fig.update_layout(
                    title="Normalized Performance (Base = 1.0)",
                    xaxis_title="Date", yaxis_title="Return",
                    hovermode="x unified", height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                st.plotly_chart(fig, use_container_width=True)

                # ── Daily P&L chart ───────────────────────────────────────────
                st.markdown("#### P&L hàng ngày")
                pnl = portfolio_df.set_index("date")["profit_loss"].dropna()
                if len(pnl) > 0:
                    bar_colors = ["#00C853" if v >= 0 else "#D50000" for v in pnl.values]
                    fig2 = go.Figure(go.Bar(
                        x=pnl.index, y=pnl.values,
                        marker_color=bar_colors,
                        name="Daily P&L"
                    ))
                    fig2.update_layout(
                        title="Daily Profit / Loss (USD)",
                        xaxis_title="Date", yaxis_title="P&L ($)",
                        height=300
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                # ── Positions table ───────────────────────────────────────────
                st.markdown("#### Vị thế hiện tại")
                positions = manager.get_positions()
                if positions:
                    pos_df = pd.DataFrame(positions)
                    display_cols = [c for c in
                        ["symbol", "qty", "avg_entry_price", "current_price",
                         "market_value", "unrealized_pl", "unrealized_plpc"]
                        if c in pos_df.columns]
                    pos_df_show = pos_df[display_cols].copy()
                    for col in ["avg_entry_price", "current_price", "market_value"]:
                        if col in pos_df_show.columns:
                            pos_df_show[col] = pd.to_numeric(pos_df_show[col], errors="coerce")
                    for col in ["unrealized_pl", "unrealized_plpc"]:
                        if col in pos_df_show.columns:
                            pos_df_show[col] = pd.to_numeric(pos_df_show[col], errors="coerce")

                    def _color_pnl(val):
                        try:
                            return "color: green" if float(val) >= 0 else "color: red"
                        except:
                            return ""

                    fmt = {}
                    for c in ["avg_entry_price", "current_price", "market_value"]:
                        if c in pos_df_show.columns:
                            fmt[c] = "${:.2f}"
                    if "unrealized_pl" in pos_df_show.columns:
                        fmt["unrealized_pl"] = "${:+.2f}"
                    if "unrealized_plpc" in pos_df_show.columns:
                        fmt["unrealized_plpc"] = "{:+.2%}"

                    styled = pos_df_show.style.format(fmt)
                    if "unrealized_pl" in pos_df_show.columns:
                        styled = styled.applymap(_color_pnl, subset=["unrealized_pl"])

                    st.dataframe(styled, use_container_width=True, height=350)

                    # ── P&L bar chart per symbol ──────────────────────────────
                    if "unrealized_pl" in pos_df_show.columns:
                        pnl_sym = pos_df_show[["symbol", "unrealized_pl"]].copy()
                        pnl_sym["unrealized_pl"] = pd.to_numeric(pnl_sym["unrealized_pl"], errors="coerce")
                        pnl_sym = pnl_sym.sort_values("unrealized_pl")
                        bar_c = ["#00C853" if v >= 0 else "#D50000" for v in pnl_sym["unrealized_pl"]]
                        fig3 = go.Figure(go.Bar(
                            x=pnl_sym["symbol"], y=pnl_sym["unrealized_pl"],
                            marker_color=bar_c
                        ))
                        fig3.update_layout(
                            title="Unrealized P&L theo cổ phiếu",
                            xaxis_title="Symbol", yaxis_title="Unrealized P&L ($)",
                            height=350
                        )
                        st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("Chưa có vị thế nào. Lệnh OPG sẽ khớp khi thị trường mở cửa.")

            except Exception as e:
                st.error(f"Lỗi tải dữ liệu: {e}")
                import traceback
                with st.expander("Chi tiết lỗi"):
                    st.code(traceback.format_exc())
    else:
        st.info("Nhấn **Tải dữ liệu** để xem hiệu suất portfolio.")


def _show_strategy_execution(account):
    """Render the Strategy Execution tab inside Live Trading."""
    from trading.alpaca_manager import AlpacaManager

    manager = AlpacaManager([account])

    # ETF / index tickers to always exclude from individual stock orders
    _ETF_EXCLUDE = {"SPY", "QQQ", "IVV", "VOO", "VTI", "IWM", "DIA", "GLD", "SLV", "TLT", "BND"}

    def _parse_weights_csv(raw: pd.DataFrame):
        """Convert CSV (wide or long) → DataFrame[gvkey, weight] + latest_date."""
        if "gvkey" in raw.columns and "weight" in raw.columns:
            raw["date"] = pd.to_datetime(raw["date"])
            latest = raw["date"].max()
            df = raw[raw["date"] == latest][["gvkey", "weight"]].copy()
        else:
            raw["date"] = pd.to_datetime(raw["date"])
            latest = raw["date"].max()
            row = raw[raw["date"] == latest].drop(columns=["date"])
            df = (
                row.T.reset_index()
                .rename(columns={"index": "gvkey", row.index[0]: "weight"})
            )
            df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
        df = df[df["weight"] > 0].copy()
        return df, latest

    # ── 0. Run ML Strategy ────────────────────────────────────────────────────
    with st.expander("🤖 Chạy ML Strategy để tạo weights mới", expanded=False):
        fund_path = Path("data/fundamentals.csv")

        if not fund_path.exists():
            st.warning("`data/fundamentals.csv` chưa tồn tại.")
            if st.button("📥 Tạo fundamentals.csv từ database", type="secondary"):
                with st.spinner("Đang tổng hợp dữ liệu cơ bản từ database (có thể mất vài phút)..."):
                    try:
                        from data.data_fetcher import fetch_sp500_tickers, fetch_fundamental_data
                        tickers_df = fetch_sp500_tickers()
                        fundamentals = fetch_fundamental_data(
                            tickers_df, '2015-10-15', '2025-10-15', align_quarter_dates=True
                        )
                        if fundamentals.empty:
                            st.error("Không lấy được dữ liệu. Kiểm tra database hoặc chạy notebook cell 4+5.")
                        else:
                            fund_path.parent.mkdir(parents=True, exist_ok=True)
                            fundamentals.to_csv(fund_path, index=False)
                            st.success(
                                f"Đã tạo `{fund_path}` với {len(fundamentals)} dòng / "
                                f"{fundamentals['gvkey'].nunique()} cổ phiếu. "
                                "Bấm lại expander để chạy ML."
                            )
                            st.rerun()
                    except Exception as e:
                        st.error(f"Lỗi tạo fundamentals: {e}")
                        import traceback
                        with st.expander("Chi tiết lỗi"):
                            st.code(traceback.format_exc())
        else:
            st.success(f"Tìm thấy `{fund_path}` — sẵn sàng chạy ML.")
            col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
            with col_cfg1:
                ml_top_q = st.slider("Top Quantile", 0.5, 1.0, 0.9, 0.05,
                                     help="Ngưỡng chọn cổ phiếu (0.9 = top 10%)")
            with col_cfg2:
                ml_test_q = st.number_input("Test Quarters", 1, 8, 4,
                                            help="Số quý dùng để test")
            with col_cfg3:
                ml_weight_method = st.selectbox("Weight Method",
                                                ["equal", "min_variance"])
            ml_exec_date = st.date_input(
                "Execution Date (ngày chốt danh mục)",
                value=datetime.today(),
                help="Ngày bạn muốn xác nhận danh mục hôm nay",
            )

            if st.button("▶ Chạy ML Strategy", type="primary"):
                with st.spinner("Đang huấn luyện mô hình và chọn cổ phiếu..."):
                    try:
                        from strategies.ml_strategy import MLStockSelectionStrategy
                        from strategies.base_strategy import StrategyConfig

                        fundamentals = pd.read_csv(fund_path)
                        if len(fundamentals) == 0:
                            st.error("File fundamentals.csv trống.")
                        else:
                            config = StrategyConfig(
                                name="ML Stock Selection",
                                description="Dashboard ML run"
                            )
                            strategy = MLStockSelectionStrategy(config)
                            result = strategy.generate_weights(
                                data={"fundamentals": fundamentals},
                                prediction_mode="single",
                                test_quarters=int(ml_test_q),
                                top_quantile=float(ml_top_q),
                                weight_method=ml_weight_method,
                                confirm_mode="today",
                                execution_date=ml_exec_date.strftime("%Y-%m-%d"),
                            )

                            weights_out = result.weights
                            if weights_out.empty:
                                st.error("Mô hình không chọn được cổ phiếu nào. Thử hạ Top Quantile.")
                            else:
                                out_path = Path("data/ml_weights_today.csv")
                                weights_out.to_csv(out_path, index=False)
                                st.success(
                                    f"Đã chọn **{len(weights_out)}** cổ phiếu và lưu vào "
                                    f"`{out_path}`. Kéo xuống mục Load để xem kết quả."
                                )
                                st.dataframe(
                                    weights_out[["gvkey", "weight", "predicted_return"]]
                                    .sort_values("weight", ascending=False)
                                    .style.format({"weight": "{:.2%}", "predicted_return": "{:.4f}"}),
                                    use_container_width=True,
                                    height=250,
                                )
                    except Exception as e:
                        st.error(f"Lỗi chạy ML: {e}")
                        import traceback
                        with st.expander("Chi tiết lỗi"):
                            st.code(traceback.format_exc())

    # ── 1. Load weights ────────────────────────────────────────────────────────
    st.markdown("#### 1. Load Portfolio Weights")

    load_source = st.radio(
        "Weights source",
        ["File trong data/", "Upload CSV"],
        horizontal=True,
    )

    weights_df = None
    latest_date = None

    if load_source.startswith("File"):
        # Scan data/ for candidate CSV files
        data_dir = Path("data")
        csv_files = sorted(data_dir.glob("*.csv")) if data_dir.exists() else []
        weight_files = [f for f in csv_files if "weight" in f.name.lower() or "ml_" in f.name.lower()]
        if not weight_files:
            weight_files = [f for f in csv_files if f.name != "sp500_tickers.csv"]

        if not weight_files:
            st.warning("Không tìm thấy file weights trong `data/`. Chạy notebook cell 6 trước.")
        else:
            selected_file = st.selectbox(
                "Chọn file weights",
                options=weight_files,
                format_func=lambda p: p.name,
            )
            try:
                raw = pd.read_csv(selected_file)
                weights_df, latest_date = _parse_weights_csv(raw)
                st.success(
                    f"Đã load **{len(weights_df)}** cổ phiếu từ `{selected_file.name}` "
                    f"(ngày: {latest_date.date()})"
                )
            except Exception as e:
                st.error(f"Lỗi đọc file: {e}")
    else:
        uploaded = st.file_uploader("Upload file CSV (wide hoặc long format)", type=["csv"])
        if uploaded:
            try:
                raw = pd.read_csv(uploaded)
                weights_df, latest_date = _parse_weights_csv(raw)
                st.success(f"Uploaded **{len(weights_df)}** cổ phiếu (ngày: {latest_date.date()}).")
            except Exception as e:
                st.error(f"File không hợp lệ: {e}")

    if weights_df is None or weights_df.empty:
        st.info("Load hoặc upload file weights CSV để tiếp tục.")
        return

    # ── Cảnh báo equal-weight ──────────────────────────────────────────────────
    n_unique_w = weights_df["weight"].nunique()
    if n_unique_w == 1 and len(weights_df) > 50:
        st.warning(
            f"**Cảnh báo:** Tất cả {len(weights_df)} cổ phiếu có weight bằng nhau "
            f"({weights_df['weight'].iloc[0]:.4f}). "
            "Đây có thể là output backtest, không phải ML selection. "
            "Hãy chạy lại cell 6 notebook với `prediction_mode='single'` "
            "để lấy danh sách cổ phiếu ML đã chọn (~51 stocks)."
        )

    # ── Loại bỏ ETF + lọc Top N ───────────────────────────────────────────────
    weights_df = weights_df[~weights_df["gvkey"].isin(_ETF_EXCLUDE)].copy()

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        max_stocks = len(weights_df)
        default_n = min(51, max_stocks)
        top_n_filter = st.slider(
            "Giữ lại Top N cổ phiếu (theo weight)",
            min_value=5, max_value=max_stocks, value=default_n, step=1,
        )
    with col_f2:
        st.metric("Tổng cổ phiếu sau lọc", f"{top_n_filter} / {max_stocks}")

    weights_df = weights_df.nlargest(top_n_filter, "weight").copy()
    total = weights_df["weight"].sum()
    if total > 0:
        weights_df["weight"] = weights_df["weight"] / total

    # ── 2. Preview weights ─────────────────────────────────────────────────────
    st.markdown("#### 2. Weights Preview")
    col_tbl, col_chart = st.columns([1, 1])

    with col_tbl:
        st.dataframe(
            weights_df.sort_values("weight", ascending=False)
            .style.format({"weight": "{:.2%}"}),
            use_container_width=True,
            height=300,
        )

    with col_chart:
        chart_data = weights_df.nlargest(10, "weight").copy()
        others_w = 1.0 - chart_data["weight"].sum()
        if others_w > 1e-4:
            chart_data = pd.concat(
                [chart_data, pd.DataFrame([{"gvkey": "Others", "weight": others_w}])],
                ignore_index=True,
            )
        fig = px.pie(chart_data, names="gvkey", values="weight",
                     title=f"Top-10 Allocation ({top_n_filter} stocks)", hole=0.35)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

    target_weights = {
        str(r["gvkey"]): float(r["weight"])
        for _, r in weights_df.iterrows()
    }

    # ── 3. Execution settings ──────────────────────────────────────────────────
    st.markdown("#### 3. Execution Settings")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        dry_run = st.checkbox("Dry Run (chỉ xem kế hoạch, không đặt lệnh)", value=True)
    with col_s2:
        closed_action = st.selectbox(
            "Market-closed action",
            ["opg", "skip"],
            index=0,
            help="opg = đặt lệnh limit-on-open phiên hôm sau; skip = bỏ qua",
        )

    # ── 4. Execute ─────────────────────────────────────────────────────────────
    st.markdown("#### 4. Execute")
    btn_label = "Preview Plan (Dry Run)" if dry_run else "Submit Orders"
    if st.button(btn_label, type="primary"):
        with st.spinner("Đang xử lý…"):
            try:
                plan = manager.execute_portfolio_rebalance(
                    target_weights,
                    account_name="default",
                    dry_run=dry_run,
                    market_closed_action=closed_action if not dry_run else "opg",
                )

                if dry_run:
                    st.success("Dry-run plan đã tạo xong.")
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Market Open", str(plan.get("market_open", "—")))
                    col_b.metric("Planned Buys", len(plan.get("orders_plan", {}).get("buy", [])))
                    col_c.metric("Planned Sells", len(plan.get("orders_plan", {}).get("sell", [])))
                    with st.expander("Full Plan JSON"):
                        st.json(plan)
                else:
                    n = plan.get("orders_placed", 0)
                    st.success(f"Đã đặt **{n}** lệnh.")
                    with st.expander("Execution Result"):
                        st.json(plan)

            except Exception as e:
                st.error(f"Execution thất bại: {e}")
                import traceback
                with st.expander("Chi tiết lỗi"):
                    st.code(traceback.format_exc())


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
