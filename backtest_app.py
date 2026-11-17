
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from ta.trend import EMAIndicator


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="FinSight AI | Backtesting",
    page_icon="📊",
    layout="wide"
)

st.title("📊 FinSight AI — Strategy Backtesting")
st.caption(
    "Historical trading strategy simulation "
    "& quantitative performance evaluation"
)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Backtesting Configuration")

ticker = st.sidebar.text_input(
    "Stock Symbol",
    value="AAPL"
).strip().upper()

period = st.sidebar.selectbox(
    "Historical Period",
    ["6mo", "1y", "2y", "5y", "10y"],
    index=2
)

transaction_cost = st.sidebar.slider(
    "Transaction Cost (%) per trade",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.05
)


# =====================================================
# FETCH DATA
# =====================================================

@st.cache_data(ttl=300)
def fetch_data(symbol, selected_period):

    data = yf.download(
        symbol,
        period=selected_period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data.dropna()


# =====================================================
# BACKTESTING ENGINE
# =====================================================

def run_backtest(data, cost_percent):

    df = data.copy()

    # Technical indicators
    df["EMA20"] = EMAIndicator(
        close=df["Close"],
        window=20
    ).ema_indicator()

    df["EMA50"] = EMAIndicator(
        close=df["Close"],
        window=50
    ).ema_indicator()

    # Daily market returns
    df["Market_Return"] = (
        df["Close"].pct_change()
    )

    # Trading signal:
    # 1 = Invested, 0 = Out of market
    df["Signal"] = (
        df["EMA20"] > df["EMA50"]
    ).astype(int)

    # Use previous day's signal to avoid look-ahead bias
    df["Position"] = df["Signal"].shift(1).fillna(0)

    # Position changes generate transaction costs
    df["Turnover"] = (
        df["Position"].diff().abs()
    )

    df["Turnover"] = df["Turnover"].fillna(
        abs(df["Position"].iloc[0])
    )

    # Transaction cost
    cost = cost_percent / 100

    df["Transaction_Cost"] = (
        df["Turnover"] * cost
    )

    # Strategy returns after costs
    df["Strategy_Return"] = (
        df["Position"] * df["Market_Return"]
        - df["Transaction_Cost"]
    )

    # Buy and hold benchmark
    df["Benchmark_Return"] = df["Market_Return"]

    # Equity curves
    df["Strategy_Equity"] = (
        1 + df["Strategy_Return"].fillna(0)
    ).cumprod()

    df["Benchmark_Equity"] = (
        1 + df["Benchmark_Return"].fillna(0)
    ).cumprod()

    # Drawdowns
    df["Strategy_Drawdown"] = (
        df["Strategy_Equity"] /
        df["Strategy_Equity"].cummax() - 1
    )

    df["Benchmark_Drawdown"] = (
        df["Benchmark_Equity"] /
        df["Benchmark_Equity"].cummax() - 1
    )

    return df


# =====================================================
# PERFORMANCE METRICS
# =====================================================

def calculate_metrics(returns, equity):

    returns = returns.dropna()

    if len(returns) == 0:
        return {}

    total_return = (equity.iloc[-1] - 1) * 100

    years = len(returns) / 252

    if equity.iloc[-1] > 0 and years > 0:
        annualized_return = (
            equity.iloc[-1] ** (1 / years) - 1
        ) * 100
    else:
        annualized_return = -100

    volatility = returns.std() * np.sqrt(252)

    if volatility > 0:
        sharpe_ratio = (
            returns.mean() * 252
        ) / volatility
    else:
        sharpe_ratio = 0

    drawdown = equity / equity.cummax() - 1

    max_drawdown = drawdown.min() * 100

    profitable_days = (
        (returns > 0).sum() / len(returns)
    ) * 100

    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown": max_drawdown,
        "Profitable Days": profitable_days
    }


# =====================================================
# MAIN DASHBOARD
# =====================================================

if not ticker:

    st.warning("Enter a stock symbol.")

else:

    try:

        with st.spinner("Fetching historical data..."):

            data = fetch_data(ticker, period)

        if data.empty:

            st.error("No market data found.")
            st.stop()

        if len(data) < 60:

            st.error(
                "At least 60 trading observations are required."
            )
            st.stop()

        result = run_backtest(
            data,
            transaction_cost
        )

        strategy_metrics = calculate_metrics(
            result["Strategy_Return"],
            result["Strategy_Equity"]
        )

        benchmark_metrics = calculate_metrics(
            result["Benchmark_Return"],
            result["Benchmark_Equity"]
        )

        # =================================================
        # PERFORMANCE OVERVIEW
        # =================================================

        st.header("Strategy Performance")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Strategy Return",
            f"{strategy_metrics['Total Return']:.2f}%"
        )

        c2.metric(
            "Benchmark Return",
            f"{benchmark_metrics['Total Return']:.2f}%"
        )

        c3.metric(
            "Sharpe Ratio",
            f"{strategy_metrics['Sharpe Ratio']:.2f}"
        )

        c4.metric(
            "Maximum Drawdown",
            f"{strategy_metrics['Maximum Drawdown']:.2f}%"
        )

        # =================================================
        # EQUITY CURVE
        # =================================================

        st.divider()

        st.subheader("Equity Curve Comparison")

        equity_fig = go.Figure()

        equity_fig.add_trace(
            go.Scatter(
                x=result.index,
                y=result["Strategy_Equity"],
                name="EMA Strategy",
                line=dict(width=2)
            )
        )

        equity_fig.add_trace(
            go.Scatter(
                x=result.index,
                y=result["Benchmark_Equity"],
                name="Buy & Hold",
                line=dict(width=2)
            )
        )

        equity_fig.update_layout(
            template="plotly_dark",
            height=450,
            yaxis_title="Growth of ₹1 / $1",
            hovermode="x unified"
        )

        st.plotly_chart(
            equity_fig,
            use_container_width=True
        )

        # =================================================
        # DRAWDOWN ANALYSIS
        # =================================================

        st.subheader("Drawdown Analysis")

        drawdown_fig = go.Figure()

        drawdown_fig.add_trace(
            go.Scatter(
                x=result.index,
                y=result["Strategy_Drawdown"] * 100,
                name="EMA Strategy"
            )
        )

        drawdown_fig.add_trace(
            go.Scatter(
                x=result.index,
                y=result["Benchmark_Drawdown"] * 100,
                name="Buy & Hold"
            )
        )

        drawdown_fig.update_layout(
            template="plotly_dark",
            height=350,
            yaxis_title="Drawdown (%)"
        )

        st.plotly_chart(
            drawdown_fig,
            use_container_width=True
        )

        # =================================================
        # METRICS COMPARISON
        # =================================================

        st.divider()

        st.subheader("Quantitative Performance Metrics")

        comparison = pd.DataFrame({
            "EMA Strategy": strategy_metrics,
            "Buy & Hold": benchmark_metrics
        })

        st.dataframe(
            comparison.round(3),
            use_container_width=True
        )

        # =================================================
        # TRADING SIGNALS
        # =================================================

        st.subheader("Historical Trading Signals")

        signal_fig = go.Figure()

        signal_fig.add_trace(
            go.Scatter(
                x=result.index,
                y=result["Close"],
                name="Stock Price"
            )
        )

        signal_fig.add_trace(
            go.Scatter(
                x=result.index,
                y=result["EMA20"],
                name="EMA 20"
            )
        )

        signal_fig.add_trace(
            go.Scatter(
                x=result.index,
                y=result["EMA50"],
                name="EMA 50"
            )
        )

        buy_points = result[
            result["Signal"].diff() == 1
        ]

        sell_points = result[
            result["Signal"].diff() == -1
        ]

        signal_fig.add_trace(
            go.Scatter(
                x=buy_points.index,
                y=buy_points["Close"],
                mode="markers",
                name="Buy Signal",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color="green"
                )
            )
        )

        signal_fig.add_trace(
            go.Scatter(
                x=sell_points.index,
                y=sell_points["Close"],
                mode="markers",
                name="Exit Signal",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color="red"
                )
            )
        )

        signal_fig.update_layout(
            template="plotly_dark",
            height=450
        )

        st.plotly_chart(
            signal_fig,
            use_container_width=True
        )

        # =================================================
        # ADDITIONAL STATISTICS
        # =================================================

        st.subheader("Additional Strategy Statistics")

        a, b, c = st.columns(3)

        a.metric(
            "Profitable Trading Days",
            f"{strategy_metrics['Profitable Days']:.2f}%"
        )

        b.metric(
            "Annualized Return",
            f"{strategy_metrics['Annualized Return']:.2f}%"
        )

        c.metric(
            "Total Position Changes",
            int(result["Turnover"].sum())
        )

        st.caption(
            "Profitable trading days are not the same as "
            "profitable completed trades."
        )

        # =================================================
        # DOWNLOAD RESULTS
        # =================================================

        st.divider()

        csv = result.to_csv().encode("utf-8")

        st.download_button(
            "Download Backtesting Results",
            data=csv,
            file_name=f"{ticker}_backtest.csv",
            mime="text/csv"
        )

        st.caption(
            "FinSight AI | Historical simulation only. "
            "Transaction costs are modeled; slippage, "
            "taxes and market impact are not included. "
            "Past performance does not guarantee future results."
        )

    except Exception as e:

        st.error(f"Backtesting error: {e}")
