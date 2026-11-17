
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="FinSight AI | Portfolio Risk",
    page_icon="💼",
    layout="wide"
)

st.title("💼 FinSight AI — Portfolio Risk Analytics")

st.caption(
    "Multi-Asset Portfolio Analysis, Risk Measurement "
    "& Asset Correlation"
)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Portfolio Configuration")

symbols_input = st.sidebar.text_area(
    "Stock Allocations",
    value="AAPL:40\nMSFT:35\nNVDA:25",
    help="Enter one stock per line, followed by allocation percentage."
)

period = st.sidebar.selectbox(
    "Historical Period",
    ["6mo", "1y", "2y", "5y"],
    index=2
)

investment = st.sidebar.number_input(
    "Investment Amount ($)",
    min_value=100.0,
    value=10000.0,
    step=1000.0
)

risk_free_rate = st.sidebar.number_input(
    "Annual Risk-Free Rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=4.0,
    step=0.25
)


# =====================================================
# PARSE PORTFOLIO
# =====================================================

def parse_allocations(text):

    allocations = {}

    for line in text.strip().splitlines():

        if not line.strip():
            continue

        try:

            symbol, weight = line.split(":")

            symbol = symbol.strip().upper()
            weight = float(weight.strip())

            if not symbol or weight <= 0:
                raise ValueError("Invalid symbol or weight.")

            allocations[symbol] = weight

        except ValueError as error:

            raise ValueError(
                f"Invalid allocation line: {line}. "
                "Use the format AAPL:40"
            ) from error

    if len(allocations) < 2:
        raise ValueError(
            "Enter at least two different stocks."
        )

    total_weight = sum(allocations.values())

    if not np.isclose(total_weight, 100, atol=0.01):
        raise ValueError(
            f"Allocations must total 100%. "
            f"Your total is {total_weight:.2f}%."
        )

    return {
        symbol: weight / 100
        for symbol, weight in allocations.items()
    }


# =====================================================
# DOWNLOAD STOCK DATA
# =====================================================

@st.cache_data(ttl=300)
def download_prices(symbols, selected_period):

    prices = {}

    failed = []

    for symbol in symbols:

        try:

            data = yf.download(
                symbol,
                period=selected_period,
                interval="1d",
                auto_adjust=True,
                progress=False
            )

            if data.empty:
                failed.append(symbol)
                continue

            if isinstance(data.columns, pd.MultiIndex):

                close = data["Close"].iloc[:, 0]

            else:

                close = data["Close"]

            prices[symbol] = close

        except Exception:

            failed.append(symbol)

    if not prices:
        return pd.DataFrame(), failed

    price_df = pd.DataFrame(prices)

    return price_df, failed


# =====================================================
# PORTFOLIO CALCULATIONS
# =====================================================

def calculate_portfolio_metrics(returns, weights, rate):

    weights_array = np.array(
        [weights[column] for column in returns.columns]
    )

    portfolio_returns = (
        returns * weights_array
    ).sum(axis=1)

    mean_daily_return = portfolio_returns.mean()

    annualized_return = (
        (1 + mean_daily_return) ** 252 - 1
    ) * 100

    volatility = (
        portfolio_returns.std() * np.sqrt(252) * 100
    )

    annual_rf = rate / 100

    daily_rf = (1 + annual_rf) ** (1 / 252) - 1

    if portfolio_returns.std() > 0:

        sharpe = (
            (mean_daily_return - daily_rf)
            / portfolio_returns.std()
        ) * np.sqrt(252)

    else:

        sharpe = 0

    # Historical one-day Value at Risk
    var_95 = max(
        0,
        -np.percentile(portfolio_returns, 5) * 100
    )

    # Conditional Value at Risk
    losses = portfolio_returns[
        portfolio_returns <= np.percentile(
            portfolio_returns, 5
        )
    ]

    cvar_95 = max(
        0,
        -losses.mean() * 100
    )

    # Portfolio equity curve
    equity = (
        1 + portfolio_returns
    ).cumprod()

    drawdown = (
        equity / equity.cummax() - 1
    )

    max_drawdown = drawdown.min() * 100

    total_return = (
        equity.iloc[-1] - 1
    ) * 100

    return {
        "returns": portfolio_returns,
        "equity": equity,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "max_drawdown": max_drawdown,
        "total_return": total_return
    }


# =====================================================
# MAIN APPLICATION
# =====================================================

try:

    allocations = parse_allocations(
        symbols_input
    )

    symbols = list(allocations.keys())

    st.header("Portfolio Overview")

    st.write(
        "Selected assets:",
        ", ".join(symbols)
    )

    with st.spinner(
        "Downloading historical asset prices..."
    ):

        prices, failed = download_prices(
            symbols,
            period
        )

    if failed:

        st.warning(
            "Unable to retrieve data for: "
            + ", ".join(failed)
        )

        st.stop()

    if prices.empty:

        st.error("No historical data available.")
        st.stop()

    # Align dates and remove incomplete observations
    prices = prices.ffill().dropna()

    if len(prices) < 60:

        st.error(
            "At least 60 common trading observations are required."
        )

        st.stop()

    # Daily asset returns
    asset_returns = prices.pct_change().dropna()

    # Portfolio performance
    metrics = calculate_portfolio_metrics(
        asset_returns,
        allocations,
        risk_free_rate
    )

    portfolio_returns = metrics["returns"]

    # =================================================
    # PORTFOLIO METRICS
    # =================================================

    st.subheader("Portfolio Risk Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Return",
        f"{metrics['total_return']:.2f}%"
    )

    c2.metric(
        "Annualized Volatility",
        f"{metrics['volatility']:.2f}%"
    )

    c3.metric(
        "Sharpe Ratio",
        f"{metrics['sharpe']:.2f}"
    )

    c4.metric(
        "Maximum Drawdown",
        f"{metrics['max_drawdown']:.2f}%"
    )

    c5, c6, c7 = st.columns(3)

    c5.metric(
        "Daily VaR (95%)",
        f"{metrics['var_95']:.2f}%"
    )

    c6.metric(
        "Daily CVaR (95%)",
        f"{metrics['cvar_95']:.2f}%"
    )

    c7.metric(
        "Annualized Return",
        f"{metrics['annualized_return']:.2f}%"
    )

    # =================================================
    # PORTFOLIO ALLOCATION
    # =================================================

    st.divider()

    st.subheader("Asset Allocation")

    allocation_df = pd.DataFrame({
        "Stock": list(allocations.keys()),
        "Weight": [
            weight * 100
            for weight in allocations.values()
        ]
    })

    fig_allocation = px.pie(
        allocation_df,
        names="Stock",
        values="Weight",
        hole=0.55,
        title="Portfolio Allocation"
    )

    fig_allocation.update_layout(
        template="plotly_dark",
        height=400
    )

    st.plotly_chart(
        fig_allocation,
        use_container_width=True
    )

    allocation_df["Invested Amount"] = (
        allocation_df["Weight"] / 100
    ) * investment

    st.dataframe(
        allocation_df.round(2),
        use_container_width=True
    )

    # =================================================
    # PORTFOLIO GROWTH
    # =================================================

    st.divider()

    st.subheader("Portfolio Growth")

    growth = metrics["equity"] * investment

    growth_fig = go.Figure()

    growth_fig.add_trace(
        go.Scatter(
            x=growth.index,
            y=growth.values,
            name="Portfolio Value",
            fill="tozeroy"
        )
    )

    growth_fig.add_hline(
        y=investment,
        line_dash="dash",
        annotation_text="Initial Investment"
    )

    growth_fig.update_layout(
        template="plotly_dark",
        height=400,
        yaxis_title="Portfolio Value ($)"
    )

    st.plotly_chart(
        growth_fig,
        use_container_width=True
    )

    # =================================================
    # CORRELATION MATRIX
    # =================================================

    st.divider()

    st.subheader("Asset Correlation Matrix")

    correlation = asset_returns.corr()

    correlation_fig = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Historical Asset Correlation"
    )

    correlation_fig.update_layout(
        template="plotly_dark",
        height=450
    )

    st.plotly_chart(
        correlation_fig,
        use_container_width=True
    )

    # =================================================
    # DRAWDOWN ANALYSIS
    # =================================================

    st.subheader("Portfolio Drawdown")

    drawdown = (
        metrics["equity"]
        / metrics["equity"].cummax() - 1
    ) * 100

    drawdown_fig = go.Figure()

    drawdown_fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            name="Portfolio Drawdown",
            fill="tozeroy"
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
    # RISK INTERPRETATION
    # =================================================

    st.divider()

    st.subheader("Portfolio Risk Interpretation")

    if metrics["volatility"] > 30:

        st.warning(
            "The portfolio has relatively high historical volatility."
        )

    elif metrics["volatility"] > 15:

        st.info(
            "The portfolio has moderate historical volatility."
        )

    else:

        st.success(
            "The portfolio has relatively low historical volatility."
        )

    if metrics["sharpe"] > 1:

        st.success(
            "The historical Sharpe ratio is above 1."
        )

    elif metrics["sharpe"] > 0:

        st.info(
            "The historical risk-adjusted return is positive."
        )

    else:

        st.warning(
            "The historical risk-adjusted return is non-positive."
        )

    st.caption(
        "VaR and CVaR are historical one-day estimates at the 95% "
        "confidence level. Annualized return uses compounded "
        "average daily returns. Historical metrics are not forecasts."
    )

    # =================================================
    # DOWNLOAD REPORT
    # =================================================

    st.divider()

    report = prices.copy()

    report["Portfolio_Return"] = portfolio_returns

    csv = report.to_csv().encode("utf-8")

    st.download_button(
        "Download Portfolio Analysis CSV",
        data=csv,
        file_name="FinSight_Portfolio_Analysis.csv",
        mime="text/csv"
    )

except ValueError as error:

    st.error(str(error))

except Exception as error:

    st.error(
        f"Portfolio analysis failed: {error}"
    )
