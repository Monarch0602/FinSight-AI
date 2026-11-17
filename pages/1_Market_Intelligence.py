
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator

from sentiment import analyze_sentiment
from news_data import fetch_financial_news
from market_model import train_market_model


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="FinSight AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 FinSight AI")
st.caption(
    "AI-Powered Financial Market Intelligence "
    "& Risk Analytics Platform"
)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Market Configuration")

ticker = st.sidebar.text_input(
    "Stock Symbol",
    value="AAPL"
).strip().upper()

period = st.sidebar.selectbox(
    "Analysis Period",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=3
)

st.sidebar.info(
    "Data source: Yahoo Finance"
)


# =====================================================
# FETCH STOCK DATA
# =====================================================

@st.cache_data(ttl=300)
def fetch_stock_data(symbol, selected_period):

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
# TECHNICAL INDICATORS
# =====================================================

def calculate_indicators(data):

    data = data.copy()

    close = data["Close"]

    data["EMA20"] = EMAIndicator(
        close=close,
        window=20
    ).ema_indicator()

    data["EMA50"] = EMAIndicator(
        close=close,
        window=50
    ).ema_indicator()

    data["RSI"] = RSIIndicator(
        close=close,
        window=14
    ).rsi()

    macd = MACD(close=close)

    data["MACD"] = macd.macd()
    data["Signal"] = macd.macd_signal()
    data["MACD_Hist"] = macd.macd_diff()

    return data


# =====================================================
# RISK ANALYSIS
# =====================================================

def calculate_risk(data):

    returns = data["Close"].pct_change().dropna()

    volatility = (
        returns.std() * np.sqrt(252) * 100
    )

    cumulative_returns = (1 + returns).cumprod()

    running_max = cumulative_returns.cummax()

    drawdown = (
        cumulative_returns / running_max - 1
    )

    max_drawdown = drawdown.min() * 100

    total_return = (
        data["Close"].iloc[-1] /
        data["Close"].iloc[0] - 1
    ) * 100

    return volatility, max_drawdown, total_return


# =====================================================
# MAIN APPLICATION
# =====================================================

if not ticker:

    st.warning("Enter a stock symbol.")

else:

    with st.spinner("Loading financial market data..."):

        try:

            df = fetch_stock_data(ticker, period)

            if df.empty:
                st.error(
                    "No stock data found. Check the ticker symbol."
                )
                st.stop()

            if len(df) < 2:
                st.error("Not enough historical observations.")
                st.stop()

            df = calculate_indicators(df)

            volatility, max_drawdown, total_return = (
                calculate_risk(df)
            )

            latest_price = float(df["Close"].iloc[-1])

            previous_price = float(df["Close"].iloc[-2])

            daily_change = (
                latest_price / previous_price - 1
            ) * 100

            latest_rsi = float(df["RSI"].iloc[-1])

            # =================================================
            # MARKET OVERVIEW
            # =================================================

            st.header(f"Market Overview: {ticker}")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Latest Price",
                f"${latest_price:,.2f}",
                f"{daily_change:.2f}%"
            )

            c2.metric(
                "RSI (14)",
                f"{latest_rsi:.2f}"
            )

            c3.metric(
                "Annualized Volatility",
                f"{volatility:.2f}%"
            )

            c4.metric(
                "Maximum Drawdown",
                f"{max_drawdown:.2f}%"
            )

            st.metric(
                "Total Period Return",
                f"{total_return:.2f}%"
            )

            # =================================================
            # PRICE CHART
            # =================================================

            st.divider()

            st.subheader(
                "📊 Stock Price & Technical Indicators"
            )

            fig = go.Figure()

            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name="Stock Price"
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["EMA20"],
                    name="EMA 20"
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["EMA50"],
                    name="EMA 50"
                )
            )

            fig.update_layout(
                height=500,
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                hovermode="x unified"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # =================================================
            # RSI
            # =================================================

            st.subheader("Relative Strength Index")

            rsi_fig = go.Figure()

            rsi_fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["RSI"],
                    name="RSI"
                )
            )

            rsi_fig.add_hline(
                y=70,
                line_dash="dash",
                line_color="red"
            )

            rsi_fig.add_hline(
                y=30,
                line_dash="dash",
                line_color="green"
            )

            rsi_fig.update_layout(
                height=300,
                template="plotly_dark",
                yaxis_range=[0, 100]
            )

            st.plotly_chart(
                rsi_fig,
                use_container_width=True
            )

            # =================================================
            # MACD
            # =================================================

            st.subheader("MACD Analysis")

            macd_fig = go.Figure()

            macd_fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["MACD"],
                    name="MACD"
                )
            )

            macd_fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["Signal"],
                    name="Signal"
                )
            )

            macd_fig.update_layout(
                height=300,
                template="plotly_dark"
            )

            st.plotly_chart(
                macd_fig,
                use_container_width=True
            )

            # =================================================
            # TECHNICAL INTERPRETATION
            # =================================================

            st.divider()

            st.subheader(
                "Technical Market Interpretation"
            )

            if latest_rsi > 70:

                st.warning(
                    "RSI indicates potentially overbought conditions."
                )

            elif latest_rsi < 30:

                st.success(
                    "RSI indicates potentially oversold conditions."
                )

            else:

                st.info(
                    "RSI is currently in the neutral range."
                )

            if df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1]:

                st.success(
                    "EMA 20 is above EMA 50: positive moving-average alignment."
                )

            else:

                st.warning(
                    "EMA 20 is below EMA 50: negative moving-average alignment."
                )

            # =================================================
            # FINBERT SENTIMENT ANALYSIS
            # =================================================

            st.divider()

            st.header(
                "🧠 AI Financial News Sentiment"
            )

            try:

                with st.spinner(
                    "Fetching financial news..."
                ):

                    headlines = fetch_financial_news(ticker)

                if headlines:

                    with st.spinner(
                        "FinBERT analyzing news..."
                    ):

                        results = analyze_sentiment(headlines)

                    if results:

                        positive = sum(
                            r["sentiment"] == "Positive"
                            for r in results
                        )

                        negative = sum(
                            r["sentiment"] == "Negative"
                            for r in results
                        )

                        neutral = sum(
                            r["sentiment"] == "Neutral"
                            for r in results
                        )

                        s1, s2, s3 = st.columns(3)

                        s1.metric("Positive News", positive)
                        s2.metric("Negative News", negative)
                        s3.metric("Neutral News", neutral)

                        sentiment_fig = go.Figure(
                            data=[
                                go.Pie(
                                    labels=[
                                        "Positive",
                                        "Negative",
                                        "Neutral"
                                    ],
                                    values=[
                                        positive,
                                        negative,
                                        neutral
                                    ],
                                    hole=0.5
                                )
                            ]
                        )

                        sentiment_fig.update_layout(
                            title="News Sentiment Distribution",
                            template="plotly_dark",
                            height=350
                        )

                        st.plotly_chart(
                            sentiment_fig,
                            use_container_width=True
                        )

                        st.subheader(
                            "Individual News Analysis"
                        )

                        for result in results:

                            sentiment = result["sentiment"]

                            icon = {
                                "Positive": "🟢",
                                "Negative": "🔴",
                                "Neutral": "🟡"
                            }.get(sentiment, "⚪")

                            with st.expander(
                                f"{icon} {result['headline']}"
                            ):

                                st.write(
                                    f"Sentiment: {sentiment}"
                                )

                                st.write(
                                    f"Confidence: "
                                    f"{result['confidence']:.2%}"
                                )

                                st.progress(
                                    float(result["confidence"])
                                )

                else:

                    st.info(
                        "No financial news available for this stock."
                    )

            except Exception as e:

                st.warning(
                    f"Sentiment analysis unavailable: {e}"
                )

            # =================================================
            # RANDOM FOREST MACHINE LEARNING
            # =================================================

            st.divider()

            st.header(
                "🤖 Machine Learning Market Analysis"
            )

            try:

                with st.spinner(
                    "Training Random Forest model..."
                ):

                    ml_results = train_market_model(df)

                evaluation = ml_results["evaluation"]

                st.subheader(
                    "Model Evaluation"
                )

                m1, m2, m3, m4 = st.columns(4)

                m1.metric(
                    "Accuracy",
                    f"{evaluation['accuracy']:.2%}"
                )

                m2.metric(
                    "Precision",
                    f"{evaluation['precision']:.2%}"
                )

                m3.metric(
                    "Recall",
                    f"{evaluation['recall']:.2%}"
                )

                m4.metric(
                    "F1 Score",
                    f"{evaluation['f1']:.2%}"
                )

                # Prediction probability

                st.subheader(
                    "Next Trading Day Probability"
                )

                probability = ml_results[
                    "upward_probability"
                ]

                p1, p2 = st.columns(2)

                p1.metric(
                    "Upward Movement Probability",
                    f"{probability:.2%}"
                )

                p1.progress(probability)

                p2.metric(
                    "Downward Movement Probability",
                    f"{1 - probability:.2%}"
                )

                p2.progress(1 - probability)

                st.caption(
                    "These are model-estimated probabilities, "
                    "not calibrated forecasts or guaranteed outcomes."
                )

                # Confusion Matrix

                st.subheader("Confusion Matrix")

                cm_df = pd.DataFrame(
                    evaluation["confusion_matrix"],
                    index=[
                        "Actual Down",
                        "Actual Up"
                    ],
                    columns=[
                        "Predicted Down",
                        "Predicted Up"
                    ]
                )

                st.dataframe(
                    cm_df,
                    use_container_width=True
                )

                # Feature Importance

                st.subheader(
                    "Model Feature Importance"
                )

                importance = ml_results[
                    "feature_importance"
                ]

                importance_fig = go.Figure(
                    go.Bar(
                        x=importance["Importance"],
                        y=importance["Feature"],
                        orientation="h"
                    )
                )

                importance_fig.update_layout(
                    template="plotly_dark",
                    height=400,
                    yaxis=dict(autorange="reversed")
                )

                st.plotly_chart(
                    importance_fig,
                    use_container_width=True
                )

            except Exception as e:

                st.warning(
                    f"Machine learning module unavailable: {e}"
                )

            # =================================================
            # HISTORICAL DATA
            # =================================================

            st.divider()

            with st.expander(
                "View Historical Market Data"
            ):

                st.dataframe(
                    df.tail(30).sort_index(
                        ascending=False
                    ),
                    use_container_width=True
                )

            st.caption(
                "FinSight AI | Data: Yahoo Finance | "
                "For educational and research purposes only. "
                "Not investment advice."
            )

        except Exception as e:

            st.error(
                f"Application error: {e}"
            )
