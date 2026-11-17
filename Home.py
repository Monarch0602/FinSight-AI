
import streamlit as st

st.set_page_config(
    page_title="FinSight AI | Financial Intelligence",
    page_icon="📈",
    layout="wide"
)

# HEADER
st.title("📈 FinSight AI")
st.caption(
    "Intelligent Financial Market Analysis "
    "| Quantitative Research | Portfolio Risk"
)

st.divider()

# HERO SECTION
st.markdown(
    """
    ## Welcome to FinSight AI

    A machine learning-powered financial intelligence
    platform designed to analyze market trends,
    financial news, trading strategies and portfolio risk.
    """
)

st.info(
    "Use the navigation menu in the sidebar to explore "
    "the different analytical modules."
)

# MODULE OVERVIEW
st.subheader("Platform Modules")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Market Intelligence")
    st.write(
        "Analyze stock prices, technical indicators, "
        "financial news sentiment and ML-based market signals."
    )

    st.markdown("### 📈 Strategy Backtesting")
    st.write(
        "Evaluate EMA crossover strategies, "
        "risk-adjusted returns and historical performance."
    )

with col2:
    st.markdown("### 💼 Portfolio Analytics")
    st.write(
        "Analyze asset allocation, volatility, "
        "correlation, VaR and CVaR."
    )

    st.markdown("### 🧠 AI & Machine Learning")
    st.write(
        "Explore FinBERT NLP sentiment analysis "
        "and Random Forest classification."
    )

st.divider()

# PROJECT INFORMATION
st.subheader("Technology Stack")

st.markdown(
    """
    - **Programming:** Python
    - **Dashboard:** Streamlit
    - **Machine Learning:** Scikit-learn, Random Forest
    - **NLP:** FinBERT, Transformers, PyTorch
    - **Financial Data:** Yahoo Finance
    - **Technical Analysis:** TA library
    - **Visualization:** Plotly
    - **Data Processing:** Pandas, NumPy
    """
)

st.divider()

st.caption(
    "FinSight AI | Financial analytics and educational research "
    "| Not investment advice."
)
