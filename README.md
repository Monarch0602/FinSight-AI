# FinSight AI – Financial Market Intelligence & Portfolio Risk Analytics

An AI-powered financial analytics platform that combines machine learning, financial news sentiment analysis, technical indicators, strategy backtesting, and portfolio risk assessment in one interactive dashboard.

## Overview

FinSight AI is a multi-page financial intelligence application designed to help users explore historical market behaviour, analyse financial news sentiment, evaluate market direction predictions, backtest trading strategies, and assess portfolio risk.

The application integrates financial data processing, machine learning, natural language processing, and quantitative risk metrics through an interactive Streamlit interface.

## Key Features

### 1. Market Intelligence
- Historical stock market data using Yahoo Finance.
- Technical indicators including RSI, MACD, EMA 20, and EMA 50.
- Historical volatility, daily returns, and maximum drawdown.
- Financial news headline collection.
- Financial sentiment classification using FinBERT.
- Random Forest model for next-day market direction classification.
- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix.

### 2. Trading Strategy Backtesting
- EMA 20 and EMA 50 crossover strategy.
- Historical strategy simulation.
- Buy-and-hold benchmark comparison.
- Transaction cost consideration.
- Equity curve and drawdown visualisation.
- Sharpe ratio and annualised return analysis.
- CSV export of backtesting results.

### 3. Portfolio Risk Analytics
- Multi-stock portfolio allocation.
- Historical portfolio performance analysis.
- Portfolio return and volatility.
- Sharpe ratio.
- Historical Value at Risk (VaR) and Conditional Value at Risk (CVaR).
- Maximum drawdown.
- Asset correlation analysis.
- Portfolio growth visualisation.
- CSV export.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend / Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Financial Data | Yahoo Finance, yfinance |
| Machine Learning | Scikit-learn |
| NLP / Sentiment Analysis | Hugging Face Transformers, FinBERT |
| Deep Learning Framework | PyTorch |
| Visualisation | Plotly |
| Technical Analysis | Technical indicators and custom calculations |

## Machine Learning Methodology

### Financial Sentiment Analysis – FinBERT

The application uses the pretrained `ProsusAI/finbert` model to classify financial news headlines into:

- Positive
- Negative
- Neutral

The sentiment output provides an additional perspective for analysing financial market information.

### Market Direction Prediction – Random Forest

A Random Forest Classifier is used to classify whether the next trading day's closing price will be higher than the current day's closing price.

The model uses engineered financial features such as:

- Daily returns
- RSI
- MACD
- MACD histogram
- EMA 20 gap
- EMA 50 gap
- Rolling volatility
- Volume change

The dataset is split chronologically into training and testing sets to reduce look-ahead bias. Model performance is evaluated using standard classification metrics.

## Project Structure

```text
FinSight-AI/
│
├── Home.py
├── app.py
├── market_model.py
├── sentiment.py
├── news_data.py
├── backtest_app.py
├── portfolio.py
│
├── pages/
│   ├── 1_Market_Intelligence.py
│   ├── 2_Backtesting.py
│   └── 3_Portfolio_Risk.py
│
├── .streamlit/
│   └── config.toml
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### Prerequisites

- Python 3.10 or later
- Git
- Internet connection for financial data and pretrained model downloads

### 1. Clone the repository

```bash
git clone https://github.com/Monarch0602/FinSight-AI.git
cd FinSight-AI
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run Home.py
```

The application will open in your browser.

## Performance Evaluation

The market prediction model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The trading strategy is evaluated using historical returns, annualised return, Sharpe ratio, drawdown, and comparison with a buy-and-hold benchmark.

Actual results depend on the selected stock, date range, available data, and model configuration. No fixed predictive accuracy or investment return is guaranteed.

## Limitations

- Historical market data does not guarantee future performance.
- Financial news availability depends on the data provider.
- FinBERT sentiment classification may not capture every financial context.
- Machine learning predictions are probabilistic and can be incorrect.
- Backtesting results do not guarantee live trading performance.
- Transaction costs and execution assumptions may differ from real-world conditions.

## Future Improvements

- Real-time market data integration.
- Advanced time-series forecasting models.
- More trading strategies and optimisation techniques.
- Enhanced portfolio optimisation.
- Additional financial news sources.
- Improved model explainability.

## Disclaimer

FinSight AI is an educational and analytical project. It is not a financial advisory service and does not provide investment recommendations. All predictions and historical analyses should be independently evaluated before making financial decisions.

## Author

**Monarch0602**

- GitHub: https://github.com/Monarch0602
- Project Repository: https://github.com/Monarch0602/FinSight-AI

---

*Built with Python, Machine Learning, NLP, and Streamlit.*
