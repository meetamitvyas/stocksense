# StockSense — AI-Powered Stock Price Forecasting

A time series forecasting system that predicts 30-day stock prices for 
major financial companies using Facebook Prophet, with component analysis 
and multi-stock comparison.

---

## What This Project Does

StockSense fetches 2 years of real stock price data for 5 major companies,
trains a Prophet forecasting model on each, predicts the next 30 days of 
prices, and produces professional charts showing trends, seasonality patterns,
and a side-by-side comparison of expected performance.

---

## Key Results

| Company | Current Price | 30-Day Forecast | Expected Change |
|---|---|---|---|
| Apple | $299.24 | $340.43 | +13.8% ↑ |
| Goldman Sachs | $1,090.67 | $1,150.11 | +5.5% ↑ |
| JPMorgan | $331.14 | $338.73 | +2.3% ↑ |
| Visa | $333.12 | $311.16 | -6.6% ↓ |
| Mastercard | $501.33 | $456.97 | -8.8% ↓ |

---

## What I Learned

**Time series forecasting is different from classification:**
- Rows are NOT independent — each price depends on previous prices
- Train/test split must respect chronological order — never shuffle
- Evaluation uses MAE, RMSE, MAPE — not accuracy or F1

**Prophet decomposes forecasts into components:**
- Trend — overall direction (bullish/bearish)
- Weekly seasonality — Tuesday-Wednesday peaks for stocks
- Yearly seasonality — January effect, summer weakness

**Model limitations discovered:**
- Prophet struggled with sudden directional changes (Apple's 2026 rally)
- Baseline MAPE of 11% — near acceptable range for financial forecasting
- Improvement: add external regressors (volume, VIX, news sentiment)

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| yfinance | Real-time stock data from Yahoo Finance |
| Prophet (Facebook) | Time series forecasting |
| Pandas | Data manipulation |
| Matplotlib | Charts and visualisations |
| Scikit-learn | Evaluation metrics (MAE, RMSE, MAPE) |

---

## Project Structure
stocksense/

├── 01_explore_stocks.py       # data exploration and visualisation

├── 02_prepare_data.py         # prepare data in Prophet format

├── 03_train_prophet.py        # train model and evaluate accuracy

├── 04_forecast_future.py      # 30-day future forecast

├── 05_components.py           # trend and seasonality breakdown

├── 06_multi_stock_forecast.py # compare all 5 companies

└── data/

└── apple_prophet.csv      # prepared dataset

---

## How to Run

```bash
git clone https://github.com/meetamitvyas/stocksense.git
cd stocksense
python -m venv venv
venv\Scripts\activate
pip install prophet pandas matplotlib yfinance scikit-learn
python 01_explore_stocks.py
python 02_prepare_data.py
python 03_train_prophet.py
python 04_forecast_future.py
python 05_components.py
python 06_multi_stock_forecast.py
```

---

## Author

**Amit Vyas** — Principal Data Engineer  
13+ years experience in data warehousing, analytics, and AI/ML  
[GitHub](https://github.com/meetamitvyas) | 
[LinkedIn](https://linkedin.com/in/meetamitvyas)