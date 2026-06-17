import yfinance as yf
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import mlflow
import mlflow.prophet
import warnings
warnings.filterwarnings("ignore")

# ── Define companies ───────────────────────────────────────────────────
companies = {
    "AAPL": "Apple",
    "JPM":  "JPMorgan",
    "GS":   "Goldman Sachs",
    "V":    "Visa",
    "MA":   "Mastercard"
}

# ── Set MLflow experiment ──────────────────────────────────────────────
# All 5 runs will be grouped under one experiment
mlflow.set_experiment("stocksense_forecasting")

print("Training and tracking all 5 stocks with MLflow...")
print("=" * 55)

for ticker, name in companies.items():
    print(f"\nProcessing {name} ({ticker})...")

    # ── Download and prepare data ──────────────────────────────────────
    raw = yf.download(tickers=ticker, period="2y", 
                      interval="1d", progress=False)
    raw.columns = raw.columns.get_level_values(0)
    df = raw[["Close"]].reset_index()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)

    # ── Train/test split ───────────────────────────────────────────────
    split_index = int(len(df) * 0.8)
    train = df.iloc[:split_index]
    test  = df.iloc[split_index:]

    # ── Start MLflow run for this stock ───────────────────────────────
    # Each stock gets its own named run inside the experiment
    with mlflow.start_run(run_name=f"prophet_{ticker}"):

        # Log parameters
        params = {
            "ticker": ticker,
            "company": name,
            "train_size": len(train),
            "test_size": len(test),
            "changepoint_prior_scale": 0.05,
            "yearly_seasonality": True,
            "weekly_seasonality": True
        }
        mlflow.log_params(params)

        # Train model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.fit(train)

        # Predict on test period
        future = model.make_future_dataframe(periods=len(test))
        forecast = model.predict(future)

        # Evaluate
        forecast_test = forecast[
            forecast["ds"] >= test["ds"].min()
        ].copy()
        merged = pd.merge(
            test[["ds", "y"]].reset_index(drop=True),
            forecast_test[["ds", "yhat"]].reset_index(drop=True),
            on="ds"
        )

        mae  = mean_absolute_error(merged["y"], merged["yhat"])
        rmse = np.sqrt(mean_squared_error(merged["y"], merged["yhat"]))
        mape = (abs((merged["y"] - merged["yhat"]) / 
                    merged["y"]) * 100).mean()

        # Log metrics
        mlflow.log_metric("mae", round(mae, 2))
        mlflow.log_metric("rmse", round(rmse, 2))
        mlflow.log_metric("mape", round(mape, 2))

        print(f"  MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | MAPE: {mape:.1f}%")

print()
print("=" * 55)
print("All runs logged to MLflow!")
print("View results at: http://127.0.0.1:5000")
print("Run: mlflow ui --workers 1")