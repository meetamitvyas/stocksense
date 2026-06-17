import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# ── Load prepared data ─────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/apple_prophet.csv")
df["ds"] = pd.to_datetime(df["ds"])

print(f"Total rows: {len(df)}")
print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
print()

# ── Train/test split ───────────────────────────────────────────────────
# 80% for training, 20% for testing
# IMPORTANT: we split by position, not randomly
# Earlier dates go to train, later dates go to test
split_index = int(len(df) * 0.8)
train = df.iloc[:split_index]   # first 80%
test  = df.iloc[split_index:]   # last 20%

print(f"Training set: {len(train)} rows")
print(f"  From: {train['ds'].min()} to {train['ds'].max()}")
print()
print(f"Test set: {len(test)} rows")
print(f"  From: {test['ds'].min()} to {test['ds'].max()}")
print()

# ── Train Prophet model ────────────────────────────────────────────────
# What each parameter means:
# yearly_seasonality = look for patterns that repeat every year
# weekly_seasonality = look for patterns that repeat every week
# daily_seasonality  = look for patterns within a single day (we turn
#                      this off because daily stock data has no intraday pattern)
# changepoint_prior_scale = how flexible the trend can be
#   higher value = more flexible (follows data closely, risk of overfitting)
#   lower value  = smoother trend (more generalised)

print("Training Prophet model...")
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)

# .fit() trains the model on the training data only
model.fit(train)
print("Training complete!")
print()

# ── Make predictions on test period ───────────────────────────────────
# We create a future DataFrame covering the test period
# make_future_dataframe extends the training dates forward
future = model.make_future_dataframe(periods=len(test))
forecast = model.predict(future)

# ── What does the forecast contain? ───────────────────────────────────
print("Forecast columns:")
print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(5))
print()

# yhat       = predicted price (the main prediction)
# yhat_lower = lower bound of confidence interval
# yhat_upper = upper bound of confidence interval

# ── Plot the forecast ──────────────────────────────────────────────────
print("Plotting forecast...")

fig, ax = plt.subplots(figsize=(14, 6))

# Plot actual training data
ax.plot(train["ds"], train["y"],
        color="steelblue", linewidth=1.5, label="Training data (actual)")

# Plot actual test data
ax.plot(test["ds"], test["y"],
        color="green", linewidth=1.5, label="Test data (actual)")

# Plot Prophet's predictions for the test period
forecast_test = forecast[forecast["ds"] >= test["ds"].min()]
ax.plot(forecast_test["ds"], forecast_test["yhat"],
        color="red", linewidth=1.5, linestyle="--", label="Prophet forecast")

# Plot confidence interval
ax.fill_between(
    forecast_test["ds"],
    forecast_test["yhat_lower"],
    forecast_test["yhat_upper"],
    alpha=0.2, color="red", label="Confidence interval"
)

ax.set_title("Apple Stock Price — Prophet Forecast vs Actual")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("forecast_vs_actual.png")
plt.show()
print("Chart saved as forecast_vs_actual.png")

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# ── Evaluate forecast accuracy ─────────────────────────────────────────
# Get Prophet's predictions for the test period only
forecast_test = forecast[forecast["ds"] >= test["ds"].min()].copy()
forecast_test = forecast_test.reset_index(drop=True)
test_reset = test.reset_index(drop=True)

# Align by matching dates
merged = pd.merge(
    test_reset[["ds", "y"]],
    forecast_test[["ds", "yhat"]],
    on="ds"
)

actual = merged["y"]
predicted = merged["yhat"]

# ── Calculate metrics ──────────────────────────────────────────────────
mae  = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
mape = (abs((actual - predicted) / actual) * 100).mean()

print()
print("=" * 40)
print("MODEL EVALUATION METRICS")
print("=" * 40)
print(f"MAE  (Mean Absolute Error):      ${mae:.2f}")
print(f"RMSE (Root Mean Squared Error):  ${rmse:.2f}")
print(f"MAPE (Mean Absolute % Error):    {mape:.1f}%")
print()
print("Interpretation:")
print(f"  On average, Prophet's predictions were")
print(f"  ${mae:.2f} away from the actual price.")
print(f"  That is {mape:.1f}% off from the true value.")