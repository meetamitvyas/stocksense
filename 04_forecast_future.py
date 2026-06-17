import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Load full dataset ──────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/apple_prophet.csv")
df["ds"] = pd.to_datetime(df["ds"])
print(f"Training on {len(df)} days of data")
print(f"Last known price: ${df['y'].iloc[-1]:.2f} on {df['ds'].iloc[-1].date()}")
print()

# ── Train Prophet on ALL available data ────────────────────────────────
# We use all 501 days now — the more data Prophet has, the better
# it understands the trend and seasonality patterns
print("Training Prophet on full dataset...")
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)
model.fit(df)
print("Training complete!")
print()

# ── Create future dates for next 30 days ──────────────────────────────
# make_future_dataframe creates a DataFrame with future dates
# periods=30 means 30 days ahead
# Prophet automatically skips weekends for stock data
future = model.make_future_dataframe(periods=30)
print(f"Forecasting {30} days into the future...")
forecast = model.predict(future)

# ── Extract just the future predictions ───────────────────────────────
last_date = df["ds"].max()
future_forecast = forecast[forecast["ds"] > last_date].copy()

print()
print("Next 30 days forecast:")
print("=" * 55)
print(f"{'Date':<15} {'Predicted':>12} {'Low':>12} {'High':>12}")
print("-" * 55)
for _, row in future_forecast.iterrows():
    print(f"{str(row['ds'].date()):<15} "
          f"${row['yhat']:>10.2f} "
          f"${row['yhat_lower']:>10.2f} "
          f"${row['yhat_upper']:>10.2f}")

print()
print(f"30-day price range prediction:")
print(f"  Lowest expected:  ${future_forecast['yhat_lower'].min():.2f}")
print(f"  Highest expected: ${future_forecast['yhat_upper'].max():.2f}")
print(f"  Central forecast: ${future_forecast['yhat'].iloc[-1]:.2f} "
      f"(in 30 days)")

# ── Plot the forecast ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

# Show last 90 days of actual data for context
recent = df[df["ds"] >= df["ds"].max() - pd.Timedelta(days=90)]
ax.plot(recent["ds"], recent["y"],
        color="steelblue", linewidth=2, label="Actual price (last 90 days)")

# Plot future forecast
ax.plot(future_forecast["ds"], future_forecast["yhat"],
        color="red", linewidth=2, linestyle="--", label="30-day forecast")

# Confidence interval
ax.fill_between(
    future_forecast["ds"],
    future_forecast["yhat_lower"],
    future_forecast["yhat_upper"],
    alpha=0.25, color="red", label="Confidence interval"
)

# Mark the last known price
ax.axvline(x=last_date, color="gray", linestyle=":", linewidth=1.5,
           label="Today")
ax.annotate(f"Last known: ${df['y'].iloc[-1]:.2f}",
            xy=(last_date, df["y"].iloc[-1]),
            xytext=(15, 10), textcoords="offset points",
            fontsize=9, color="steelblue")

ax.set_title("Apple Stock Price — 30-Day Future Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("future_forecast.png")
plt.show()
print("Chart saved as future_forecast.png")