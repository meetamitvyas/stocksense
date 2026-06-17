import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# ── Load data and train model ──────────────────────────────────────────
print("Loading data and training model...")
df = pd.read_csv("data/apple_prophet.csv")
df["ds"] = pd.to_datetime(df["ds"])

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)
model.fit(df)

# ── Generate forecast ──────────────────────────────────────────────────
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# ── Plot components ────────────────────────────────────────────────────
# Prophet has a built-in method to plot each component separately
# This shows us WHAT is driving the forecast
print("Plotting components...")
fig = model.plot_components(forecast)
plt.suptitle("FinSight AI — Prophet Forecast Components", 
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("forecast_components.png", bbox_inches="tight")
plt.show()
print("Chart saved as forecast_components.png")

# ── Also plot the change points ────────────────────────────────────────
# Change points are moments where the trend changed direction significantly
# Prophet automatically detects these
from prophet.plot import add_changepoints_to_plot

print()
print("Plotting trend with change points...")
fig2, ax2 = plt.subplots(figsize=(14, 5))
fig2 = model.plot(forecast, ax=ax2)
add_changepoints_to_plot(ax2, model, forecast)
ax2.set_title("Apple Stock Price — Trend Change Points")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price (USD)")
plt.tight_layout()
plt.savefig("changepoints.png")
plt.show()
print("Chart saved as changepoints.png")