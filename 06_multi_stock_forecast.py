import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Define our 5 companies ─────────────────────────────────────────────
companies = {
    "AAPL": "Apple",
    "JPM":  "JPMorgan",
    "GS":   "Goldman Sachs",
    "V":    "Visa",
    "MA":   "Mastercard"
}

# ── Store results ──────────────────────────────────────────────────────
results = []

print("Forecasting 30 days ahead for all 5 companies...")
print("=" * 55)

for ticker, name in companies.items():
    print(f"\nProcessing {name} ({ticker})...")

    # Download data
    raw = yf.download(tickers=ticker, period="2y", interval="1d", 
                      progress=False)

    # Flatten MultiIndex columns
    raw.columns = raw.columns.get_level_values(0)

    # Prepare for Prophet
    df = raw[["Close"]].reset_index()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)

    # Train Prophet
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model.fit(df)

    # Forecast 30 days ahead
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Get current and predicted prices
    current_price = df["y"].iloc[-1]
    predicted_price = forecast["yhat"].iloc[-1]
    predicted_low = forecast["yhat_lower"].iloc[-1]
    predicted_high = forecast["yhat_upper"].iloc[-1]
    pct_change = ((predicted_price - current_price) / current_price) * 100

    results.append({
        "Company": name,
        "Ticker": ticker,
        "Current Price": round(current_price, 2),
        "Predicted Price": round(predicted_price, 2),
        "Predicted Low": round(predicted_low, 2),
        "Predicted High": round(predicted_high, 2),
        "Expected Change %": round(pct_change, 2)
    })

    print(f"  Current:   ${current_price:.2f}")
    print(f"  Predicted: ${predicted_price:.2f} ({pct_change:+.1f}%)")

# ── Summary table ──────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Expected Change %", ascending=False)

print()
print("=" * 65)
print("30-DAY FORECAST SUMMARY — ALL COMPANIES")
print("=" * 65)
print(f"{'Company':<15} {'Current':>10} {'Predicted':>10} {'Change %':>10}")
print("-" * 65)
for _, row in results_df.iterrows():
    arrow = "↑" if row["Expected Change %"] > 0 else "↓"
    print(f"{row['Company']:<15} "
          f"${row['Current Price']:>9.2f} "
          f"${row['Predicted Price']:>9.2f} "
          f"{arrow} {row['Expected Change %']:>8.1f}%")

# ── Bar chart comparison ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Chart 1: Expected % change comparison
colors = ["green" if x > 0 else "red" 
          for x in results_df["Expected Change %"]]
axes[0].barh(results_df["Company"], 
             results_df["Expected Change %"],
             color=colors, alpha=0.7)
axes[0].axvline(x=0, color="black", linewidth=0.8)
axes[0].set_title("30-Day Expected Price Change (%)")
axes[0].set_xlabel("Expected Change (%)")
for i, (val, company) in enumerate(zip(results_df["Expected Change %"],
                                        results_df["Company"])):
    axes[0].text(val + 0.1, i, f"{val:+.1f}%", va="center", fontsize=9)

# Chart 2: Current vs predicted price
x = range(len(results_df))
width = 0.35
axes[1].bar([i - width/2 for i in x], results_df["Current Price"],
            width, label="Current Price", color="steelblue", alpha=0.7)
axes[1].bar([i + width/2 for i in x], results_df["Predicted Price"],
            width, label="Predicted Price (30 days)", 
            color="orange", alpha=0.7)
axes[1].set_xticks(list(x))
axes[1].set_xticklabels(results_df["Company"], rotation=15)
axes[1].set_title("Current vs Predicted Price")
axes[1].set_ylabel("Price (USD)")
axes[1].legend()

plt.suptitle("StockSense — 30-Day Multi-Stock Forecast", 
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("multi_stock_forecast.png")
plt.show()
print()
print("Chart saved as multi_stock_forecast.png")