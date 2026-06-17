import yfinance as yf
import pandas as pd
import os

# ── Download stock data ───────────────────────────────────────────────
print("Downloading Apple stock data...")
apple = yf.download(tickers="AAPL", period="2y", interval="1d")

# ── Flatten the MultiIndex columns ────────────────────────────────────
# yfinance returns a two-level column header
# We need to flatten it to a single level
# Before: ('Close', 'AAPL'), ('High', 'AAPL') etc.
# After:  'Close', 'High' etc.
apple.columns = apple.columns.get_level_values(0)
print(f"Columns after flattening: {apple.columns.tolist()}")

# ── Keep only the Close price ─────────────────────────────────────────
# Prophet only needs dates and closing prices
apple = apple[["Close"]].copy()

# ── Reset index to make Date a regular column ──────────────────────────
# Prophet needs 'ds' as a regular column, not an index
apple = apple.reset_index()
print(f"\nBefore renaming:")
print(apple.head(3))

# ── Rename columns to what Prophet expects ─────────────────────────────
# ds = datestamp (the date)
# y  = the value we want to forecast (closing price)
apple = apple.rename(columns={"Date": "ds", "Close": "y"})

# ── Remove timezone from dates ─────────────────────────────────────────
# Prophet does not handle timezone-aware dates well
apple["ds"] = pd.to_datetime(apple["ds"]).dt.tz_localize(None)

print(f"\nAfter renaming (Prophet format):")
print(apple.head(3))
print(f"\nShape: {apple.shape}")
print(f"Date range: {apple['ds'].min()} to {apple['ds'].max()}")

# ── Save to CSV ────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
apple.to_csv("data/apple_prophet.csv", index=False)
print(f"\nSaved to data/apple_prophet.csv")