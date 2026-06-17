import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ── What are we doing? ────────────────────────────────────────────────
# We are loading 2 years of Apple stock price history
# and exploring it to understand the patterns before forecasting

# ── Step 1: Download Apple stock data ─────────────────────────────────
# ticker = the stock market symbol for Apple
# period = how far back to go
# interval = one row per day
print("Downloading Apple stock data...")
apple = yf.download(tickers="AAPL", period="2y", interval="1d")

print(f"Downloaded {len(apple)} trading days of data")
print()

# ── Step 2: Look at the data ──────────────────────────────────────────
print("First 5 rows:")
print(apple.head())
print()
print("Last 5 rows:")
print(apple.tail())
print()

# ── Step 3: Basic statistics ──────────────────────────────────────────
print("Basic statistics for Closing Price:")
print(apple["Close"].describe())
print()

# ── Step 4: Plot the closing price over time ──────────────────────────
# This gives us a visual sense of the trend
plt.figure(figsize=(12, 5))
plt.plot(apple.index, apple["Close"], color="steelblue", linewidth=1.5)
plt.title("Apple Stock Price - Last 2 Years")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("apple_price_history.png")
plt.show()
print("Chart saved as apple_price_history.png")