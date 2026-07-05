import pandas as pd
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import yfinance as yf

choice = input(
    "Choose a comparison:\n"
    "1 = Visa vs Mastercard (V/MA)\n"
    "2 = Gold vs Silver (GLD/SLV)\n"
    "Enter 1 or 2: "
).strip()

if choice == "1":
    tickers = ["V", "MA"]
    labels = ["Visa", "Mastercard"]
    title = "Visa vs Mastercard"
elif choice == "2":
    tickers = ["GLD", "SLV"]
    labels = ["Gold (GLD)", "Silver (SLV)"]
    title = "Gold vs Silver"
else:
    raise SystemExit("Please enter 1 or 2.")

# Pull the longest available history for the selected pair.
data = yf.download(tickers, start="1900-01-01", auto_adjust=True, progress=False)

if isinstance(data.columns, pd.MultiIndex):
    close = data.xs("Close", axis=1, level=0)
else:
    close = data[["Close"]].copy()

close = close.dropna()
if close.shape[1] < 2:
    raise SystemExit("Not enough data was returned for that comparison.")

available_start = close.index.min()
available_end = close.index.max()
max_years = max(1, int((available_end - available_start).days / 365.25))

years_back_input = input(
    f"How many years back? Max available: {max_years} years\n"
    "Enter a number: "
).strip()

try:
    years_back = int(years_back_input)
except ValueError:
    raise SystemExit("Please enter a whole number.")

if years_back < 1:
    raise SystemExit("Please enter a positive number.")

years_back = min(years_back, max_years)
cutoff = available_end - pd.DateOffset(years=years_back)
filtered = close[close.index >= cutoff]

normalized = filtered / filtered.iloc[0] * 100
normalized.columns = labels

fig, ax = plt.subplots(figsize=(12, 6))
for column in normalized.columns:
    normalized[column].plot(ax=ax, label=column, linewidth=2)

ax.set_title(f"{title} (normalized to first available close)")
ax.set_xlabel("Date")
ax.set_ylabel("Normalized price (first close = 100)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
