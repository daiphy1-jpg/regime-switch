# FX Regime Detection: Trend Following vs Mean Reversion (USD/CAD)

This project implements a **moving-average–based regime detection framework** to classify the USD/CAD foreign exchange market as either **trend-following** or **mean-reverting** over the past six months.

The goal is to dynamically identify market conditions and enable **adaptive strategy selection**, rather than relying on a single trading style across all environments.

---

## 📌 Overview

Financial markets alternate between periods of sustained trends and range-bound, mean-reverting behavior.  
This project uses **technical indicators derived from price action** to classify regimes using:

- 20-day and 50-day moving averages  
- Moving-average slopes (trend persistence)
- Volatility-normalized distance from the moving average  

The resulting regime labels can be used to **switch between trend-following and mean-reversion strategies**.

---

## 🧠 Methodology

### 1. Data
- **Instrument:** USD/CAD FX (ticker: `CAD=X`)
- **Source:** Yahoo Finance (`yfinance`)
- **Frequency:** Daily
- **Lookback:** Last 6 months

### 2. Features
- **MA20 / MA50:** Short- and medium-term trend indicators
- **MA Slopes:** Directional strength over a rolling window
- **Rolling Volatility:** 20-day standard deviation of returns
- **Z-Score vs MA:** Price deviation from MA scaled by volatility

### 3. Regime Classification Logic
- **Trend-Following Regime**
  - MA20 above (or below) MA50
  - Both MA slopes aligned and non-flat
- **Mean-Reversion Regime**
  - Flat or frequently crossing moving averages
  - Price oscillating around MA without directional persistence

Each trading day is labeled accordingly.

---

## ⚙️ Strategy Signals (Optional Extension)

- **Trend-Following Signal**
  - Long USD when MA20 > MA50
  - Short USD when MA20 < MA50

- **Mean-Reversion Signal**
  - Fade volatility-adjusted deviations from MA20
  - Long when price is significantly below MA
  - Short when price is significantly above MA

Signals are selected dynamically based on the detected regime.

---

## 🛠️ Tech Stack

- Python
- pandas
- numpy
- yfinance

---

## 📁 Project Structure
.
├── regime_switch.py
├── README.md
└── outputs/
    └── simple_regime_data.csv   (optional)
    └── simple_regime_spread.png   (optional)


