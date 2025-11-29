# src/data.py
import yfinance as yf
import pandas as pd
import numpy as np

def get_data(assets, period="2y"):
    print(f"Downloading data for {assets}...")
    raw_prices = yf.download(assets, period=period)["Close"]
    
    if isinstance(raw_prices, pd.Series):
        prices_df = raw_prices.to_frame()
        prices_df.columns = assets
    else:
        prices_df = raw_prices.dropna()
    
    return prices_df

def prepare_data(prices_df):
    # Daily returns
    returns_daily = prices_df.pct_change().dropna()

    # Estimate periods-per-year (trading_days) from the data and annualize stats
    counts_per_year = returns_daily.index.to_series().groupby(returns_daily.index.year).count()
    if len(counts_per_year) > 0:
        trading_days = int(counts_per_year.mean())
    else:
        trading_days = 252  # fallback to business days per year

    mu_annual = returns_daily.mean() * trading_days        # expected annual return per asset
    Sigma_annual = returns_daily.cov() * trading_days      # annualized covariance matrix
    
    return mu_annual, Sigma_annual, trading_days
