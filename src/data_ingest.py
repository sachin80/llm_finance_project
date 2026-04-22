import time

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# As done in classe, load in the Excel S&P500 sheet and create the dataframe.
def load_universe(path_in, sheet):
    """Load the stock universe from in-class Excel file and keep one clean row 
    per ticker."""
    df = pd.read_excel(path_in, sheet_name=sheet)

    if "Ticker" not in df.columns:
        raise ValueError("Expected a 'Ticker' column in the spreadsheet.")

    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df = df[df["Ticker"] != ""].copy()
    df = df.drop_duplicates(subset=["Ticker"]).reset_index(drop=True)
    return df


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(Exception),
)
# As done in class, fetch company descriptions from the Yahoo Finance API. 
def fetch_description(ticker):
    """Fetch one company description from Yahoo Finance."""
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.get_info()
    description = info.get("longBusinessSummary") or info.get("longDescription") or ""
    return description.strip()

# As done in class, loop through each ticker as key to company description as 
# value in a hashmap. Then convert hmap to dataframe and as a column. 
def add_company_descriptions(df, sleep_sec=0.25):
    """Add a 'Company Description' column using Yahoo Finance summaries."""
    description_map = {}
    tickers = df["Ticker"].tolist()

    for i, ticker in enumerate(tickers, start=1):
        try:
            description = fetch_description(ticker)
            time.sleep(sleep_sec)
        except Exception:
            description = ""

        description_map[ticker] = description
        print(f"Descriptions: processed {i}/{len(tickers)}")

    df = df.copy()
    # Create company description column by using ticker to company description
    # hmap. 
    df["Company Description"] = df["Ticker"].map(description_map).fillna("")
    return df

# Use yfinance to download prices for the tickers. Clean columns and return as 
# a dataframe sorted by date.
def download_prices(tickers, start_date, end_date):
    """Download the adjusted close prices for the stock universe."""
    prices = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False)["Close"]
    
    # Check if prices is a dataframe 
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    
    # Drop any columns of NaNs then sort by DateIndex. 
    prices = prices.dropna(axis=1, how="all").sort_index()
    return prices

# Compute simple daily returns for the prices we 
def compute_returns(prices):
    """Convert price levels into simple daily returns."""
    return prices.pct_change().dropna(how="all")
