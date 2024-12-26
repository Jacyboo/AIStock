import pandas as pd
from typing import Dict, Any, List
from datetime import datetime

def get_manual_financial_metrics(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process manually input financial metrics."""
    return [data]

def get_manual_insider_trades(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process manually input insider trades."""
    return data

def get_manual_market_cap(value: float) -> float:
    """Process manually input market cap."""
    return value

def get_manual_prices(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process manually input price data."""
    # Ensure each price entry has required fields
    for entry in data:
        if not all(key in entry for key in ["time", "open", "close", "high", "low", "volume"]):
            raise ValueError("Each price entry must contain: time, open, close, high, low, volume")
        try:
            datetime.strptime(entry["time"], "%Y-%m-%d")
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")
    
    return sorted(data, key=lambda x: x["time"])

def prices_to_df(prices: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame(prices)
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df

def get_price_data(prices: List[Dict[str, Any]], start_date: str, end_date: str) -> pd.DataFrame:
    """Filter and return price data for the specified date range."""
    df = prices_to_df(prices)
    mask = (df.index >= start_date) & (df.index <= end_date)
    return df[mask] 