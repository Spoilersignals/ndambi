import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_stock_data(symbol, period='1y', start_price=100):
    """Generate realistic synthetic stock data for demo purposes"""
    
    # Map period to days
    period_map = {
        '1d': 1,
        '5d': 5,
        '1mo': 30,
        '3mo': 90,
        '6mo': 180,
        '1y': 365,
        '2y': 730,
        '5y': 1825,
        'max': 3650
    }
    
    days = period_map.get(period, 365)
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate price data with realistic movement
    np.random.seed(hash(symbol) % 10000)  # Consistent data per symbol
    
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
    prices = [start_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Add some trend
    trend = np.linspace(0, 0.2, len(dates))
    prices = np.array(prices) * (1 + trend)
    
    # Generate volume
    base_volume = np.random.randint(1000000, 10000000)
    volumes = np.random.randint(base_volume * 0.5, base_volume * 1.5, len(dates))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Close': prices,
        'Volume': volumes,
        'Open': prices * np.random.uniform(0.98, 1.02, len(prices)),
        'High': prices * np.random.uniform(1.00, 1.05, len(prices)),
        'Low': prices * np.random.uniform(0.95, 1.00, len(prices)),
    }, index=dates)
    
    return df

# Preset symbols with realistic starting prices
DEMO_SYMBOLS = {
    'AAPL': 150,
    'GOOGL': 140,
    'MSFT': 350,
    'AMZN': 130,
    'TSLA': 200,
    'META': 300,
    'NVDA': 450,
    'JPM': 150,
    'V': 250,
    'WMT': 160
}

def is_valid_demo_symbol(symbol):
    """Check if symbol is available in demo mode"""
    return symbol.upper() in DEMO_SYMBOLS

def get_demo_price(symbol):
    """Get starting price for a symbol"""
    return DEMO_SYMBOLS.get(symbol.upper(), 100)
