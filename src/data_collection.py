import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
import os
from textblob import TextBlob
from config import Config

class DataCollector:
    def __init__(self, symbol=Config.DEFAULT_STOCK_SYMBOL):
        self.symbol = symbol
        self.config = Config()
    
    def fetch_stock_data(self, start_date, end_date):
        print(f"Fetching stock data for {self.symbol} from {start_date} to {end_date}...")
        try:
            stock = yf.Ticker(self.symbol)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                raise ValueError(f"No data found for {self.symbol}")
            
            df = df.reset_index()
            df['Symbol'] = self.symbol
            
            print(f"Successfully fetched {len(df)} records")
            return df
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return pd.DataFrame()
    
    def fetch_economic_indicators(self):
        print("Fetching economic indicators...")
        
        indicators_data = {
            'Date': [],
            'GDP_Growth': [],
            'Inflation_Rate': [],
            'Interest_Rate': []
        }
        
        print("Note: Economic indicators require additional API setup (FRED, World Bank, etc.)")
        print("Using placeholder data for demonstration. Implement real API calls as needed.")
        
        return pd.DataFrame(indicators_data)
    
    def fetch_news_sentiment(self, days_back=30):
        print(f"Fetching news sentiment for {self.symbol} (last {days_back} days)...")
        
        try:
            stock = yf.Ticker(self.symbol)
            news = stock.news
            
            sentiment_data = []
            for article in news[:20]:
                title = article.get('title', '')
                published = article.get('providerPublishTime', None)
                
                if title:
                    blob = TextBlob(title)
                    sentiment = blob.sentiment.polarity
                    
                    sentiment_data.append({
                        'Date': datetime.fromtimestamp(published) if published else datetime.now(),
                        'Title': title,
                        'Sentiment': sentiment
                    })
            
            df = pd.DataFrame(sentiment_data)
            if not df.empty:
                df['Date'] = pd.to_datetime(df['Date']).dt.date
                daily_sentiment = df.groupby('Date')['Sentiment'].mean().reset_index()
                daily_sentiment.columns = ['Date', 'Avg_Sentiment']
                print(f"Processed sentiment for {len(daily_sentiment)} days")
                return daily_sentiment
            
        except Exception as e:
            print(f"Error fetching news sentiment: {e}")
        
        return pd.DataFrame()
    
    def fetch_company_fundamentals(self):
        print(f"Fetching company fundamentals for {self.symbol}...")
        
        try:
            stock = yf.Ticker(self.symbol)
            info = stock.info
            
            fundamentals = {
                'PE_Ratio': info.get('trailingPE', None),
                'PB_Ratio': info.get('priceToBook', None),
                'Market_Cap': info.get('marketCap', None),
                'Revenue': info.get('totalRevenue', None),
                'Profit_Margin': info.get('profitMargins', None),
                'ROE': info.get('returnOnEquity', None)
            }
            
            print(f"Fetched fundamentals: {fundamentals}")
            return fundamentals
            
        except Exception as e:
            print(f"Error fetching fundamentals: {e}")
            return {}
    
    def collect_all_data(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        stock_data = self.fetch_stock_data(start_date, end_date)
        
        if stock_data.empty:
            raise ValueError("Failed to fetch stock data")
        
        sentiment_data = self.fetch_news_sentiment()
        fundamentals = self.fetch_company_fundamentals()
        
        stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
        
        if not sentiment_data.empty:
            sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date']).dt.date
            stock_data = stock_data.merge(sentiment_data, on='Date', how='left')
            stock_data['Avg_Sentiment'].fillna(0, inplace=True)
        
        for key, value in fundamentals.items():
            stock_data[key] = value
        
        output_path = os.path.join(Config.RAW_DATA_DIR, f'{self.symbol}_raw_data.csv')
        stock_data.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
        
        return stock_data

if __name__ == "__main__":
    collector = DataCollector('AAPL')
    data = collector.collect_all_data()
    print(f"\nCollected data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
