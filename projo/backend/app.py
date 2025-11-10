from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
from model import StockPredictor
from datetime import datetime, timedelta
import requests_cache
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter
from demo_data import generate_stock_data, is_valid_demo_symbol, get_demo_price
import os

app = Flask(__name__)
CORS(app)

# Enable demo mode if Yahoo Finance is unavailable
DEMO_MODE = os.environ.get('DEMO_MODE', 'true').lower() == 'true'

# Create a session with caching and rate limiting to avoid blocks
class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass

session = CachedLimiterSession(
    limiter=Limiter(RequestRate(2, Duration.SECOND*5)),
    bucket_class=MemoryQueueBucket,
    backend=SQLiteCache("yfinance.cache"),
)

session.headers['User-agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

predictor = StockPredictor()

@app.route('/api/stock/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    try:
        period = request.args.get('period', '1y')
        
        # Try real data first, fallback to demo
        if DEMO_MODE:
            if not is_valid_demo_symbol(symbol):
                return jsonify({'error': f'Symbol {symbol} not available. Try: AAPL, GOOGL, MSFT, AMZN, TSLA, META, NVDA'}), 404
            hist = generate_stock_data(symbol, period, get_demo_price(symbol))
        else:
            stock = yf.Ticker(symbol, session=session)
            hist = stock.history(period=period)
            
            if hist.empty:
                return jsonify({'error': f'No data found for symbol {symbol}. Try: AAPL, MSFT, GOOGL, AMZN'}), 404
        
        data = {
            'dates': hist.index.strftime('%Y-%m-%d').tolist(),
            'prices': hist['Close'].tolist(),
            'volume': hist['Volume'].tolist(),
            'symbol': symbol
        }
        
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict/<symbol>', methods=['POST'])
def predict_stock(symbol):
    try:
        data = request.json
        days = data.get('days', 30)
        
        if DEMO_MODE:
            if not is_valid_demo_symbol(symbol):
                return jsonify({'error': f'Symbol {symbol} not available'}), 404
            hist = generate_stock_data(symbol, '2y', get_demo_price(symbol))
        else:
            stock = yf.Ticker(symbol, session=session)
            hist = stock.history(period='2y')
            
            if hist.empty:
                return jsonify({'error': f'No data found for symbol {symbol}'}), 404
        
        predictions = predictor.predict(hist['Close'].values, days)
        metrics = predictor.evaluate(hist['Close'].values)
        
        last_date = hist.index[-1]
        future_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                       for i in range(days)]
        
        return jsonify({
            'predictions': predictions.tolist(),
            'dates': future_dates,
            'symbol': symbol,
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/train/<symbol>', methods=['POST'])
def train_model(symbol):
    try:
        if DEMO_MODE:
            if not is_valid_demo_symbol(symbol):
                return jsonify({'error': f'Symbol {symbol} not available'}), 404
            hist = generate_stock_data(symbol, '5y', get_demo_price(symbol))
        else:
            stock = yf.Ticker(symbol, session=session)
            hist = stock.history(period='5y')
            
            if hist.empty:
                return jsonify({'error': f'No data found for symbol {symbol}'}), 404
        
        history = predictor.train(hist['Close'].values)
        metrics = predictor.evaluate(hist['Close'].values)
        
        return jsonify({
            'message': f'Model trained successfully for {symbol}',
            'metrics': metrics,
            'loss': history.history['loss'][-1] if history else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
