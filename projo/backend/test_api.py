import unittest
import json
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from app import app

class TestStockAPI(unittest.TestCase):
    
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        
        # Create mock stock data
        dates = pd.date_range('2024-01-01', periods=100)
        self.mock_hist = pd.DataFrame({
            'Close': np.random.uniform(150, 200, 100),
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
    
    @patch('app.yf.Ticker')
    def test_get_stock_data(self, mock_ticker):
        mock_ticker.return_value.history.return_value = self.mock_hist
        
        response = self.app.get('/api/stock/AAPL?period=1mo')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('dates', data)
        self.assertIn('prices', data)
        self.assertIn('volume', data)
        self.assertIn('symbol', data)
        self.assertEqual(data['symbol'], 'AAPL')
    
    @patch('app.yf.Ticker')
    def test_get_invalid_stock(self, mock_ticker):
        mock_ticker.return_value.history.return_value = pd.DataFrame()
        
        response = self.app.get('/api/stock/INVALID999')
        
        self.assertEqual(response.status_code, 404)
    
    @patch('app.yf.Ticker')
    def test_predict_stock(self, mock_ticker):
        mock_ticker.return_value.history.return_value = self.mock_hist
        
        response = self.app.post('/api/predict/AAPL',
                                json={'days': 7},
                                content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('predictions', data)
        self.assertIn('dates', data)
        self.assertIn('symbol', data)
        self.assertIn('metrics', data)
        self.assertEqual(len(data['predictions']), 7)
        self.assertEqual(len(data['dates']), 7)
    
    @patch('app.yf.Ticker')
    def test_predict_without_days(self, mock_ticker):
        mock_ticker.return_value.history.return_value = self.mock_hist
        
        response = self.app.post('/api/predict/AAPL',
                                json={},
                                content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(len(data['predictions']), 30)
    
    @patch('app.yf.Ticker')
    def test_train_model(self, mock_ticker):
        mock_ticker.return_value.history.return_value = self.mock_hist
        
        response = self.app.post('/api/train/AAPL')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('message', data)
        self.assertIn('metrics', data)
        self.assertIn('AAPL', data['message'])

if __name__ == '__main__':
    unittest.main()
