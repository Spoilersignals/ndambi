import unittest
import numpy as np
from model import StockPredictor

class TestStockPredictor(unittest.TestCase):
    
    def setUp(self):
        self.predictor = StockPredictor(look_back=10)
        self.sample_data = np.random.rand(100) * 100 + 50
    
    def test_model_initialization(self):
        self.assertIsNotNone(self.predictor.model)
        self.assertEqual(self.predictor.look_back, 10)
        self.assertIsNotNone(self.predictor.scaler)
    
    def test_prepare_data(self):
        X, y = self.predictor._prepare_data(self.sample_data)
        
        expected_samples = len(self.sample_data) - self.predictor.look_back
        self.assertEqual(len(X), expected_samples)
        self.assertEqual(len(y), expected_samples)
        self.assertEqual(X.shape[1], self.predictor.look_back)
    
    def test_train(self):
        history = self.predictor.train(self.sample_data, epochs=2, batch_size=8)
        
        self.assertIsNotNone(history)
        self.assertIn('loss', history.history)
        self.assertGreater(len(history.history['loss']), 0)
    
    def test_predict(self):
        self.predictor.train(self.sample_data, epochs=2, batch_size=8)
        
        predictions = self.predictor.predict(self.sample_data, days=5)
        
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(isinstance(p, (int, float, np.number)) for p in predictions))
    
    def test_evaluate(self):
        self.predictor.train(self.sample_data, epochs=2, batch_size=8)
        
        metrics = self.predictor.evaluate(self.sample_data)
        
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        self.assertGreater(metrics['mse'], 0)
        self.assertGreater(metrics['mae'], 0)
        self.assertGreater(metrics['rmse'], 0)
    
    def test_predict_without_training(self):
        predictions = self.predictor.predict(self.sample_data, days=3)
        
        self.assertEqual(len(predictions), 3)

if __name__ == '__main__':
    unittest.main()
