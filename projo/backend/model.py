import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

class StockPredictor:
    def __init__(self, look_back=60):
        self.look_back = look_back
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._build_model()
        self.training_history = None
        
    def _build_model(self):
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(self.look_back, 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def _prepare_data(self, data):
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.look_back, len(scaled_data)):
            X.append(scaled_data[i-self.look_back:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def train(self, data, epochs=50, batch_size=32):
        X, y = self._prepare_data(data)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        self.training_history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
        
        return self.training_history
    
    def evaluate(self, data):
        X, y = self._prepare_data(data)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        predictions = self.model.predict(X, verbose=0)
        predictions = self.scaler.inverse_transform(predictions)
        y_actual = self.scaler.inverse_transform(y.reshape(-1, 1))
        
        mse = mean_squared_error(y_actual, predictions)
        mae = mean_absolute_error(y_actual, predictions)
        rmse = np.sqrt(mse)
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse)
        }
        
    def predict(self, data, days=30):
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        last_data = scaled_data[-self.look_back:]
        predictions = []
        
        current_batch = last_data.reshape(1, self.look_back, 1)
        
        for _ in range(days):
            pred = self.model.predict(current_batch, verbose=0)[0]
            predictions.append(pred[0])
            
            current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
        
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()
