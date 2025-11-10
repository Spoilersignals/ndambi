import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import os
from config import Config

class StockPredictionANN:
    def __init__(self, input_shape, learning_rate=Config.LEARNING_RATE):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
    def build_model(self, layers_config=None):
        print("Building ANN model...")
        
        if layers_config is None:
            layers_config = [
                {'units': 128, 'dropout': 0.2},
                {'units': 64, 'dropout': 0.2},
                {'units': 32, 'dropout': 0.15},
            ]
        
        model = keras.Sequential()
        
        model.add(layers.Input(shape=self.input_shape))
        model.add(layers.Flatten())
        
        for i, layer_config in enumerate(layers_config):
            model.add(layers.Dense(
                units=layer_config['units'],
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001),
                name=f'dense_{i+1}'
            ))
            
            if layer_config.get('dropout', 0) > 0:
                model.add(layers.Dropout(layer_config['dropout'], name=f'dropout_{i+1}'))
            
            model.add(layers.BatchNormalization(name=f'batch_norm_{i+1}'))
        
        model.add(layers.Dense(1, activation='linear', name='output'))
        
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        
        print("\nModel Architecture:")
        model.summary()
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=Config.EPOCHS, 
              batch_size=Config.BATCH_SIZE):
        print(f"\nTraining model for {epochs} epochs with batch size {batch_size}...")
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=Config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            
            ModelCheckpoint(
                filepath=os.path.join(Config.MODELS_DIR, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
        return self.history
    
    def save_model(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(Config.MODELS_DIR, 'stock_prediction_model.h5')
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(Config.MODELS_DIR, 'stock_prediction_model.h5')
        
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not loaded or built.")
        
        predictions = self.model.predict(X)
        return predictions

class LSTMStockPredictor:
    def __init__(self, input_shape, learning_rate=Config.LEARNING_RATE):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
    def build_model(self):
        print("Building LSTM model...")
        
        model = keras.Sequential()
        
        model.add(layers.LSTM(
            units=100, 
            return_sequences=True, 
            input_shape=self.input_shape,
            dropout=0.2,
            recurrent_dropout=0.2
        ))
        
        model.add(layers.LSTM(
            units=50, 
            return_sequences=False,
            dropout=0.2,
            recurrent_dropout=0.2
        ))
        
        model.add(layers.Dense(25, activation='relu'))
        model.add(layers.Dropout(0.2))
        
        model.add(layers.Dense(1))
        
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        
        print("\nLSTM Model Architecture:")
        model.summary()
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=Config.EPOCHS, 
              batch_size=Config.BATCH_SIZE):
        print(f"\nTraining LSTM model for {epochs} epochs...")
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=Config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            
            ModelCheckpoint(
                filepath=os.path.join(Config.MODELS_DIR, 'best_lstm_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def save_model(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(Config.MODELS_DIR, 'lstm_stock_model.h5')
        
        self.model.save(filepath)
        print(f"LSTM model saved to {filepath}")
    
    def load_model(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(Config.MODELS_DIR, 'lstm_stock_model.h5')
        
        self.model = keras.models.load_model(filepath)
        print(f"LSTM model loaded from {filepath}")
        return self.model
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not loaded or built.")
        
        predictions = self.model.predict(X)
        return predictions

if __name__ == "__main__":
    print("Model classes defined. Use train.py to train models.")
