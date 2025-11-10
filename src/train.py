import numpy as np
import os
import json
from datetime import datetime
from src.model import StockPredictionANN, LSTMStockPredictor
from config import Config

def load_preprocessed_data():
    print("Loading preprocessed data...")
    
    data_dir = Config.PROCESSED_DATA_DIR
    
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    print(f"Data loaded successfully!")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Input shape: {X_train.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_ann_model():
    print("\n" + "="*60)
    print("TRAINING FEEDFORWARD ANN MODEL")
    print("="*60 + "\n")
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data()
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    ann = StockPredictionANN(input_shape=input_shape)
    
    ann.build_model()
    
    history = ann.train(
        X_train, y_train,
        X_val, y_val,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE
    )
    
    ann.save_model()
    
    save_training_history(history, 'ann_training_history.json')
    
    return ann, history

def train_lstm_model():
    print("\n" + "="*60)
    print("TRAINING LSTM MODEL")
    print("="*60 + "\n")
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data()
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    lstm = LSTMStockPredictor(input_shape=input_shape)
    
    lstm.build_model()
    
    history = lstm.train(
        X_train, y_train,
        X_val, y_val,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE
    )
    
    lstm.save_model()
    
    save_training_history(history, 'lstm_training_history.json')
    
    return lstm, history

def save_training_history(history, filename):
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'mae': [float(x) for x in history.history['mae']],
        'val_mae': [float(x) for x in history.history['val_mae']],
        'mse': [float(x) for x in history.history['mse']],
        'val_mse': [float(x) for x in history.history['val_mse']],
    }
    
    filepath = os.path.join(Config.MODELS_DIR, filename)
    
    with open(filepath, 'w') as f:
        json.dump(history_dict, f, indent=4)
    
    print(f"Training history saved to {filepath}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'lstm':
        train_lstm_model()
    else:
        train_ann_model()
