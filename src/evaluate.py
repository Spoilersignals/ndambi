import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from src.model import StockPredictionANN, LSTMStockPredictor
from config import Config
import json

class ModelEvaluator:
    def __init__(self, model_type='ann'):
        self.model_type = model_type
        self.model = None
        self.predictions = None
        self.actuals = None
        
    def load_model(self):
        print(f"Loading {self.model_type.upper()} model...")
        
        if self.model_type == 'ann':
            self.model = StockPredictionANN(input_shape=(60, 16))
            self.model.load_model()
        elif self.model_type == 'lstm':
            self.model = LSTMStockPredictor(input_shape=(60, 16))
            self.model.load_model()
        else:
            raise ValueError("Model type must be 'ann' or 'lstm'")
    
    def load_test_data(self):
        print("Loading test data...")
        
        X_test = np.load(os.path.join(Config.PROCESSED_DATA_DIR, 'X_test.npy'))
        y_test = np.load(os.path.join(Config.PROCESSED_DATA_DIR, 'y_test.npy'))
        
        print(f"Test data loaded: {X_test.shape}")
        return X_test, y_test
    
    def make_predictions(self, X_test):
        print("Making predictions on test data...")
        
        predictions = self.model.predict(X_test)
        
        return predictions.flatten()
    
    def calculate_metrics(self, y_true, y_pred):
        print("\nCalculating evaluation metrics...")
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2_Score': r2,
            'MAPE': mape
        }
        
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        for metric, value in metrics.items():
            print(f"{metric:15s}: {value:.6f}")
        print("="*50 + "\n")
        
        return metrics
    
    def plot_predictions(self, y_true, y_pred, save_path=None):
        print("Generating prediction plots...")
        
        plt.figure(figsize=(15, 6))
        
        plt.plot(y_true, label='Actual Prices', color='blue', linewidth=2)
        plt.plot(y_pred, label='Predicted Prices', color='red', linewidth=2, alpha=0.7)
        
        plt.title(f'Stock Price Prediction - {self.model_type.upper()} Model', fontsize=16, fontweight='bold')
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Normalized Price', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(Config.REPORTS_DIR, f'{self.model_type}_predictions.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction plot saved to {save_path}")
        plt.close()
    
    def plot_error_distribution(self, y_true, y_pred, save_path=None):
        print("Generating error distribution plot...")
        
        errors = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].hist(errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Error', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].scatter(y_pred, errors, alpha=0.5, color='coral')
        axes[1].axhline(y=0, color='black', linestyle='--', linewidth=2)
        axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted Values', fontsize=12)
        axes[1].set_ylabel('Residuals', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(Config.REPORTS_DIR, f'{self.model_type}_error_analysis.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error analysis plot saved to {save_path}")
        plt.close()
    
    def plot_training_history(self, save_path=None):
        print("Generating training history plots...")
        
        history_file = os.path.join(Config.MODELS_DIR, f'{self.model_type}_training_history.json')
        
        if not os.path.exists(history_file):
            print(f"Training history file not found: {history_file}")
            return
        
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss (MSE)', fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history['mae'], label='Training MAE', linewidth=2)
        axes[1].plot(history['val_mae'], label='Validation MAE', linewidth=2)
        axes[1].set_title('Model MAE', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(Config.REPORTS_DIR, f'{self.model_type}_training_history.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
        plt.close()
    
    def generate_report(self, metrics, save_path=None):
        print("Generating evaluation report...")
        
        report = f"""
{'='*70}
STOCK MARKET FORECASTING MODEL - EVALUATION REPORT
{'='*70}

Model Type: {self.model_type.upper()}
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*70}
PERFORMANCE METRICS
{'='*70}

Mean Squared Error (MSE):        {metrics['MSE']:.6f}
Root Mean Squared Error (RMSE):  {metrics['RMSE']:.6f}
Mean Absolute Error (MAE):       {metrics['MAE']:.6f}
R² Score:                        {metrics['R2_Score']:.6f}
Mean Absolute Percentage Error:  {metrics['MAPE']:.2f}%

{'='*70}
INTERPRETATION
{'='*70}

The model achieved an R² score of {metrics['R2_Score']:.4f}, indicating that 
{metrics['R2_Score']*100:.2f}% of the variance in stock prices is explained by the model.

The Mean Absolute Error (MAE) of {metrics['MAE']:.6f} represents the average 
absolute difference between predicted and actual normalized prices.

{'='*70}
LIMITATIONS AND RECOMMENDATIONS
{'='*70}

1. Stock market prediction is inherently uncertain due to unpredictable 
   market behavior and unforeseen events.

2. This model is trained on historical data and may not capture sudden 
   market changes or black swan events.

3. Model predictions should be used as one of many factors in investment 
   decisions, not as the sole basis for trading.

4. Continuous retraining with recent data is recommended to maintain 
   model relevance.

5. Consider ensemble methods combining multiple models for improved 
   robustness.

{'='*70}
"""
        
        if save_path is None:
            save_path = os.path.join(Config.REPORTS_DIR, f'{self.model_type}_evaluation_report.txt')
        
        with open(save_path, 'w') as f:
            f.write(report)
        
        print(f"Evaluation report saved to {save_path}")
        print(report)
    
    def evaluate(self):
        self.load_model()
        
        X_test, y_test = self.load_test_data()
        
        y_pred = self.make_predictions(X_test)
        
        self.predictions = y_pred
        self.actuals = y_test
        
        metrics = self.calculate_metrics(y_test, y_pred)
        
        self.plot_predictions(y_test, y_pred)
        self.plot_error_distribution(y_test, y_pred)
        self.plot_training_history()
        
        self.generate_report(metrics)
        
        return metrics

if __name__ == "__main__":
    import sys
    
    model_type = 'ann'
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    
    evaluator = ModelEvaluator(model_type=model_type)
    evaluator.evaluate()
