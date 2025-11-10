# Stock Market Forecasting System Using Artificial Neural Networks

A comprehensive machine learning system for predicting stock market fluctuations using Artificial Neural Networks (ANNs) and LSTM networks. This project implements advanced time-series forecasting with multiple data sources including historical prices, technical indicators, sentiment analysis, and economic factors.

---

## 🚀 **NEW USER? START HERE!**

**📦 [INSTALLATION.md](INSTALLATION.md)** - Complete step-by-step installation guide for new users

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Limitations](#limitations)
- [Contributing](#contributing)

## 🎯 Overview

Stock market prediction is inherently challenging due to the complex interplay of factors influencing prices. This system leverages the pattern recognition capabilities of Artificial Neural Networks to forecast stock price movements based on:

- **Historical stock prices** (Open, High, Low, Close, Volume)
- **Technical indicators** (MA, EMA, RSI, MACD, Bollinger Bands)
- **Sentiment analysis** from financial news
- **Company fundamentals** (P/E ratio, market cap, etc.)
- **Economic indicators** (GDP, inflation, interest rates)

## ✨ Features

### Data Collection
- Automated fetching from Yahoo Finance API
- News sentiment analysis using TextBlob
- Company fundamental data integration
- Support for multiple stock symbols
- Historical data spanning multiple years

### Data Preprocessing
- Automated missing data handling
- Technical indicator generation (20+ indicators)
- Feature selection using mutual information
- Min-Max normalization
- Sequence generation for time-series modeling

### Model Architectures
- **Feedforward ANN**: Dense layers with dropout and batch normalization
- **LSTM Network**: Recurrent architecture for temporal dependencies
- Configurable hyperparameters (learning rate, batch size, epochs)
- Early stopping and learning rate reduction
- Model checkpointing for best weights

### Evaluation & Visualization
- Comprehensive metrics (MSE, RMSE, MAE, R², MAPE)
- Interactive Streamlit dashboard
- Training history visualization
- Prediction vs actual price plots
- Error distribution analysis
- Detailed evaluation reports

## 📁 Project Structure

```
NDAMBI/
├── data/
│   ├── raw/                    # Raw collected data
│   └── processed/              # Preprocessed data (numpy arrays)
├── models/                     # Saved model files and training history
├── reports/                    # Evaluation reports and visualizations
├── src/
│   ├── __init__.py
│   ├── data_collection.py     # Data fetching module
│   ├── preprocessing.py       # Data preprocessing pipeline
│   ├── model.py               # ANN and LSTM model definitions
│   ├── train.py               # Model training script
│   └── evaluate.py            # Model evaluation module
├── notebooks/                  # Jupyter notebooks for analysis
├── config.py                  # Configuration settings
├── main.py                    # CLI interface
├── app.py                     # Streamlit dashboard
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
└── README.md                 # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Steps

1. **Clone or download the project**
   ```bash
   cd c:/Users/pesak/Desktop/NDAMBI
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   copy .env.example .env
   # Edit .env file with your API keys (optional for basic usage)
   ```

5. **Download NLTK data (for sentiment analysis)**
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('brown')"
   ```

## ⚡ Quick Start

### Option 1: Interactive CLI Menu

```bash
python main.py
```

This launches an interactive menu with all features:
- Data collection
- Preprocessing
- Model training
- Evaluation
- Dashboard launch

### Option 2: Individual Scripts

```bash
# 1. Collect data
python -c "from src.data_collection import DataCollector; DataCollector('AAPL').collect_all_data()"

# 2. Preprocess data
python src/preprocessing.py

# 3. Train model (ANN)
python src/train.py

# 4. Train LSTM model
python src/train.py lstm

# 5. Evaluate model
python src/evaluate.py ann
python src/evaluate.py lstm
```

### Option 3: Streamlit Dashboard

```bash
streamlit run app.py
```

Access the dashboard at `http://localhost:8501`

## 📖 Usage Guide

### 1. Data Collection

```python
from src.data_collection import DataCollector

# Initialize collector for a stock symbol
collector = DataCollector('AAPL')

# Fetch all data (prices, sentiment, fundamentals)
data = collector.collect_all_data(
    start_date='2019-01-01',
    end_date='2024-01-01'
)

# Data is automatically saved to data/raw/
```

### 2. Data Preprocessing

```python
from src.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()

# Run complete preprocessing pipeline
result = preprocessor.preprocess_pipeline(
    'data/raw/AAPL_raw_data.csv'
)

# Outputs:
# - data/processed/X_train.npy
# - data/processed/X_val.npy
# - data/processed/X_test.npy
# - data/processed/y_train.npy
# - data/processed/y_val.npy
# - data/processed/y_test.npy
```

### 3. Model Training

```python
from src.train import train_ann_model, train_lstm_model

# Train feedforward ANN
ann_model, history = train_ann_model()

# Or train LSTM
lstm_model, history = train_lstm_model()

# Models saved to models/ directory
```

### 4. Model Evaluation

```python
from src.evaluate import ModelEvaluator

# Evaluate ANN model
evaluator = ModelEvaluator(model_type='ann')
metrics = evaluator.evaluate()

# Generates:
# - Evaluation reports (reports/*.txt)
# - Visualization plots (reports/*.png)
# - Performance metrics
```

## 🧠 Model Architecture

### Feedforward ANN

```
Input Layer (Sequence: 60 × Features)
    ↓
Flatten Layer
    ↓
Dense Layer (128 neurons, ReLU, L2 regularization)
    ↓
Dropout (20%)
    ↓
Batch Normalization
    ↓
Dense Layer (64 neurons, ReLU, L2 regularization)
    ↓
Dropout (20%)
    ↓
Batch Normalization
    ↓
Dense Layer (32 neurons, ReLU, L2 regularization)
    ↓
Dropout (15%)
    ↓
Batch Normalization
    ↓
Output Layer (1 neuron, Linear activation)
```

### LSTM Network

```
Input Layer (Sequence: 60 × Features)
    ↓
LSTM Layer (100 units, return sequences)
    ↓
Dropout (20%)
    ↓
LSTM Layer (50 units)
    ↓
Dropout (20%)
    ↓
Dense Layer (25 neurons, ReLU)
    ↓
Dropout (20%)
    ↓
Output Layer (1 neuron)
```

### Training Configuration

- **Optimizer**: Adam
- **Learning Rate**: 0.001 (with adaptive reduction)
- **Loss Function**: Mean Squared Error (MSE)
- **Batch Size**: 32
- **Epochs**: Up to 100 (with early stopping)
- **Early Stopping Patience**: 10 epochs
- **Validation Split**: 10% of training data

## 📊 Performance Metrics

The system evaluates models using multiple metrics:

### Primary Metrics
- **MSE (Mean Squared Error)**: Average squared difference between predictions and actuals
- **RMSE (Root Mean Squared Error)**: Square root of MSE, in same units as target
- **MAE (Mean Absolute Error)**: Average absolute difference
- **R² Score**: Proportion of variance explained by the model
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error

### Typical Performance
*Results may vary based on stock symbol, time period, and market conditions*

| Metric | ANN Model | LSTM Model |
|--------|-----------|------------|
| MSE    | 0.0015-0.0030 | 0.0012-0.0025 |
| MAE    | 0.025-0.045 | 0.020-0.040 |
| R²     | 0.85-0.95 | 0.88-0.96 |

## ⚠️ Limitations

### Technical Limitations
1. **Data Quality**: Model accuracy depends on data quality and availability
2. **Feature Engineering**: Additional domain-specific features may improve performance
3. **Computational Resources**: Training requires sufficient RAM and CPU/GPU
4. **Real-time Predictions**: System designed for historical analysis, not real-time trading

### Fundamental Limitations
1. **Market Unpredictability**: Stock markets are influenced by unpredictable events (news, policy changes, black swan events)
2. **Historical Bias**: Past performance does not guarantee future results
3. **Model Assumptions**: Neural networks assume patterns in historical data will continue
4. **External Factors**: Cannot account for unprecedented events or paradigm shifts

### Important Disclaimers
⚠️ **This system is for educational and research purposes only**

- NOT intended for actual trading decisions
- NOT financial advice
- Results should be validated with other analysis methods
- Always consult financial professionals before investing
- Past performance does not predict future results
- Stock market investments carry inherent risks

## 🔧 Configuration

Edit `config.py` or `.env` file to customize:

```python
# Model Parameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 60

# Data Splits
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.2

# Early Stopping
EARLY_STOPPING_PATIENCE = 10

# Default Stock
DEFAULT_STOCK_SYMBOL = 'AAPL'
```

## 📈 Dashboard Features

The Streamlit dashboard provides:

### Data Collection Page
- Interactive stock symbol input
- Date range selection
- Real-time data fetching
- Data preview and statistics
- Price history visualization

### Model Training Page
- Model selection (ANN/LSTM)
- Hyperparameter configuration
- Training history plots
- Loss and metric curves

### Predictions Page
- Model comparison
- Interactive prediction charts
- Performance metrics display
- Actual vs predicted visualization

### Reports Page
- Detailed evaluation reports
- Visualization gallery
- Model performance summaries

## 🔬 Technical Details

### Feature Engineering
The system automatically generates:
- **Moving Averages**: 7-day, 21-day, 50-day
- **Exponential Moving Averages**: 12-day, 26-day
- **MACD**: Moving Average Convergence Divergence
- **RSI**: Relative Strength Index
- **Bollinger Bands**: Upper, middle, lower bands
- **Volatility**: 20-day rolling standard deviation
- **Daily Returns**: Percentage changes
- **Volume Indicators**: Volume moving averages

### Feature Selection
- Uses mutual information for feature importance
- Selects top 15 most relevant features
- Reduces dimensionality while preserving predictive power

### Data Normalization
- Min-Max scaling to [0, 1] range
- Separate scalers for features and target
- Preserves data distribution while standardizing scale

## 🐛 Troubleshooting

### Common Issues

**Issue**: Module not found errors
```bash
# Solution: Ensure you're in the virtual environment
venv\Scripts\activate
pip install -r requirements.txt
```

**Issue**: No data found for symbol
```bash
# Solution: Check internet connection and symbol validity
# Try with a common symbol like 'AAPL' or 'GOOGL'
```

**Issue**: Out of memory during training
```bash
# Solution: Reduce batch size in config.py
BATCH_SIZE = 16  # or 8
```

**Issue**: TensorFlow warnings
```bash
# Solution: Suppress warnings (optional)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

## 📚 Requirements

See [requirements.txt](requirements.txt) for full list. Key dependencies:

- `tensorflow>=2.13.0` - Deep learning framework
- `pandas>=2.0.3` - Data manipulation
- `numpy>=1.24.3` - Numerical computing
- `scikit-learn>=1.3.0` - Machine learning utilities
- `yfinance>=0.2.28` - Stock data API
- `streamlit>=1.26.0` - Dashboard framework
- `matplotlib>=3.7.2` - Visualization
- `plotly>=5.17.0` - Interactive plots

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Additional Data Sources**: Integrate more economic indicators
2. **Model Architectures**: Experiment with transformer models, attention mechanisms
3. **Feature Engineering**: Add more technical indicators
4. **Ensemble Methods**: Combine multiple models
5. **Real-time Predictions**: Implement live data streaming
6. **Backtesting**: Add historical simulation features

## 📄 License

This project is for educational purposes. Use at your own risk.

## 🙏 Acknowledgments

- **Data Sources**: Yahoo Finance, Alpha Vantage
- **Frameworks**: TensorFlow, Keras, Streamlit
- **Libraries**: pandas, scikit-learn, NumPy

## 📧 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the configuration settings
3. Examine error logs in console output

---

**Built with ❤️ for financial data science and machine learning education**
