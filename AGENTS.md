# AI Agent Instructions for NDAMBI Project

## Frequently Used Commands

### Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('brown')"

# Setup email authentication (optional)
copy .env.example .env
# Edit .env with your Gmail credentials
```

### Running the System

```bash
# Interactive CLI menu (recommended for beginners)
python main.py

# Streamlit dashboard
streamlit run app.py

# Individual scripts
python src/data_collection.py
python src/preprocessing.py
python src/train.py          # Train ANN
python src/train.py lstm     # Train LSTM
python src/evaluate.py ann   # Evaluate ANN
python src/evaluate.py lstm  # Evaluate LSTM
```

## Project Structure

- `src/` - Core source code modules
  - `data_collection.py` - Fetches stock data, news sentiment, fundamentals
  - `preprocessing.py` - Data cleaning, feature engineering, normalization
  - `model.py` - ANN and LSTM model architectures
  - `train.py` - Model training logic
  - `evaluate.py` - Model evaluation and metrics

- `data/` - Data storage
  - `raw/` - Raw collected data (CSV files)
  - `processed/` - Preprocessed data (NumPy arrays)

- `models/` - Trained models and training history
- `reports/` - Evaluation reports and visualizations
- `config.py` - Configuration settings
- `main.py` - CLI interface
- `app.py` - Streamlit dashboard

## Code Conventions

### Style Guidelines
- Use **snake_case** for function and variable names
- Use **PascalCase** for class names
- Add docstrings to all public functions and classes
- Type hints are encouraged but not required
- Maximum line length: 100 characters

### Imports Organization
```python
# Standard library
import os
import sys

# Third-party libraries
import numpy as np
import pandas as pd
import tensorflow as tf

# Local modules
from config import Config
from src.model import StockPredictionANN
```

### Error Handling
- Use try-except blocks for external API calls
- Provide informative error messages
- Log errors when appropriate

### Testing
- Test scripts can be run independently
- Each module has `if __name__ == "__main__":` block for testing
- Manual testing recommended due to external data dependencies

## Configuration

### Environment Variables (.env)
```
ALPHA_VANTAGE_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
RANDOM_SEED=42
DEFAULT_STOCK_SYMBOL=AAPL
```

### Model Hyperparameters (config.py)
```python
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 60
EARLY_STOPPING_PATIENCE = 10
```

## Data Flow

1. **Data Collection** → `data/raw/{SYMBOL}_raw_data.csv`
2. **Preprocessing** → `data/processed/{X,y}_{train,val,test}.npy`
3. **Training** → `models/stock_prediction_model.h5`
4. **Evaluation** → `reports/*.png`, `reports/*.txt`

## Dependencies

Key libraries:
- **TensorFlow 2.13+** - Deep learning
- **pandas** - Data manipulation
- **yfinance** - Stock data API
- **scikit-learn** - Preprocessing and metrics
- **Streamlit** - Dashboard
- **matplotlib/plotly** - Visualization

## Common Tasks

### Adding a New Feature
1. Modify `preprocessing.py` → `create_technical_indicators()`
2. Update feature selection if needed
3. Retrain models with new features

### Changing Model Architecture
1. Edit `model.py` → `build_model()` method
2. Adjust `layers_config` parameter
3. Retrain and evaluate

### Adding New Stock Symbol
1. Use CLI: `python main.py` → Option 1
2. Or directly: `DataCollector('SYMBOL').collect_all_data()`
3. Follow with preprocessing and training

### Debugging

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check data shapes
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Verify model summary
model.model.summary()
```

## Performance Optimization

- **Memory**: Reduce `BATCH_SIZE` if OOM errors occur
- **Training Speed**: Increase `BATCH_SIZE` if memory allows
- **Overfitting**: Increase dropout rates, add L2 regularization
- **Underfitting**: Increase model capacity (more layers/neurons)

## Typical Workflow

```bash
# 1. Collect data for a stock
python main.py  # Choose option 1, enter symbol (e.g., AAPL)

# 2. Preprocess the data
python main.py  # Choose option 2

# 3. Train model
python main.py  # Choose option 3 (ANN) or 4 (LSTM)

# 4. Evaluate model
python main.py  # Choose option 5 (ANN) or 6 (LSTM)

# 5. View results in dashboard
python main.py  # Choose option 8
```

## Security Notes

- Never commit `.env` file (contains API keys)
- Use `.env.example` as template
- Keep sensitive data in environment variables
- Sanitize user inputs if adding interactive features

## Future Enhancements

Ideas for expansion:
- Multi-stock portfolio prediction
- Real-time data streaming
- Automated retraining pipeline
- A/B testing framework
- Ensemble model combinations
- Advanced attention mechanisms
