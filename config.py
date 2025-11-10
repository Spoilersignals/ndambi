import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
    
    RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))
    TRAIN_TEST_SPLIT = float(os.getenv('TRAIN_TEST_SPLIT', 0.8))
    VALIDATION_SPLIT = float(os.getenv('VALIDATION_SPLIT', 0.2))
    
    DEFAULT_STOCK_SYMBOL = os.getenv('DEFAULT_STOCK_SYMBOL', 'AAPL')
    
    DATA_DIR = 'data'
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    MODELS_DIR = 'models'
    REPORTS_DIR = 'reports'
    
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 10
    
    SEQUENCE_LENGTH = 60
    PREDICTION_DAYS = 1
