import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
import os
from config import Config

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.config = Config()
        self.selected_features = None
        
    def load_data(self, filepath):
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        print(f"Loaded {len(df)} records")
        return df
    
    def handle_missing_data(self, df):
        print("Handling missing data...")
        initial_rows = len(df)
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mean(), inplace=True)
        
        df = df.dropna()
        
        print(f"Removed {initial_rows - len(df)} rows with missing values")
        return df
    
    def create_technical_indicators(self, df):
        print("Creating technical indicators...")
        
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_21'] = df['Close'].rolling(window=21).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
        
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        df = df.dropna()
        print(f"Created technical indicators. Remaining rows: {len(df)}")
        return df
    
    def feature_selection(self, df, target_col='Close', n_features=15):
        print(f"Performing feature selection (selecting top {n_features} features)...")
        
        feature_cols = [col for col in df.columns if col not in ['Date', 'Symbol', target_col]]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        mi_scores = mutual_info_regression(X, y, random_state=Config.RANDOM_SEED)
        
        mi_scores_series = pd.Series(mi_scores, index=feature_cols)
        mi_scores_series = mi_scores_series.sort_values(ascending=False)
        
        self.selected_features = mi_scores_series.head(n_features).index.tolist()
        
        print(f"Selected features: {self.selected_features}")
        return self.selected_features
    
    def normalize_data(self, df, feature_cols):
        print("Normalizing data...")
        
        df_normalized = df.copy()
        
        df_normalized[feature_cols] = self.feature_scaler.fit_transform(df[feature_cols])
        
        if 'Close' in df.columns:
            df_normalized[['Close']] = self.scaler.fit_transform(df[['Close']])
        
        return df_normalized
    
    def create_sequences(self, data, target_col='Close', sequence_length=60):
        print(f"Creating sequences with length {sequence_length}...")
        
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i][data.columns.get_loc(target_col)])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Created {len(X)} sequences")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y
    
    def split_data(self, X, y, train_ratio=0.8, val_ratio=0.1):
        print("Splitting data into train, validation, and test sets...")
        
        train_size = int(len(X) * train_ratio)
        val_size = int(len(X) * val_ratio)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess_pipeline(self, input_file, output_dir=Config.PROCESSED_DATA_DIR):
        df = self.load_data(input_file)
        
        df = self.handle_missing_data(df)
        
        df = self.create_technical_indicators(df)
        
        selected_features = self.feature_selection(df, n_features=15)
        
        df_normalized = self.normalize_data(df, selected_features)
        
        sequence_cols = selected_features + ['Close']
        X, y = self.create_sequences(
            df_normalized[sequence_cols], 
            sequence_length=Config.SEQUENCE_LENGTH
        )
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
        
        print(f"Preprocessed data saved to {output_dir}")
        
        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'selected_features': selected_features
        }

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    input_file = os.path.join(Config.RAW_DATA_DIR, 'AAPL_raw_data.csv')
    
    if os.path.exists(input_file):
        result = preprocessor.preprocess_pipeline(input_file)
        print("\nPreprocessing complete!")
    else:
        print(f"Error: {input_file} not found. Run data_collection.py first.")
