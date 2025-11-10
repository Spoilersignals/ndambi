# Smart Stock - Stock Market Prediction Using Artificial Neural Network

AI-powered stock market prediction web application using LSTM neural networks.

## Features

- **Real-time Stock Data**: Fetch historical stock data for analysis
- **LSTM Neural Network**: Deep learning model for price prediction
- **Performance Metrics**: MSE, MAE, RMSE evaluation metrics
- **Interactive Charts**: Visualize historical prices and predictions
- **Demo Mode**: Realistic generated data for testing

## Tech Stack

### Backend
- Python 3.10
- Flask (REST API)
- TensorFlow/Keras (LSTM Model)
- yfinance (Stock Data)
- Pandas & NumPy (Data Processing)

### Frontend
- React 18
- Recharts (Data Visualization)
- Axios (HTTP Client)

## Installation

### Prerequisites
- Python 3.10 or higher
- Node.js 16 or higher
- pip and npm

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Flask server:
```bash
python app.py
```

The backend will start on `http://127.0.0.1:5000`

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install Node dependencies:
```bash
npm install
```

3. Start the React development server:
```bash
npm start
```

The frontend will start on `http://localhost:3000`

## Usage

1. Open your browser and go to `http://localhost:3000`
2. Enter a stock symbol (e.g., AAPL, GOOGL, MSFT, AMZN, TSLA)
3. Click **Search** to view historical data
4. Click **Predict** to generate future price predictions
5. Click **Train Model** to retrain the LSTM model (optional)

### Available Stock Symbols (Demo Mode)
- AAPL (Apple)
- GOOGL (Google)
- MSFT (Microsoft)
- AMZN (Amazon)
- TSLA (Tesla)
- META (Meta)
- NVDA (NVIDIA)
- JPM (JP Morgan)
- V (Visa)
- WMT (Walmart)

## Configuration

### Demo Mode
By default, the app runs in demo mode with generated data. To use real Yahoo Finance data:

**Windows:**
```bash
set DEMO_MODE=false
```

**Linux/Mac:**
```bash
export DEMO_MODE=false
```

Then restart the backend server.

## Testing

### Backend Tests
```bash
cd backend
python -m pytest -v
```

### Frontend Tests
```bash
cd frontend
npm test
```

## API Endpoints

### GET `/api/stock/<symbol>`
Fetch historical stock data
- **Parameters**: `period` (optional, default: 1y)
- **Response**: JSON with dates, prices, volume

### POST `/api/predict/<symbol>`
Generate price predictions
- **Body**: `{"days": 30}` (optional, default: 30)
- **Response**: JSON with predictions, dates, metrics

### POST `/api/train/<symbol>`
Train LSTM model on stock data
- **Response**: JSON with training metrics and loss

## Project Structure

```
projo/
├── backend/
│   ├── app.py              # Flask API server
│   ├── model.py            # LSTM model implementation
│   ├── demo_data.py        # Demo data generator
│   ├── test_api.py         # API tests
│   ├── test_model.py       # Model tests
│   └── requirements.txt    # Python dependencies
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── StockSearch.js
│   │   │   ├── StockChart.js
│   │   │   └── PredictionPanel.js
│   │   ├── App.js
│   │   └── index.js
│   └── package.json        # Node dependencies
│
└── README.md
```

## Performance Metrics

The model evaluates predictions using:
- **MSE** (Mean Squared Error): Average squared difference between predicted and actual values
- **MAE** (Mean Absolute Error): Average absolute difference
- **RMSE** (Root Mean Squared Error): Square root of MSE

## Research Objectives

This application implements the following research objectives:
1. ✅ Data collection and preprocessing from stock market sources
2. ✅ LSTM neural network architecture for time series prediction
3. ✅ Model training and hyperparameter optimization
4. ✅ Performance evaluation with MSE, MAE, RMSE metrics
5. ✅ Web-based visualization and prediction interface

## License

This project is developed for academic research purposes.

## Authors

Smart Stock Research Team

## Troubleshooting

### Yahoo Finance Connection Issues
If you encounter errors fetching real stock data, the app automatically falls back to demo mode. You can also manually enable demo mode by setting `DEMO_MODE=true`.

### TensorFlow Warnings
TensorFlow may show warnings about CPU optimization. These are informational and do not affect functionality.

### Port Already in Use
If port 5000 or 3000 is already in use:
- **Backend**: Edit `app.py` and change the port in `app.run(port=5000)`
- **Frontend**: Set `PORT=3001` before running `npm start`
