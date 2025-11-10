import React, { useState } from 'react';
import axios from 'axios';
import './PredictionPanel.css';

const API_URL = 'http://localhost:5000/api';

function PredictionPanel({ stockData, predictions, setPredictions, setLoading }) {
  const [days, setDays] = useState(30);
  const [error, setError] = useState('');

  const handlePredict = async () => {
    setLoading(true);
    setError('');

    try {
      const response = await axios.post(`${API_URL}/predict/${stockData.symbol}`, { days });
      setPredictions(response.data);
    } catch (err) {
      setError('Failed to generate predictions. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleTrain = async () => {
    setLoading(true);
    setError('');

    try {
      await axios.post(`${API_URL}/train/${stockData.symbol}`);
      alert('Model trained successfully!');
    } catch (err) {
      setError('Failed to train model. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="prediction-panel">
      <h2>Prediction Controls</h2>
      
      <div className="control-group">
        <label>Prediction Days:</label>
        <input
          type="number"
          min="1"
          max="90"
          value={days}
          onChange={(e) => setDays(parseInt(e.target.value))}
          className="days-input"
        />
      </div>

      <div className="button-group">
        <button onClick={handlePredict} className="predict-button">
          Generate Prediction
        </button>
        <button onClick={handleTrain} className="train-button">
          Train Model
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      {predictions && (
        <div className="prediction-stats">
          <h3>Prediction Summary</h3>
          <p><strong>Symbol:</strong> {predictions.symbol}</p>
          <p><strong>Predicted Days:</strong> {predictions.predictions.length}</p>
          <p><strong>Last Predicted Price:</strong> ${predictions.predictions[predictions.predictions.length - 1].toFixed(2)}</p>
          
          {predictions.metrics && (
            <div className="metrics-section">
              <h4>Model Performance Metrics</h4>
              <div className="metrics-grid">
                <div className="metric-card">
                  <span className="metric-label">MSE</span>
                  <span className="metric-value">{predictions.metrics.mse.toFixed(4)}</span>
                </div>
                <div className="metric-card">
                  <span className="metric-label">MAE</span>
                  <span className="metric-value">{predictions.metrics.mae.toFixed(4)}</span>
                </div>
                <div className="metric-card">
                  <span className="metric-label">RMSE</span>
                  <span className="metric-value">{predictions.metrics.rmse.toFixed(4)}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default PredictionPanel;
