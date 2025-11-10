import React, { useState } from 'react';
import axios from 'axios';
import './StockSearch.css';

const API_URL = 'http://localhost:5000/api';

function StockSearch({ setStockData, setPredictions, setLoading }) {
  const [symbol, setSymbol] = useState('');
  const [error, setError] = useState('');

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!symbol) return;

    setLoading(true);
    setError('');
    setPredictions(null);

    try {
      const response = await axios.get(`${API_URL}/stock/${symbol.toUpperCase()}`);
      setStockData(response.data);
    } catch (err) {
      setError('Failed to fetch stock data. Please check the symbol and try again.');
      setStockData(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="stock-search">
      <form onSubmit={handleSearch}>
        <input
          type="text"
          placeholder="Enter stock symbol (e.g., AAPL, TSLA, GOOGL)"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
          className="search-input"
        />
        <button type="submit" className="search-button">
          Search
        </button>
      </form>
      {error && <div className="error-message">{error}</div>}
    </div>
  );
}

export default StockSearch;
