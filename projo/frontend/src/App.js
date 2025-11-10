import React, { useState } from 'react';
import './App.css';
import StockSearch from './components/StockSearch';
import StockChart from './components/StockChart';
import PredictionPanel from './components/PredictionPanel';

function App() {
  const [stockData, setStockData] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);

  return (
    <div className="App">
      <header className="App-header">
        <h1>📈 Smart Stock</h1>
        <p>AI-Powered Stock Market Prediction</p>
      </header>
      
      <main className="App-main">
        <StockSearch 
          setStockData={setStockData}
          setPredictions={setPredictions}
          setLoading={setLoading}
        />
        
        {loading && <div className="loader">Loading...</div>}
        
        {stockData && (
          <StockChart 
            stockData={stockData}
            predictions={predictions}
          />
        )}
        
        {stockData && (
          <PredictionPanel 
            stockData={stockData}
            predictions={predictions}
            setPredictions={setPredictions}
            setLoading={setLoading}
          />
        )}
      </main>
    </div>
  );
}

export default App;
