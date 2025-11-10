import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import './StockChart.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function StockChart({ stockData, predictions }) {
  if (!stockData) return null;

  const chartData = {
    labels: predictions 
      ? [...stockData.dates, ...predictions.dates]
      : stockData.dates,
    datasets: [
      {
        label: 'Historical Price',
        data: stockData.prices,
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.1,
      },
      ...(predictions ? [{
        label: 'Predicted Price',
        data: [...Array(stockData.prices.length).fill(null), ...predictions.predictions],
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        borderDash: [5, 5],
        tension: 0.1,
      }] : [])
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: 'white',
          font: { size: 14 }
        }
      },
      title: {
        display: true,
        text: `${stockData.symbol} Stock Price`,
        color: 'white',
        font: { size: 18 }
      },
    },
    scales: {
      x: {
        ticks: { color: 'white' },
        grid: { color: 'rgba(255, 255, 255, 0.1)' }
      },
      y: {
        ticks: { color: 'white' },
        grid: { color: 'rgba(255, 255, 255, 0.1)' }
      }
    }
  };

  return (
    <div className="stock-chart">
      <Line data={chartData} options={options} />
    </div>
  );
}

export default StockChart;
