import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import axios from 'axios';
import PredictionPanel from './PredictionPanel';

jest.mock('axios');

describe('PredictionPanel Component', () => {
  const mockStockData = {
    symbol: 'AAPL',
    dates: ['2023-01-01', '2023-01-02'],
    prices: [150, 155]
  };

  const mockSetPredictions = jest.fn();
  const mockSetLoading = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders prediction controls', () => {
    render(
      <PredictionPanel
        stockData={mockStockData}
        predictions={null}
        setPredictions={mockSetPredictions}
        setLoading={mockSetLoading}
      />
    );

    expect(screen.getByText(/prediction controls/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/prediction days/i)).toBeInTheDocument();
    expect(screen.getByText(/generate prediction/i)).toBeInTheDocument();
    expect(screen.getByText(/train model/i)).toBeInTheDocument();
  });

  test('updates days input value', () => {
    render(
      <PredictionPanel
        stockData={mockStockData}
        predictions={null}
        setPredictions={mockSetPredictions}
        setLoading={mockSetLoading}
      />
    );

    const input = screen.getByLabelText(/prediction days/i);
    fireEvent.change(input, { target: { value: '60' } });

    expect(input.value).toBe('60');
  });

  test('generates predictions on button click', async () => {
    const mockPredictions = {
      predictions: [160, 162, 165],
      dates: ['2023-01-03', '2023-01-04', '2023-01-05'],
      symbol: 'AAPL',
      metrics: { mse: 1.23, mae: 0.95, rmse: 1.11 }
    };

    axios.post.mockResolvedValue({ data: mockPredictions });

    render(
      <PredictionPanel
        stockData={mockStockData}
        predictions={null}
        setPredictions={mockSetPredictions}
        setLoading={mockSetLoading}
      />
    );

    const button = screen.getByText(/generate prediction/i);
    fireEvent.click(button);

    await waitFor(() => {
      expect(mockSetLoading).toHaveBeenCalledWith(true);
      expect(mockSetPredictions).toHaveBeenCalledWith(mockPredictions);
      expect(mockSetLoading).toHaveBeenCalledWith(false);
    });
  });

  test('displays prediction summary with metrics', () => {
    const mockPredictions = {
      predictions: [160, 162, 165],
      dates: ['2023-01-03', '2023-01-04', '2023-01-05'],
      symbol: 'AAPL',
      metrics: { mse: 1.23, mae: 0.95, rmse: 1.11 }
    };

    render(
      <PredictionPanel
        stockData={mockStockData}
        predictions={mockPredictions}
        setPredictions={mockSetPredictions}
        setLoading={mockSetLoading}
      />
    );

    expect(screen.getByText(/prediction summary/i)).toBeInTheDocument();
    expect(screen.getByText(/AAPL/i)).toBeInTheDocument();
    expect(screen.getByText(/model performance metrics/i)).toBeInTheDocument();
    expect(screen.getByText(/1.2300/)).toBeInTheDocument();
    expect(screen.getByText(/0.9500/)).toBeInTheDocument();
  });
});
