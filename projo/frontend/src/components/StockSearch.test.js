import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import axios from 'axios';
import StockSearch from './StockSearch';

jest.mock('axios');

describe('StockSearch Component', () => {
  const mockSetStockData = jest.fn();
  const mockSetPredictions = jest.fn();
  const mockSetLoading = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders search input and button', () => {
    render(
      <StockSearch
        setStockData={mockSetStockData}
        setPredictions={mockSetPredictions}
        setLoading={mockSetLoading}
      />
    );

    expect(screen.getByPlaceholderText(/enter stock symbol/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /search/i })).toBeInTheDocument();
  });

  test('updates input value on change', () => {
    render(
      <StockSearch
        setStockData={mockSetStockData}
        setPredictions={mockSetPredictions}
        setLoading={mockSetLoading}
      />
    );

    const input = screen.getByPlaceholderText(/enter stock symbol/i);
    fireEvent.change(input, { target: { value: 'AAPL' } });

    expect(input.value).toBe('AAPL');
  });

  test('calls API on form submit with valid symbol', async () => {
    const mockData = {
      dates: ['2023-01-01', '2023-01-02'],
      prices: [150, 155],
      volume: [1000, 1100],
      symbol: 'AAPL'
    };

    axios.get.mockResolvedValue({ data: mockData });

    render(
      <StockSearch
        setStockData={mockSetStockData}
        setPredictions={mockSetPredictions}
        setLoading={mockSetLoading}
      />
    );

    const input = screen.getByPlaceholderText(/enter stock symbol/i);
    const button = screen.getByRole('button', { name: /search/i });

    fireEvent.change(input, { target: { value: 'AAPL' } });
    fireEvent.click(button);

    await waitFor(() => {
      expect(mockSetLoading).toHaveBeenCalledWith(true);
      expect(mockSetStockData).toHaveBeenCalledWith(mockData);
      expect(mockSetLoading).toHaveBeenCalledWith(false);
    });
  });

  test('displays error message on API failure', async () => {
    axios.get.mockRejectedValue(new Error('Network error'));

    render(
      <StockSearch
        setStockData={mockSetStockData}
        setPredictions={mockSetPredictions}
        setLoading={mockSetLoading}
      />
    );

    const input = screen.getByPlaceholderText(/enter stock symbol/i);
    const button = screen.getByRole('button', { name: /search/i });

    fireEvent.change(input, { target: { value: 'INVALID' } });
    fireEvent.click(button);

    await waitFor(() => {
      expect(screen.getByText(/failed to fetch stock data/i)).toBeInTheDocument();
    });
  });
});
