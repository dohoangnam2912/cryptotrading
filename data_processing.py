import ccxt
import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor

class CryptoDataPipeline:
    """
    A class to fetch and preprocess cryptocurrency OHLCV data.
    
    Attributes:
        symbols (list): A list of symbols to fetch data for (e.g., ['BTC/USDT', 'ETH/USDT']).
        save_dir (str): The directory where the fetched CSV files will be saved.
        sequence_length (int): The number of timesteps in each sequence for preprocessing.
    
    Methods:
        fetch_ohlcv(symbol):
            Fetches OHLCV data for a specific symbol and saves it to a CSV file.
        
        fetch_all_data():
            Fetches OHLCV data for all symbols in parallel.
        
        preprocess(file_path):
            Preprocesses the data by calculating log changes and creating sequences.
        
        preprocess_all():
            Preprocesses the data for all symbols and returns the sequences.
    """
    def __init__(self, symbols, save_dir, limit, sequence_length=60):
        self.symbols = symbols
        self.save_dir = save_dir
        self.sequence_length = sequence_length
        self.limit = limit
        self.symbol_data = {}

    def download_data(self, symbol):
        if os.path.exists(f'{self.save_dir}/{symbol.replace("/","_")}_ohlcv.csv'):
            print(f"{symbol} data is already in the folder. Loading the data...")
            df = pd.read_csv(f'{self.save_dir}/{symbol.replace("/","_")}_ohlcv.csv')
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            print("Data loaded.")
        else:
            print(f"{symbol} data is not in the folder. Downloading the data...")
            exchange = ccxt.binance()
            timeframe = '1h'

            limit = 1000 # Maximum limit per request
            since = exchange.parse8601('2019-01-01T00:00:00Z')
            all_candles = []

            while since < exchange.milliseconds():
                try:
                    # Fetch OHLCV data
                    candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                    if not candles:
                        break

                    all_candles += candles

                    # Update 'since' to the timestamp of the last fetched candles
                    since = candles[-1][0] + 1

                    # Sleep to avoid hitting the rate limit
                    time.sleep(exchange.rateLimit / 1000)

                except Exception as e:
                    print(f"Error: {e}")
                    break

            df = pd.DataFrame(all_candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])

            # Convert for readable
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')

            # Save to CSV
            df.to_csv(f'{self.save_dir}/{symbol.replace("/","_")}_ohlcv.csv', index=False)
            # print(f"Successfully fetched {symbol} data. Saved to {self.save_dir}/{symbol.replace("/","_")}_ohlcv.csv")
            print(f"Finishing download and loaded {symbol} data.")
    
    # # Function to fetch OHCLV data for a specific symbol
    # def fetch_ohlcv(self, symbol):
    #     exchange = ccxt.binance()
    #     bars = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=self.limit)
    #     df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    #     df.to_csv(f'{self.save_dir}/{symbol.replace("/","_")}_ohlcv.csv', index=False)
    #     print(f"Successfully fetched {symbol} data. Saved to {self.save_dir}/{symbol.replace("/","_")}_ohlcv.csv")
    #     return df

    # Method to fetch data for all symbols in parallel
    def fetch_all_data(self):
        with ThreadPoolExecutor() as executor:
            dataframes = list(executor.map(self.download_data, self.symbols))
        self.symbol_data = {self.symbols[i]: dataframes[i] for i in range(len(self.symbols))}
        return self.symbol_data

    # Method for preprocessing data
    def preprocess(self, file_path):
        """
        This is the function for preprocessing the data.

        Args:
            file_path (str): Data file directions

        Returns:
            pd.DataFrame : A dataframe containing preprocessed value
        """
        # Load OHCLV data from CSV
        df = pd.read_csv(file_path)

        # Normalize OHCL data (logarithmic change)
        df['log_open'] = np.log(df['open']) - np.log(df['open'].shift(1))
        df['log_high'] = np.log(df['high']) - np.log(df['high'].shift(1))
        df['log_low'] = np.log(df['low']) - np.log(df['low'].shift(1))
        df['log_close'] = np.log(df['close']) - np.log(df['close'].shift(1))

        # Drop the first row with NaN values after shifting
        df.dropna(inplace=True)

        # Normalize the volume column separately
        scaler = MinMaxScaler(feature_range=(0, 1))
        df['volume'] = scaler.fit_transform(df[['volume']])

        # Select only the log-transformed price data and volume for the sequences
        log_data = df[['log_open', 'log_high', 'log_low', 'log_close', 'volume']].values

        # Create sequences (sliding windows)
        def create_sequences(data, seq_length):
            sequences = []
            for i in range(len(data) - seq_length):
                sequences.append(data[i: i + seq_length])
            return np.array(sequences)

        sequences = create_sequences(log_data, self.sequence_length)
        return sequences

    # Method to preprocess all symbol data
    def preprocess_all(self):
        all_sequences = {}
        for symbol in self.symbols:
            file_path = f'{self.save_dir}/{symbol}_ohlcv.csv'
            sequences = self.preprocess(file_path)
            all_sequences[symbol] = sequences
        print(all_sequences)
        return all_sequences


if __name__ == "__main__":
    dataPipelineArgs = {
        "symbols": ["BTC/USDT", "ETH/USDT"],
        "save_dir": "./src/data/raw_data",
        "sequence_length": 60,
        "limit": 1000
    }

    # Create an instance of the pipeline and fetch data
    pipeline = CryptoDataPipeline(**dataPipelineArgs)

    # Fetch all data
    pipeline.fetch_all_data()


