import os

import numpy as np
from datetime import timedelta, time
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

class StockData:
    def __init__(self, stock):
        self._stock = stock
        self._sec = yf.Ticker(self._stock.get_ticker())
        self._min_max = MinMaxScaler(feature_range=(0, 1))

    def __data_verification(self, train):
        print('mean:', train.mean(axis=0))
        print('max', train.max())
        print('min', train.min())
        print('Std dev:', train.std(axis=0))

    def get_stock_short_name(self):
        return self._sec.info['shortName']

    def get_min_max(self):
        return self._min_max

    def get_stock_currency(self):
        return self._sec.info['currency']

    def EMA(Close_arr, n):
        a = 2 / (n + 1)
        EMA_n = np.zeros(len(Close_arr))
        EMA_n[:n] = np.nan
    
        # Initialize the first EMA value
        EMA_n[n] = np.mean(Close_arr[:n])
    
        # Calculate EMA for the rest of the values
        for i in range(n + 1, len(Close_arr)):
            EMA_n[i] = (Close_arr[i] - EMA_n[i - 1]) * a + EMA_n[i - 1]
    
        return EMA_n
    
    def gains(Close_arr):
        gain_arr = np.diff(Close_arr)
        gain_arr[gain_arr < 0] = 0
        return gain_arr
    
    def losses(Close_arr):
        loss_arr = np.diff(Close_arr)
        loss_arr[loss_arr > 0] = 0
        return np.abs(loss_arr)

    def RSI(Close_arr, n=14):
        gain_arr = gains(Close_arr)
        loss_arr = losses(Close_arr)
    
        EMA_u = EMA(gain_arr, n)
        EMA_d = EMA(loss_arr, n)
    
        EMA_diff = EMA_u / EMA_d
    
        RSI_n = 100 - (100 / (1 + EMA_diff))
        RSI_n = np.concatenate((np.full(n, np.nan), RSI_n))  # Align lengths by padding initial values with NaN
        return RSI_n
        
    def download_transform_to_numpy(self, time_steps, project_folder):
        end_date = datetime.today()
        print('End Date: ' + end_date.strftime("%Y-%m-%d"))
        data = yf.download([self._stock.get_ticker()], interval='5m', start=self._stock.get_start_date(), end=end_date)
        # Reset index to access the datetime column
        data.index = data.index.tz_convert("Asia/Kolkata")
        data.reset_index(inplace=True)
        data.to_csv(os.path.join(project_folder, 'downloaded_data_'+self._stock.get_ticker()+'.csv'), index=False)
        #print(data)

        data=pd.read_csv(os.path.join(project_folder, 'downloaded_data_'+self._stock.get_ticker()+'.csv'))
        
        data['Close']=pd.to_numeric(data['Close'], errors='coerce')
        
        # # Step 2: Add Technical Indicators
        # data['Delta'] = data['Close'].diff()  # Delta (difference between consecutive Close prices)
        # data['RSI'] = RSIIndicator(close=data['Close'], window=14).rsi()
        
        # # Calculate MACD and Signal Line
        # macd = MACD(close=data['Close'], window_slow=26, window_fast=12, window_sign=9)
        # data['MACD'] = macd.macd()
        # data['MACD_signal'] = macd.macd_signal()
        
        
        # # Add Bollinger Bands
        # bollinger = BollingerBands(close=data['Close'], window=20)
        # data['BB_upper'] = bollinger.bollinger_hband()
        # data['BB_lower'] = bollinger.bollinger_lband()
        
        # # Add Lag Features
        # for lag in [1, 3, 5, 10]:
        #     data[f'Close_lag_{lag}'] = data['Close'].shift(lag)
        # data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
        # # Add Time-Based Features
        # data['Hour'] = data['Datetime'].dt.hour
        # data['Minute'] = data['Datetime'].dt.minute

        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        RS = gain / loss
        data['RSI'] = 100 - (100 / (1 + RS))
    
        # MACD (Moving Average Convergence Divergence)
        data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
        # Bollinger Bands
        data['20_MA'] = data['Close'].rolling(window=20).mean()
        data['20_SD'] = data['Close'].rolling(window=20).std()
        data['Upper_Band'] = data['20_MA'] + (2 * data['20_SD'])
        data['Lower_Band'] = data['20_MA'] - (2 * data['20_SD'])
    
        # Additional Indicators
        # RSI interpretation
        data['RSI_Text'] = ''
        data.loc[data['RSI'] < 30, 'RSI_Text'] = 'Readings below 30 generally indicate that the stock is oversold.'
        data.loc[data['RSI'] > 70, 'RSI_Text'] = 'Readings above 70 generally indicate that the stock is overbought.'
    
        # Bollinger Bands indication
        data['BB_Text'] = ''
        data.loc[data['Close'] > data['Upper_Band'], 'BB_Text'] = 'Price is above upper Bollinger Band, potentially overbought.'
        data.loc[data['Close'] < data['Lower_Band'], 'BB_Text'] = 'Price is below lower Bollinger Band, potentially oversold.'

        # Volume
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        # Use the closing prices to calculate RSI
        Close = data['Close'].values
        RSI14 = RSI(Close, n=14)
        
        # Ensure the lengths match by trimming the RSI array if necessary
        RSI14 = RSI14[:len(df)]
        
        # Adding the RSI to the dataframe
        df['RSI14'] = RSI14
        # Drop rows with NaN values
        data.dropna(inplace=True)
        data.to_csv(os.path.join(project_folder, 'data_'+self._stock.get_ticker()+'.csv'), index=False)
        training_data = data[data['Datetime'] < pd.Timestamp(self._stock.get_validation_date()).tz_localize('UTC')].copy()
        test_data = data[data['Datetime'] >= pd.Timestamp(self._stock.get_validation_date()).tz_localize('UTC')].copy()
        training_data = training_data.set_index('Datetime')
        # Set the data frame index using column Date
        test_data = test_data.set_index('Datetime')
        #print(test_data)

        # train_scaled = self._min_max.fit_transform(training_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Delta', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'Hour', 'Minute']])
        train_scaled = self._min_max.fit_transform(training_data[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'EMA12', 'EMA26', 'MACD', 'Signal_Line', '20_MA', '20_SD', 'Upper_Band', 'Lower_Band', 'RSI_Text', 'BB_Text', 'Volume_MA', 'RSI14']])
        self.__data_verification(train_scaled)

        # Training Data Transformation
        x_train = []
        y_train = []
        for i in range(time_steps, train_scaled.shape[0]):
            x_train.append(train_scaled[i - time_steps:i])
            y_train.append(train_scaled[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        total_data = pd.concat((training_data, test_data), axis=0)
        inputs = total_data[len(total_data) - len(test_data) - time_steps:]
        # test_scaled = self._min_max.fit_transform(inputs[['Open', 'High', 'Low', 'Close', 'Volume', 'Delta', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'Hour', 'Minute']])
        test_scaled = self._min_max.fit_transform(inputs[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'EMA12', 'EMA26', 'MACD', 'Signal_Line', '20_MA', '20_SD', 'Upper_Band', 'Lower_Band', 'RSI_Text', 'BB_Text', 'Volume_MA', 'RSI14']])

        # Testing Data Transformation
        x_test = []
        y_test = []
        for i in range(time_steps, test_scaled.shape[0]):
            x_test.append(test_scaled[i - time_steps:i])
            y_test.append(test_scaled[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return (x_train, y_train), (x_test, y_test), (training_data, test_data)

    def __date_range(self, start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    def __date_range_5_min(self, start_date, end_date):
        market_open = time(9, 15)  # 9:15 AM
        market_close = time(15, 30)  # 3:30 PM
    
        current_date = start_date
        while current_date <= end_date:
            # Generate intraday 5-minute intervals within trading hours
            current_time = datetime.combine(current_date.date(), market_open)
            while current_time.time() <= market_close:
                # Convert to pandas Timestamp with UTC timezone
                yield pd.Timestamp(current_time, tz='UTC')
                current_time += timedelta(minutes=5)
            
            # Move to the next day
            current_date += timedelta(days=1)

    def negative_positive_random(self):
        return 1 if random.random() < 0.5 else -1

    def pseudo_random(self):
        return random.uniform(0.01, 0.03)

    def generate_future_data(self, time_steps, min_max, start_date, end_date, latest_close_price, latest_open_price, latest_low_price, latest_high_price):
        x_future = []
        c_future = []
        o_future = []
        l_future = []
        h_future = []


        # We need to provide a randomisation algorithm for the close price
        # This is my own implementation and it will provide a variation of the
        # close price for a +-1-3% of the original value, when the value wants to go below
        # zero, it will be forced to go up.

        original_price = latest_close_price

        for single_date in self.__date_range_5_min(start_date, end_date):
            x_future.append(single_date)
            direction = self.negative_positive_random()
            random_slope = direction * (self.pseudo_random())
            #print(random_slope)
            original_price = original_price + (original_price * random_slope)
            latest_open_price = latest_open_price + (latest_open_price * random_slope)
            latest_low_price = latest_low_price + (latest_low_price * random_slope)
            latest_high_price = latest_high_price + (latest_high_price * random_slope)

            #print(original_price)
            if original_price < 0:
                original_price = 0
            if latest_open_price < 0:
                latest_open_price = 0
            if latest_low_price < 0:
                latest_low_price = 0
            if latest_high_price < 0:
                latest_high_price = 0

            c_future.append(original_price)
            h_future.append(latest_high_price)
            l_future.append(latest_low_price)
            o_future.append(latest_open_price)

        test_data = pd.DataFrame({'Datetime': x_future, 'Close': c_future, 'High': h_future, 'Low': l_future, 'Open': o_future})
        test_data = test_data.set_index('Datetime')

        test_scaled = min_max.fit_transform(test_data)
        x_test = []
        y_test = []
        #print(test_scaled.shape[0])
        for i in range(time_steps, test_scaled.shape[0]):
            x_test.append(test_scaled[i - time_steps:i])
            y_test.append(test_scaled[i, 0])
            #print(i - time_steps)

        x_test, y_test = np.array(x_test), np.array(y_test)
        # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return x_test, y_test, test_data



