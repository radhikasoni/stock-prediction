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

    def download_transform_to_numpy(self, time_steps, project_folder):
        end_date = datetime.today()
        print('End Date: ' + end_date.strftime("%Y-%m-%d"))
        data = yf.download([self._stock.get_ticker()], interval='5m', start=self._stock.get_start_date(), end=end_date)
        # Reset index to access the datetime column
        data.reset_index(inplace=True)
        data.to_csv(os.path.join(project_folder, 'downloaded_data_'+self._stock.get_ticker()+'.csv'))
        #print(data)

        training_data = data[data['Datetime'] < pd.Timestamp(self._stock.get_validation_date()).tz_localize('UTC')][['Datetime', 'Close', 'High', 'Low', 'Open']].copy()
        test_data = data[data['Datetime'] >= pd.Timestamp(self._stock.get_validation_date()).tz_localize('UTC')][['Datetime', 'Close', 'High', 'Low', 'Open']].copy()
        training_data = training_data.set_index('Datetime')
        # Set the data frame index using column Date
        test_data = test_data.set_index('Datetime')
        #print(test_data)

        train_scaled = self._min_max.fit_transform(training_data)
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
        test_scaled = self._min_max.fit_transform(inputs)

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
            latest_volume_price = latest_volume_price + (latest_volume_price * random_slope)

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



