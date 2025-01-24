import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import plotly.express as px
from datetime import timedelta, time
from statsmodels.tsa.arima.model import ARIMA
import os 
import secrets
from ta.volatility import BollingerBands
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, Dense, LSTM
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

PROJECT_FOLDER = ''

if "previous_symbol" not in st.session_state:
    st.session_state.previous_symbol = ""

if "project_folder" not in st.session_state:
    st.session_state.project_folder = ""

def generate_market_time_range_1_minute(start_date, periods):
    """
    Generates a range of timestamps within stock market hours (9:15 AM to 3:30 PM) for a given number of periods.
    Each period corresponds to a 1-minute interval.
    """
    market_open = time(9, 15)  # 9:15 AM
    market_close = time(15, 30)  # 3:30 PM
    timestamps = []
    current_date = start_date.date()
    count = 0

    while count < periods:
        # Generate intraday 1-minute intervals within trading hours
        current_time = datetime.datetime.combine(current_date, market_open)
        while current_time.time() <= market_close and count < periods:
            timestamps.append(current_time)
            current_time += timedelta(minutes=1)
            count += 1
        current_date += timedelta(days=1)  # Move to the next day
    return pd.DatetimeIndex(timestamps)

def generate_market_time_range_5_minute(start_date, periods):
    """
    Generates a range of timestamps within stock market hours (9:15 AM to 3:30 PM) for a given number of periods.
    Each period corresponds to a 5-minute interval.
    """
    market_open = time(9, 15)  # 9:15 AM
    market_close = time(15, 30)  # 3:30 PM
    timestamps = []
    current_date = start_date.date()
    count = 0

    while count < periods:
        # Generate intraday 5-minute intervals within trading hours
        current_time = datetime.datetime.combine(current_date, market_open)
        while current_time.time() <= market_close and count < periods:
            timestamps.append(current_time)
            current_time += timedelta(minutes=5)
            count += 1
        current_date += timedelta(days=1)  # Move to the next day
    return pd.DatetimeIndex(timestamps)

def generate_market_time_range_daily(start_date, days):
    """
    Generates a range of timestamps for the next `days` trading days.
    Each timestamp corresponds to the market close time (3:30 PM).
    """
    market_close = time(15, 30)  # 3:30 PM
    timestamps = []
    current_date = start_date.date()
    trading_days = 0

    while trading_days < days:
        # Skip weekends (Saturday and Sunday)
        if current_date.weekday() < 5:  # Monday to Friday are trading days
            timestamps.append(datetime.datetime.combine(current_date, market_close))
            trading_days += 1
        current_date += timedelta(days=1)  # Move to the next day
    
    return pd.DatetimeIndex(timestamps)

# Header formatting using Markdown and CSS
header_style = """
    <style>
        .header {
            color: #0077b6;
            font-size: 48px;
            text-shadow: 2px 2px 2px rgba(0, 0, 0, 0.15);
        }
        .subheader {
            color: #0077b6;
            font-size: 30px;
        }
    </style>
"""

# Apply header formatting to the title
st.markdown(header_style, unsafe_allow_html=True)
st.markdown('<h1 class="header">Stock Price Forecast</h1>', unsafe_allow_html=True)

# Sidebar with prediction options
# Sidebar header with styling
st.sidebar.markdown("""
    <h2 style="color: #0077b6; text-align: center;">Prediction Model</h2>
    <p style="font-size: 14px; color: #4a4a4a; text-align: justify;">
        Choose a prediction model to forecast stock prices. Each model has its strengths:
        <ul>
            <li><b>LSTM:</b> Long Short-Term Memory, ideal for handling sequential data.</li>
            <li><b>ARIMA:</b> AutoRegressive Integrated Moving Average, great for time series forecasting.</li>
        </ul>
    </p>
""", unsafe_allow_html=True)

# Add the radio button menu with icons
menu_option = st.sidebar.radio(
    "🔮 Select Prediction Model",
    ("🧠 Predict with LSTM", "📈 Predict with ARIMA"),
    help="Select one of the prediction models to forecast stock prices."
)

# Parse the selected option to extract the model
if "LSTM" in menu_option:
    menu_option = "Predict with LSTM"
elif "ARIMA" in menu_option:
    menu_option = "Predict with ARIMA"

# Display the selected model in the sidebar
st.sidebar.markdown(f"""
    <div style="margin-top: 20px; text-align: center;">
        <h3 style="color: #0077b6;">You selected:</h3>
        <h3 style="color: #023e8a;">{menu_option}</h3>
    </div>
""", unsafe_allow_html=True)


# User input for stock symbol
stock_symbol = st.text_input("Enter the stock symbol (e.g. VOLTAS.NS for Voltas):", 'COROMANDEL.NS')

# Validate the stock symbol
if stock_symbol:
    ticker = yf.Ticker(stock_symbol)
    try:
        ticker_info = ticker.info
        if stock_symbol != st.session_state.previous_symbol:
            TODAY_RUN = datetime.datetime.today().strftime("%Y%m%d")
            TOKEN = stock_symbol + '_' + TODAY_RUN + '_' + secrets.token_hex(16)
            PROJECT_FOLDER = os.path.join(os.getcwd(), TOKEN)
            if not os.path.exists(PROJECT_FOLDER):
                os.makedirs(PROJECT_FOLDER)
            st.session_state.previous_symbol = stock_symbol
            st.session_state.project_folder = os.path.join(os.getcwd(), TOKEN)
    except:
        st.warning(f"'{stock_symbol}' is not a valid stock ticker symbol. Please enter a valid ticker.")

EPOCHS = 1
BATCH_SIZE = 32
TIME_STEPS = 60
PROJECT_FOLDER = st.session_state.project_folder
print("PROJECT_FOLDER: ", PROJECT_FOLDER)
scaler = MinMaxScaler(feature_range=(0, 1))

# Ask the user how they want to fetch data
data_option = st.radio(
    "How would you like to fetch the data?",
    ("Download last 8 days of precise data (1-minute interval)", "Download last 60 days of precise data (5-minute interval)", "Select date range manually")
)

if data_option == "Download last 8 days of precise data (1-minute interval)":
    # Fetch data for the last 8 days with a 1-minute interval
    end_date = datetime.datetime.now()
    start_date = end_date - timedelta(days=8)
    last_3_day = end_date - timedelta(days=3)
    validation_date = pd.to_datetime(last_3_day)  
    PTEDICTED_TIME = 360
    PREDICT_START_DATE = datetime.datetime.today()


    stock_data = yf.download(
        stock_symbol,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval="1m"
    )

    if stock_data.empty:
        st.error("No data found for the given interval. Please try a different symbol or adjust the interval.")
    else:
        stock_data.index = pd.to_datetime(stock_data.index)
        stock_data.index = stock_data.index.tz_convert("Asia/Kolkata")
        stock_data.reset_index(inplace=True)
        stock_data = stock_data.drop(index=stock_data.index[0])
        stock_data.to_csv(os.path.join(PROJECT_FOLDER, 'downloaded_data_'+ stock_symbol+'.csv'), index=False)
        stock_data=pd.read_csv(os.path.join(PROJECT_FOLDER, 'downloaded_data_'+ stock_symbol +'.csv'))
        stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')
        st.success(f"Downloaded 1-minute interval data for the last 8 days.")
        # st.write(stock_data)

elif data_option == "Download last 60 days of precise data (5-minute interval)":
    # Fetch data for the last 8 days with a 1-minute interval
    end_date = datetime.datetime.now()
    start_date = end_date - timedelta(days=59)
    last_15_day = end_date - timedelta(days=15)
    validation_date = pd.to_datetime(last_15_day)  
    # You are predicting for the next 3 days. Each day has 6 hours of trading time (from 9:15 AM to 3:30 PM),
    # and each hour has 12 intervals (5 minutes each).
    # So, 6 * 12 = 72 intervals per day, and for 3 days, the total periods are 216
    PTEDICTED_TIME = 216
    PREDICT_START_DATE = datetime.datetime.today()


    stock_data = yf.download(
        stock_symbol,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval="5m"
    )
    if stock_data.empty:
        st.error("No data found for the given interval. Please try a different symbol or adjust the interval.")
    else:
        stock_data.index = pd.to_datetime(stock_data.index)
        stock_data.index = stock_data.index.tz_convert("Asia/Kolkata")
        stock_data.reset_index(inplace=True)
        stock_data = stock_data.drop(index=stock_data.index[0])
        stock_data.to_csv(os.path.join(PROJECT_FOLDER, 'downloaded_data_'+ stock_symbol+'.csv'), index=False)
        stock_data=pd.read_csv(os.path.join(PROJECT_FOLDER, 'downloaded_data_'+ stock_symbol +'.csv'))
        stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')
        st.success(f"Downloaded 1-minute interval data for the last 8 days.")
        # st.write(stock_data)

elif data_option == "Select date range manually":
    # Let the user manually select the date range
    start_date = st.date_input("Enter the start date:", datetime.date(2017, 1, 1))
    end_date = datetime.date.today()
    # Calculate the duration from start_date to today
    today = datetime.datetime.today().date()
    total_duration = (today - start_date).days  # Total days between start_date and today

    # Calculate 70% of the total duration
    validation_duration = int(0.7 * total_duration)

    # Calculate validation_date
    validation_date = start_date + timedelta(days=validation_duration)

    PTEDICTED_TIME = 30
    PREDICT_START_DATE = datetime.datetime.today()

    if start_date > end_date:
        st.error("Start date must be earlier than end date.")
    else:
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        if stock_data.empty:
            st.error("No data found for the given date range. Please try a different range.")
        else:
            stock_data.index = pd.to_datetime(stock_data.index)
            stock_data.index = stock_data.index.tz_localize("Asia/Kolkata")
            stock_data.reset_index(inplace=True)
            stock_data = stock_data.drop(index=stock_data.index[0])
            # Rename 'Date' to 'Datetime'
            if 'Date' in stock_data.columns:
                stock_data.rename(columns={'Date': 'Datetime'}, inplace=True)
            stock_data.to_csv(os.path.join(PROJECT_FOLDER, 'downloaded_data_'+ stock_symbol+'.csv'), index=False)
            stock_data=pd.read_csv(os.path.join(PROJECT_FOLDER, 'downloaded_data_'+ stock_symbol +'.csv'))
            stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')
            st.success(f"Downloaded data from {start_date} to {end_date}.")
            # st.write(stock_data)

# Plot 50-day and 100-day Simple Moving Averages
sma_50 = stock_data['Close'].rolling(window=50).mean()
sma_100 = stock_data['Close'].rolling(window=100).mean()
# stock_data['SMA_50'] = sma_50
stock_data['SMA_100'] = sma_100

# Plot 50-day and 100-day Exponential Moving Averages
ema_50 = stock_data['Close'].ewm(span=50, adjust=False).mean()
ema_100 = stock_data['Close'].ewm(span=100, adjust=False).mean()
# stock_data['EMA_50'] = ema_50
# stock_data['EMA_100'] = ema_100

# Plot Relative Strength Index (RSI)
rsi_period = 14
delta = stock_data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=rsi_period).mean()
avg_loss = loss.rolling(window=rsi_period).mean()
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))
stock_data['RSI'] = rsi

# Plot Moving Average Convergence Divergence (MACD)
short_period = 12
long_period = 26
ema_short = stock_data['Close'].ewm(span=short_period, adjust=False).mean()
ema_long = stock_data['Close'].ewm(span=long_period, adjust=False).mean()
macd_line = ema_short - ema_long
signal_line = macd_line.ewm(span=9, adjust=False).mean()
macd_histogram = macd_line - signal_line
stock_data['MACD'] = macd_histogram

#create instance of SES
# SES reduces this noise by giving more weight to recent data and
# less weight to older data, creating a smoothed version of the stock's price movement.
# ses = SimpleExpSmoothing(stock_data['Close'])
# #fit SES to data
# alpha = 0.7
# res = ses.fit(smoothing_level=alpha, optimized=False)
# stock_data['SES'] = res.fittedvalues

# Add Lag Features
for lag in [1, 3, 5, 10]:
    stock_data[f'Close_lag_{lag}'] = stock_data['Close'].shift(lag)

stock_data['Volume'] = pd.to_numeric(stock_data['Volume'], errors='coerce')

# VWAP (Volume-Weighted Average Price)
stock_data['Cum_Price_Volume'] = (stock_data['Close'] * stock_data['Volume']).cumsum()
stock_data['Cum_Volume'] = stock_data['Volume'].cumsum()
stock_data['VWAP'] = stock_data['Cum_Price_Volume'] / stock_data['Cum_Volume']
stock_data.drop(columns=['Cum_Price_Volume', 'Cum_Volume'], inplace=True)

# ATR (Average True Range)
stock_data['High'] = pd.to_numeric(stock_data['High'], errors='coerce')
stock_data['Low'] = pd.to_numeric(stock_data['Low'], errors='coerce')
stock_data['Open'] = pd.to_numeric(stock_data['Open'], errors='coerce')

stock_data['High-Low'] = stock_data['High'] - stock_data['Low']
stock_data['High-Close'] = abs(stock_data['High'] - stock_data['Close'].shift())
stock_data['Low-Close'] = abs(stock_data['Low'] - stock_data['Close'].shift())
stock_data['True_Range'] = stock_data[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
stock_data['ATR'] = stock_data['True_Range'].rolling(window=14).mean()
stock_data.drop(columns=['High-Low', 'High-Close', 'Low-Close', 'True_Range'], inplace=True)

stock_data.dropna(subset=['Datetime'], inplace=True)

# Drop rows with NaN values
stock_data.dropna(inplace=True)
stock_data.to_csv(os.path.join(PROJECT_FOLDER, 'data_'+stock_symbol+'.csv'), index=False)
st.write(stock_data)

# Quick Stock Quote Assessment
st.markdown('<h2 class="subheader">Stock Quote</h2>', unsafe_allow_html=True)

# Fetch stock quote using yfinance
quote_data = yf.Ticker(stock_symbol)
quote_info = quote_data.info
# Display relevant information in a tabular format
quote_table = {
    "Category": ["Company Name", "Current Stock Price", "Change Perecentage", "Open Price", "High Price", "Low Price",
                 "Volume", "Market Capitalization", "52-Week Range", "Dividend Yield", "P/E", "EPS"],
    "Value": [quote_info.get('longName', 'N/A'),
              f"${quote_info.get('currentPrice', 'N/A'):.2f}" if isinstance(quote_info.get('currentPrice'), float) else 'N/A',
              f"{quote_info.get('regularMarketChangePercent', 'N/A'):.2%}" if quote_info.get('regularMarketChangePercent') is not None else 'N/A',
              f"${quote_info.get('open', 'N/A'):.2f}" if isinstance(quote_info.get('open'), float) else 'N/A',
              f"${quote_info.get('dayHigh', 'N/A'):.2f}" if isinstance(quote_info.get('dayHigh'), float) else 'N/A',
              f"${quote_info.get('dayLow', 'N/A'):.2f}" if isinstance(quote_info.get('dayLow'), float) else 'N/A',
              f"{quote_info.get('regularMarketVolume', 'N/A') / 1000000:.2f}M" if isinstance(quote_info.get('regularMarketVolume'), int) else 'N/A',
              f"${quote_info.get('marketCap', 'N/A'):,}" if isinstance(quote_info.get('marketCap'), int) else 'N/A',
              f"${quote_info.get('fiftyTwoWeekLow', 'N/A'):.2f} - ${quote_info.get('fiftyTwoWeekHigh', 'N/A'):.2f}" if isinstance(quote_info.get('fiftyTwoWeekLow'), float) and isinstance(quote_info.get('fiftyTwoWeekHigh'), float) else 'N/A',
              f"{quote_info.get('dividendYield', 'N/A'):.2%}" if quote_info.get('dividendYield') is not None else 'N/A',
              quote_info.get('trailingPE', 'N/A'),
              quote_info.get('trailingEps', 'N/A')]
}

quote_table_df = pd.DataFrame(quote_table)
quote_table_df.index = range(1, len(quote_table_df) + 1)
st.table(quote_table_df)

# Visualize Stock Price
st.markdown('<h2 class="subheader">Stock Prices Over Time</h2>', unsafe_allow_html=True)


fig = px.line(stock_data, x=stock_data.index, y='Close')
st.plotly_chart(fig)

# Create a horizontal slider to navigate through different indicators
st.markdown('<h2 style="color: #0077b6; font-size: 28px;">Select a Technical Indicator</h2>', unsafe_allow_html=True)

# Add an explanation about the indicators
st.markdown("""
    <p style="font-size: 16px; color: #4a4a4a;">
        Choose a technical indicator from the dropdown to visualize it. Here's what each option means:
    </p>
    <ul style="font-size: 14px; color: #4a4a4a;">
        <li><b>SMA</b>: Simple Moving Average - Helps smooth out price data to identify trends.</li>
        <li><b>EMA</b>: Exponential Moving Average - Similar to SMA but gives more weight to recent prices.</li>
        <li><b>RSI</b>: Relative Strength Index - Indicates overbought or oversold conditions.</li>
        <li><b>MACD</b>: Moving Average Convergence Divergence - Highlights momentum and trend strength.</li>
        <li><b>VWAP</b>: Volume Weighted Average Price - True average price based on its liquidity.</li>
        <li><b>ATR</b>: Average True Range - Average range of price movement for a stock over a specified period. </li>
    </ul>
""", unsafe_allow_html=True)

# Add the selectbox with icons
selected_indicator = st.selectbox(
    "📈 Choose an indicator",
    ["📊 SMA - Simple Moving Average", 
    "📉 EMA - Exponential Moving Average", 
    "🔍 RSI - Relative Strength Index", 
    "📶 MACD - Moving Average Convergence Divergence", 
    "📐 VWAP - Volume Weighted Average Price", 
    "📏 ATR - Average True Range"],
    help="Select one of the technical indicators to display its visualization."
)

if selected_indicator == "📊 SMA - Simple Moving Average":
    fig_sma = go.Figure()

    # Add Candlestick Chart
    fig_sma.add_trace(go.Candlestick(
        x=stock_data['Datetime'],
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name='Candlestick'
    ))

    # Add Short-term SMA (20-period)
    fig_sma.add_trace(go.Scatter(
        x=stock_data['Datetime'],
        y=sma_100,
        mode='lines',
        name='100-Period SMA',
        line=dict(color='green', width=2)
    ))

    # Add Long-term SMA (50-period)
    fig_sma.add_trace(go.Scatter(
        x=stock_data['Datetime'],
        y=sma_50,
        mode='lines',
        name='50-Period SMA',
        line=dict(color='blue', width=2)
    ))

    # Update Layout
    fig_sma.update_layout(
        title='Simple Moving Averages (SMA)',
        xaxis_title='Datetime',
        yaxis_title='Price',
        template='plotly_dark',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=600
    )

    # Display in Streamlit
    st.plotly_chart(fig_sma)

elif selected_indicator == "📉 EMA - Exponential Moving Average":
    # Plot EMA Chart
    fig_ema = go.Figure()

    # Add Candlestick Chart
    fig_ema.add_trace(go.Candlestick(
        x=stock_data['Datetime'],
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name='Candlestick'
    ))

    # Add 50-day EMA
    fig_ema.add_trace(go.Scatter(
        x=stock_data['Datetime'],
        y=ema_50,
        mode='lines',
        name='50-day EMA',
        line=dict(color='green', width=2)
    ))

    # Add 100-day EMA
    fig_ema.add_trace(go.Scatter(
        x=stock_data['Datetime'],
        y=ema_100,
        mode='lines',
        name='100-day EMA',
        line=dict(color='orange', width=2)
    ))

    # Update Layout
    fig_ema.update_layout(
        title='Exponential Moving Averages (EMA)',
        xaxis_title='Datetime',
        yaxis_title='Price',
        template='plotly_dark',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=600
    )

    # Display in Streamlit
    st.plotly_chart(fig_ema, use_container_width=True)

elif selected_indicator == "🔍 RSI - Relative Strength Index":
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=stock_data['Datetime'], y=rsi, mode='lines', name=f'RSI ({rsi_period}-day)'))
    fig_rsi.add_trace(go.Scatter(x=stock_data['Datetime'], y=[70] * len(stock_data), mode='lines', name='Overbought (70)', line=dict(color='red', dash='dash')))
    fig_rsi.add_trace(go.Scatter(x=stock_data['Datetime'], y=[30] * len(stock_data), mode='lines', name='Oversold (30)', line=dict(color='green', dash='dash')))
    fig_rsi.update_layout(title='Relative Strength Index (RSI)', xaxis_title='Date', yaxis_title='RSI Value')
    st.plotly_chart(fig_rsi)

elif selected_indicator == "📶 MACD - Moving Average Convergence Divergence":
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=stock_data['Datetime'], y=macd_line, mode='lines', name='MACD Line'))
    fig_macd.add_trace(go.Scatter(x=stock_data['Datetime'], y=signal_line, mode='lines', name='Signal Line', line=dict(color='orange')))
    fig_macd.add_trace(go.Bar(x=stock_data['Datetime'], y=macd_histogram, name='MACD Histogram', marker_color='grey'))
    fig_macd.add_trace(go.Scatter(x=stock_data['Datetime'], y=[0] * len(stock_data), mode='lines', name='Zero Line', line=dict(color='black', dash='dash')))
    fig_macd.update_layout(title='Moving Average Convergence Divergence (MACD)', xaxis_title='Date', yaxis_title='MACD Value')
    st.plotly_chart(fig_macd)

elif selected_indicator == "📐 VWAP - Volume Weighted Average Price":
    fig_vwap = go.Figure()
    fig_vwap.add_trace(go.Candlestick(x=stock_data['Datetime'],
        open=stock_data['Open'],  # Assuming 'Open' is same as 'High' for simplicity
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name='Candlestick'
    ))
    fig_vwap.add_trace(go.Scatter(
        x=stock_data['Datetime'],
        y=stock_data['VWAP'],
        mode='lines',
        line=dict(color='orange', width=2),
        name='VWAP'
    ))
    fig_vwap.update_layout(
        title='VWAP and Candlestick Chart',
        xaxis_title='Datetime',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=600
    )
    # Show the plot
    st.plotly_chart(fig_vwap)

elif selected_indicator == "📏 ATR - Average True Range":
    # Plot ATR Chart
    fig_atr = go.Figure()

    # Add ATR line
    fig_atr.add_trace(go.Scatter(
        x=stock_data['Datetime'],
        y=stock_data['ATR'],
        mode='lines',
        name='ATR',
        line=dict(color='orange', width=2)
    ))

    # Add Candlestick Chart
    fig_atr.add_trace(go.Candlestick(
        x=stock_data['Datetime'],
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name='Candlestick'
    ))

    # Update layout for better visuals
    fig_atr.update_layout(
        title='Average True Range (ATR)',
        xaxis_title='Datetime',
        yaxis_title='Value',
        template='plotly_dark',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=600
    )

    # Display in Streamlit
    st.plotly_chart(fig_atr)

menu_option_trained = st.radio(
    "Choose an option:", 
    ["None Selected","Train New Model", "Use Pre-Trained Model"], 
    index=0,
    )


# Handle Model Selection
if menu_option == "Predict with LSTM":
    stock_data['Datetime'] = pd.to_datetime(stock_data['Datetime'], errors='coerce')
    training_data = stock_data[stock_data['Datetime'] < pd.Timestamp(validation_date).tz_localize("Asia/Kolkata")].copy()
    test_data = stock_data[stock_data['Datetime'] >= pd.Timestamp(validation_date).tz_localize('Asia/Kolkata')].copy()
    training_data = training_data.set_index('Datetime')
    # Set the data frame index using column Date
    test_data = test_data.set_index('Datetime')

    train_scaled = scaler.fit_transform(training_data[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'VWAP', 'SMA_100',  'ATR']])

    # Training Data Transformation
    x_train = []
    y_train = []
    for i in range(TIME_STEPS, len(train_scaled)):
        x_train.append(train_scaled[i - TIME_STEPS:i])
        y_train.append(train_scaled[i, 3]) 

    x_train, y_train = np.array(x_train), np.array(y_train)
    total_data = pd.concat((training_data, test_data), axis=0)
    inputs = total_data[len(total_data) - len(test_data) - TIME_STEPS:]
    test_scaled = scaler.fit_transform(inputs[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'VWAP', 'SMA_100', 'ATR']])
    # Testing Data Transformation
    x_test = []
    y_test = []
    for i in range(TIME_STEPS, len(test_scaled)):
        x_test.append(test_scaled[i - TIME_STEPS:i])
        y_test.append(test_scaled[i, 3])

    x_test, y_test = np.array(x_test), np.array(y_test)
    print("x_train.shape: ", x_train.shape)
    print("x_test.shape: ", x_test.shape)
    if menu_option_trained == "None Selected":
        print("None Selected")
    elif menu_option_trained == "Train New Model":

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)  # Predict Close Price
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics= tf.keras.metrics.MeanSquaredError(name='MSE'))
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test),
                            callbacks=[reduce_lr])
        print("saving weights")
        model.save(os.path.join(PROJECT_FOLDER, 'close_model_weights.h5'))
        test_predictions_baseline = model.predict(x_test)

        test_predictions_baseline_padded = np.zeros((test_predictions_baseline.shape[0], x_test.shape[2]))
        test_predictions_baseline_padded[:, 3] = test_predictions_baseline.flatten()

        # Perform inverse transform
        x_test_predictions_baseline = scaler.inverse_transform(test_predictions_baseline_padded)[:, 3]  # Extract only the first column

        test_predictions_baseline = pd.DataFrame({
                'Datetime': test_data.index,  # Datetime from test_data index
                f'{stock_symbol}_actual': test_data.Close,  # Actual Close price
                f'{stock_symbol}_predicted': x_test_predictions_baseline  # Predicted Close price
            })
        test_predictions_baseline.set_index('Datetime')
        test_predictions_baseline.to_csv(os.path.join(PROJECT_FOLDER, 'predictions.csv'))

        model = tf.keras.models.load_model(os.path.join(PROJECT_FOLDER, "close_model_weights.h5"))

        # Adjust the number of features dynamically
        num_features = x_test.shape[2]  # Number of features in the dataset

        # Perform inverse scaling on predicted values
        predictions = model.predict(x_test)
        # predictions = predictions[:PTEDICTED_TIME]
        # y_test = y_test[:PTEDICTED_TIME]
        predictions = predictions[-PTEDICTED_TIME:]
        y_test = y_test[-PTEDICTED_TIME:]

        # Inverse scaling for 'Close' price
        predicted_close = scaler.inverse_transform(np.column_stack((
            np.zeros((len(predictions), 3)),  # Placeholders for Open, High, Low
            predictions,  # Predicted Close (assuming it is the first column of predictions)
            np.zeros((len(predictions), 6))  # Placeholder for remaining features
        )))[:, 3]  # Here we select index 3 for 'Close' if 'Close' is the fourth column

        if data_option == "Download last 8 days of precise data (1-minute interval)":
            predicted_dates = generate_market_time_range_1_minute(PREDICT_START_DATE, PTEDICTED_TIME)
        elif data_option == "Download last 60 days of precise data (5-minute interval)":
            predicted_dates = generate_market_time_range_5_minute(PREDICT_START_DATE, PTEDICTED_TIME)
        else:
            predicted_dates = generate_market_time_range_daily(PREDICT_START_DATE, PTEDICTED_TIME)

        # Add predictions and actual values to test_data
        close_data = pd.DataFrame({
            'Datetime': predicted_dates,
            'Predicted Close Price': predicted_close,
        })
        close_data.to_csv(os.path.join(PROJECT_FOLDER, "predicted_data.csv"), index=False)

        # Create the plot using Plotly
        close_fig = go.Figure()

        # Add the predicted close price line
        close_fig.add_trace(go.Scatter(
            x=predicted_dates,
            y=predicted_close,
            mode='lines',
            name='Predicted Close Price',
            line=dict(color='red', width=2)
        ))

        # Customize layout
        close_fig.update_layout(
            title="Predicted Close Prices",
            xaxis=dict(
                title="Datetime",
            ),
            yaxis=dict(
                title="Close Price"
            ),
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            height=600
        )

        # Add gridlines and rotation for x-axis labels
        close_fig.update_xaxes(
            tickangle=45,  # Rotate x-axis labels by 45 degrees
            showgrid=True
        )
        close_fig.update_yaxes(showgrid=True)
        st.plotly_chart(close_fig, use_container_width=True)

    elif menu_option_trained == "Use Pre-Trained Model":
        file_name = "close_model_weights.h5"

        # Check if the file exists in the specified directory
        file_path = os.path.join(PROJECT_FOLDER, file_name)

        if os.path.isfile(file_path):
            print(f"File '{file_name}' exists in the directory '{PROJECT_FOLDER}'.")        
        else:
            st.success("Please wait...")
            print(f"File '{file_name}' does NOT exist in the directory '{PROJECT_FOLDER}'.")
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
                Dropout(0.2),
                LSTM(64, return_sequences=False),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)  # Predict Close Price
            ])
            model.compile(optimizer='adam', loss='mean_squared_error', metrics= tf.keras.metrics.MeanSquaredError(name='MSE'))
            # Reduce learning rate on plateau
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test),
                                callbacks=[reduce_lr])
            print("saving weights")
            model.save(os.path.join(PROJECT_FOLDER, 'close_model_weights.h5'))
            test_predictions_baseline = model.predict(x_test)

            test_predictions_baseline_padded = np.zeros((test_predictions_baseline.shape[0], x_test.shape[2]))
            test_predictions_baseline_padded[:, 3] = test_predictions_baseline.flatten()

            # Perform inverse transform
            x_test_predictions_baseline = scaler.inverse_transform(test_predictions_baseline_padded)[:, 3]  # Extract only the first column

            test_predictions_baseline = pd.DataFrame({
                    'Datetime': test_data.index,  # Datetime from test_data index
                    f'{stock_symbol}_actual': test_data.Close,  # Actual Close price
                    f'{stock_symbol}_predicted': x_test_predictions_baseline  # Predicted Close price
                })
            test_predictions_baseline.set_index('Datetime')
            test_predictions_baseline.to_csv(os.path.join(PROJECT_FOLDER, 'predictions.csv'))
            st.success("Model training success")
            print("st.session_state: ", st.session_state)

            fig = go.Figure()

            # Add predicted price line
            fig.add_trace(go.Scatter(
                x=test_predictions_baseline['Datetime'],
                y=test_predictions_baseline[stock_symbol + '_predicted'],
                mode='lines',
                name=stock_symbol + ' Predicted Price',
                line=dict(color='red')
            ))

            fig.add_trace(go.Scatter(
                x=test_predictions_baseline['Datetime'],
                y=test_predictions_baseline[stock_symbol + '_actual'],
                mode='lines',
                name=stock_symbol + ' Actual Price',
                line=dict(color='green')
            ))

            # Update layout
            fig.update_layout(
                title= stock_symbol + ' Prediction vs Actual',
                xaxis_title='Datetime',
                yaxis_title='Price',
                template='plotly_dark',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                height=600
            )

            # Display in Streamlit
            st.plotly_chart(fig, use_container_width=True)

            print("prediction is finished")

        model = tf.keras.models.load_model(os.path.join(PROJECT_FOLDER, "close_model_weights.h5"))

        # Adjust the number of features dynamically
        num_features = x_test.shape[2]  # Number of features in the dataset

        # Perform inverse scaling on predicted values
        predictions = model.predict(x_test)
        # predictions = predictions[:PTEDICTED_TIME]
        # y_test = y_test[:PTEDICTED_TIME]
        predictions = predictions[-PTEDICTED_TIME:]
        y_test = y_test[-PTEDICTED_TIME:]

        # Inverse scaling for 'Close' price
        predicted_close = scaler.inverse_transform(np.column_stack((
            np.zeros((len(predictions), 3)),  # Placeholders for Open, High, Low
            predictions,  # Predicted Close (assuming it is the first column of predictions)
            np.zeros((len(predictions), 6))  # Placeholder for remaining features
        )))[:, 3]  # Here we select index 3 for 'Close' if 'Close' is the fourth column

        if data_option == "Download last 8 days of precise data (1-minute interval)":
            predicted_dates = generate_market_time_range_1_minute(PREDICT_START_DATE, PTEDICTED_TIME)
        elif data_option == "Download last 60 days of precise data (5-minute interval)":
            predicted_dates = generate_market_time_range_5_minute(PREDICT_START_DATE, PTEDICTED_TIME)
        else:
            predicted_dates = generate_market_time_range_daily(PREDICT_START_DATE, PTEDICTED_TIME)

        # Add predictions and actual values to test_data
        close_data = pd.DataFrame({
            'Datetime': predicted_dates,
            'Predicted Close Price': predicted_close,
        })
        close_data.to_csv(os.path.join(PROJECT_FOLDER, "predicted_data.csv"), index=False)

        # Create the plot using Plotly
        close_fig = go.Figure()

        # Add the predicted close price line
        close_fig.add_trace(go.Scatter(
            x=predicted_dates,
            y=predicted_close,
            mode='lines',
            name='Predicted Close Price',
            line=dict(color='red', width=2)
        ))

        # Customize layout
        close_fig.update_layout(
            title="Predicted Close Prices",
            xaxis=dict(
                title="Datetime",
            ),
            yaxis=dict(
                title="Close Price"
            ),
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            height=600
        )

        # Add gridlines and rotation for x-axis labels
        close_fig.update_xaxes(
            tickangle=45,  # Rotate x-axis labels by 45 degrees
            showgrid=True
        )
        close_fig.update_yaxes(showgrid=True)
        st.plotly_chart(close_fig, use_container_width=True)

elif menu_option == "Predict with ARIMA":
    st.markdown('<h2 class="subheader">ARIMA Prediction</h2>', unsafe_allow_html=True)
    stock_data['Datetime'] = pd.to_datetime(stock_data['Datetime'])

    # Split data into training and testing
    training_data = stock_data[stock_data['Datetime'] < pd.Timestamp(validation_date).tz_localize("Asia/Kolkata")].copy()
    testing_data = stock_data[stock_data['Datetime'] >= pd.Timestamp(validation_date).tz_localize("Asia/Kolkata")].copy()


    p, d, q = 5, 1, 0
    model = ARIMA(stock_data["Close"], order=(p,d,q))
    fitted = model.fit()

    if data_option == "Download last 8 days of precise data (1-minute interval)":
        # Generate a 1-minute interval range
        predicted_dates = pd.date_range(
            start=PREDICT_START_DATE,
            periods=PTEDICTED_TIME,
            freq='T'  # 'T' stands for minute frequency
        )
    elif data_option == "Download last 60 days of precise data (5-minute interval)":
        # Generate a 5-minute interval range
        predicted_dates = pd.date_range(
            start=PREDICT_START_DATE,
            periods=PTEDICTED_TIME,
            freq='5T'  # '5T' stands for 5-minute frequency
        )
    else:
        # Generate a daily interval range
        predicted_dates = pd.date_range(
            start=PREDICT_START_DATE,
            periods=PTEDICTED_TIME,
            freq='D'  # 'D' stands for daily frequency
        )

    # Predict for the generated date range
    steps = len(predicted_dates)  # Number of steps to forecast
    forecast_values = fitted.forecast(steps=steps)  # Directly returns a numpy array

    # Visualize ARIMA Forecasted Prices
    forecast_arima_df = pd.DataFrame({'Datetime': predicted_dates, 'Forecasted Prices': forecast_values})
    fig = px.line(forecast_arima_df, x='Datetime', y='Forecasted Prices')
    st.plotly_chart(fig)
