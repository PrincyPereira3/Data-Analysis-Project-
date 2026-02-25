import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =============================
# 1. Load Data
# =============================
history_file = "Coca-Cola_stock_history.csv"
info_file = "Coca-Cola_stock_info.csv"

stock_history = pd.read_csv(history_file)
stock_info = pd.read_csv(info_file)

# =============================
# 2. Data Cleaning
# =============================
stock_history['Date'] = stock_history['Date'].astype(str).str.split(' ').str[0]
stock_history['Date'] = pd.to_datetime(stock_history['Date'], format='%Y-%m-%d', errors='coerce')
stock_history.dropna(subset=['Date'], inplace=True)
stock_history.sort_values('Date', inplace=True)

# =============================
# 3. Feature Engineering
# =============================
stock_history['MA20'] = stock_history['Close'].rolling(window=20).mean()
stock_history['MA50'] = stock_history['Close'].rolling(window=50).mean()
stock_history['Daily_Return'] = stock_history['Close'].pct_change()
stock_history['Volatility'] = stock_history['Daily_Return'].rolling(window=20).std()

# RSI Calculation
window_length = 14
delta = stock_history['Close'].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(window=window_length).mean()
avg_loss = pd.Series(loss).rolling(window=window_length).mean()
rs = avg_gain / avg_loss
stock_history['RSI'] = 100 - (100 / (1 + rs))

# MACD Calculation
short_ema = stock_history['Close'].ewm(span=12, adjust=False).mean()
long_ema = stock_history['Close'].ewm(span=26, adjust=False).mean()
stock_history['MACD'] = short_ema - long_ema
stock_history['Signal'] = stock_history['MACD'].ewm(span=9, adjust=False).mean()

# Bollinger Bands
stock_history['BB_Middle'] = stock_history['Close'].rolling(window=20).mean()
stock_history['BB_Upper'] = stock_history['BB_Middle'] + (stock_history['Close'].rolling(window=20).std() * 2)
stock_history['BB_Lower'] = stock_history['BB_Middle'] - (stock_history['Close'].rolling(window=20).std() * 2)

stock_history.dropna(inplace=True)

# =============================
# 4. Backtesting Strategy
# =============================
stock_history['Signal_Strategy'] = 0
stock_history.loc[stock_history['MA20'] > stock_history['MA50'], 'Signal_Strategy'] = 1
stock_history['Position'] = stock_history['Signal_Strategy'].diff()

initial_balance = 10000
balance = initial_balance
position = 0
for i in range(len(stock_history)):
    if stock_history['Position'].iloc[i] == 1:  # Buy
        position = balance / stock_history['Close'].iloc[i]
        balance = 0
    elif stock_history['Position'].iloc[i] == -1:  # Sell
        balance = position * stock_history['Close'].iloc[i]
        position = 0
final_balance = balance + (position * stock_history['Close'].iloc[-1])

# =============================
# 5. Modeling
# =============================
features = ['Open','High','Low','Volume','MA20','MA50','Daily_Return','Volatility']
X = stock_history[features]
y = stock_history['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'rf_model.pkl')

predictions = model.predict(X_test)
print(f"Model Evaluation: MAE={mean_absolute_error(y_test, predictions):.2f}, "
      f"MSE={mean_squared_error(y_test, predictions):.2f}, "
      f"Final Backtest Balance=${final_balance:.2f}")

# =============================
# 6. Streamlit Dashboard
# =============================
def run_dashboard():
    st.title('Coca-Cola Stock Analysis & Prediction')
    st.write(f"Backtesting Final Balance: ${final_balance:.2f}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Close Price", "Open Price", "RSI", "MACD", "Bollinger Bands", "Prediction"])

    with tab1:
        fig_close = px.line(stock_history, x='Date', y='Close', title='Closing Price')
        st.plotly_chart(fig_close)

    with tab2:
        fig_open = px.line(stock_history, x='Date', y='Open', title='Opening Price')
        st.plotly_chart(fig_open)

    with tab3:
        fig_rsi = px.line(stock_history, x='Date', y='RSI', title='RSI (14-day)')
        st.plotly_chart(fig_rsi)

    with tab4:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=stock_history['Date'], y=stock_history['MACD'], mode='lines', name='MACD'))
        fig_macd.add_trace(go.Scatter(x=stock_history['Date'], y=stock_history['Signal'], mode='lines', name='Signal'))
        fig_macd.update_layout(title='MACD and Signal Line')
        st.plotly_chart(fig_macd)

    with tab5:
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=stock_history['Date'], y=stock_history['Close'], mode='lines', name='Close'))
        fig_bb.add_trace(go.Scatter(x=stock_history['Date'], y=stock_history['BB_Upper'], mode='lines', name='Upper Band'))
        fig_bb.add_trace(go.Scatter(x=stock_history['Date'], y=stock_history['BB_Lower'], mode='lines', name='Lower Band'))
        fig_bb.update_layout(title='Bollinger Bands')
        st.plotly_chart(fig_bb)

    with tab6:
        model = joblib.load('rf_model.pkl')
        if st.button("Predict Today's Closing Price"):
            data = yf.download('KO', period='5d', interval='1d')
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data['MA50'] = data['Close'].rolling(window=50).mean()
            data['Daily_Return'] = data['Close'].pct_change()
            data['Volatility'] = data['Daily_Return'].rolling(window=20).std()
            latest = data.iloc[-1]
            input_features = np.array([[latest['Open'], latest['High'], latest['Low'], latest['Volume'],
                                        latest['MA20'], latest['MA50'], latest['Daily_Return'], latest['Volatility']]])
            prediction = model.predict(input_features)[0]
            st.success(f'Predicted Closing Price: ${prediction:.2f}')

if __name__ == "__main__":
    run_dashboard()