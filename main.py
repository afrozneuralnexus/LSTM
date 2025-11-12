import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
from datetime import date

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("📈 Stock Price Forecasting App")

stocks = ("AAPL", "GOOG", "MSFT", "GME", "TSLA", "AMZN")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction:", 1, 5)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("✅ Data loaded successfully!")

st.subheader("Raw data")
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="stock_open"))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="stock_close"))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Prepare data for Prophet
df_train = data[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
df_train["ds"] = pd.to_datetime(df_train["ds"])
df_train["y"] = pd.to_numeric(df_train["y"], errors="coerce")

# Forecasting
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Forecast data")
st.write(forecast.tail())

st.write("📉 Forecast plot")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("🔮 Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

