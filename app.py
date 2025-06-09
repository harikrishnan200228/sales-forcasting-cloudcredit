
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def create_lag_features(series, lags=5):
    df_lag = pd.DataFrame({'sales': series})
    for lag in range(1, lags + 1):
        df_lag[f'lag_{lag}'] = df_lag['sales'].shift(lag)
    df_lag.dropna(inplace=True)
    return df_lag

st.title("📈 Time Series Sales Forecasting")
st.markdown("Upload a CSV file with `id` and `sales` columns.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Raw Data")
    st.write(df.head())

    sales = df['sales']
    st.subheader("Sales Over Time")
    st.line_chart(sales)

    train_size = int(len(sales) * 0.8)
    train, test = sales[:train_size], sales[train_size:]

    arima_model = ARIMA(train, order=(5, 1, 0))
    arima_result = arima_model.fit()
    arima_forecast = arima_result.forecast(steps=len(test))
    arima_mae = mean_absolute_error(test, arima_forecast)
    arima_rmse = np.sqrt(mean_squared_error(test, arima_forecast))

    ets_model = ExponentialSmoothing(train, trend='add', seasonal=None)
    ets_result = ets_model.fit()
    ets_forecast = ets_result.forecast(len(test))
    ets_mae = mean_absolute_error(test, ets_forecast)
    ets_rmse = np.sqrt(mean_squared_error(test, ets_forecast))

    lags = 5
    df_lagged = create_lag_features(sales, lags)
    X = df_lagged.drop(columns='sales')
    y = df_lagged['sales']
    X_train = X[:train_size - lags]
    X_test = X[train_size - lags:]
    y_train = y[:train_size - lags]
    y_test = y[train_size - lags:]
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_forecast = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_forecast)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_forecast))

    st.subheader("Model Performance")
    perf_df = pd.DataFrame({
        'Model': ['ARIMA', 'ETS', 'Random Forest'],
        'MAE': [arima_mae, ets_mae, rf_mae],
        'RMSE': [arima_rmse, ets_rmse, rf_rmse]
    })
    st.dataframe(perf_df)

    st.subheader("Forecast vs Actual")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test.index, test.values, label='Actual', color='black')
    ax.plot(test.index, arima_forecast, label='ARIMA', linestyle='--')
    ax.plot(test.index, ets_forecast, label='ETS', linestyle='--')
    ax.plot(y_test.index, rf_forecast, label='Random Forest', linestyle='--')
    ax.set_title("Sales Forecast Comparison")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
