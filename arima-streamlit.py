import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Título de la aplicación
st.title("Simulación y Pronóstico con ARIMA")

# Generar 150 observaciones simuladas
np.random.seed(42)
data = np.random.randn(150).cumsum()
date_range = pd.date_range(start='2023-01-01', periods=150, freq='D')
series = pd.Series(data, index=date_range)

# Mostrar las observaciones simuladas
st.write("Observaciones Simuladas:")
st.line_chart(series)

# Casilla para que el usuario ingrese el número de periodos a pronosticar
num_periods = st.number_input('Número de periodos para pronosticar:', min_value=1, max_value=200, value=30)

# Ajustar el modelo ARIMA
model = ARIMA(series, order=(5,1,0))  # Orden (p,d,q)
model_fit = model.fit()

# Realizar el pronóstico
forecast = model_fit.forecast(steps=num_periods)
forecast_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=num_periods, freq='D')
forecast_series = pd.Series(forecast, index=forecast_dates)

# Mostrar las predicciones
st.write(f"Pronóstico para los próximos {num_periods} periodos:")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(series, label='Datos Históricos')
ax.plot(forecast_series, label='Pronóstico', color='red')
ax.set_title(f'Pronóstico para los próximos {num_periods} días')
ax.legend()
st.pyplot(fig)
