# -*- coding: utf-8 -*-
"""
---------------------------------------
@Time    : 2024-08-17 9:36
@Author  : lijing
@File    : main.py
@Description: 
---------------------------------------
"""

# https://github.com/unit8co/darts
# https://unit8co.github.io/darts/
# https://blog.csdn.net/Trb401012/article/details/135952747

from darts import TimeSeries
from darts.models import AutoARIMA
from .metrics import mape, rmse, mae
import pandas as pd
import matplotlib.pyplot as plt

# Read a pandas DataFrame
df = pd.read_csv("international-airline-passengers.csv", delimiter=",")
# Create a TimeSeries, specifying the time and value columns
series = TimeSeries.from_dataframe(df, "Month", "Passengers")

# Split the data into training and validation sets
train_size = int(len(series) * 0.8)
train, validation = series[:train_size], series[train_size:]


# Initialize and fit the AutoARIMA model
model = AutoARIMA(start_p=8, max_p=12, start_q=1)
model.fit(train)


# Generate forecasts
forecast = model.predict(n=len(validation))
print("Forecasts:", forecast)

# Evaluate the model
mape_score = mape(validation, forecast)
rmse_score = rmse(validation, forecast)
mae_score = mae(validation, forecast)

print(f"MAPE: {mape_score:.2f}")
print(f"RMSE: {rmse_score:.2f}")
print(f"MAE: {mae_score:.2f}")


# Plot actual and forecast values
plt.figure(figsize=(12, 6))
series.plot(label="Actual")
forecast.plot(label="Forecast")

plt.legend()
plt.title("Actual and Forecasted Values with AutoARIMA")
plt.xlabel("Time")
plt.ylabel("Passengers")
plt.show()
