import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ccxt
import time
from arch import arch_model

# Initialize KuCoin API via CCXT
exchange = ccxt.kucoin()

def fetch_kucoin_data(symbol="BTC/USDT", timeframe="5m", limit=200):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["Time", "Open", "High", "Low", "Close", "Volume"])
    df["Time"] = pd.to_datetime(df["Time"], unit='ms')
    df.set_index("Time", inplace=True)
    df = df.astype(float)
    return df

# Number of iterations
n_iterations = 10

# Fetch Data from KuCoin
data = fetch_kucoin_data()

data['Returns'] = data['Close'].pct_change().apply(lambda x: np.log(1 + x))
data.dropna(inplace=True)

# Scale Returns by 100 for better GARCH performance
data['Returns'] = data['Returns'] * 100  

# ATR Calculation
def calculate_atr(data, window=14):
    hl = data['High'] - data['Low']
    hc = abs(data['High'] - data['Close'].shift(1))
    lc = abs(data['Low'] - data['Close'].shift(1))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=window, adjust=False).mean()

data['ATR'] = calculate_atr(data)

# RSI Calculation
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = calculate_rsi(data)

# Fit GARCH(1,1) Model
model = arch_model(data['Returns'], vol='Garch', p=1, q=1, rescale=False)
model_fit = model.fit(disp='off')

# Define Forecast Horizons
horizons = [19, 21, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 51, 55, 59, 61, 65, 69, 
            71, 75, 79, 81, 85, 89, 91, 95, 99, 103, 105, 109, 115, 119, 125, 129, 135, 139, 
            145, 149, 155, 159, 165, 175, 185, 200]
forecast_sums = np.zeros(7)

for _ in range(n_iterations):
    btdr_scores = {}
    forecasted_prices_dict = {}
    last_price = data['Close'].iloc[-1]
    mu = model_fit.params['mu']
    
    for horizon in horizons:
        forecast = model_fit.forecast(horizon=7)
        forecast_variances = forecast.variance.values[-1]
        forecast_volatilities = np.sqrt(forecast_variances)
        forecasted_prices = []
        current_price = last_price
        
        for i in range(7):
            forecasted_return = (
                mu / 100 + (forecast_volatilities[i] if i < len(forecast_volatilities) else forecast_volatilities[-1])
                * np.random.normal()
            ) / 100  # Rescale forecasted return back to original scale
            
            current_price *= np.exp(forecasted_return)
            forecasted_prices.append(current_price)
        
        forecasted_prices_dict[horizon] = forecasted_prices
        
        price_drop_ratio = (np.array(forecasted_prices) - np.min(forecasted_prices)) / (
            np.max(forecasted_prices) - np.min(forecasted_prices) + 1e-9
        )
        bullish_rsi_factor = data['RSI'].iloc[-7:].values / 100
        atr_factor = data['ATR'].iloc[-7:].values / np.max(data['ATR'].iloc[-7:])
        
        btdr_score = ((1 - price_drop_ratio) * 25) + (bullish_rsi_factor * 25) + (atr_factor * 50)
        btdr_scores[horizon] = np.mean(btdr_score)
    
    best_horizon = max(btdr_scores, key=btdr_scores.get)
    forecast_sums += np.array(forecasted_prices_dict[best_horizon])

final_forecast = forecast_sums / n_iterations

# Save Final Forecast to Excel
forecast_data = {"Period": list(range(1, 8)), "Final Forecast": final_forecast}
forecasted_prices_df = pd.DataFrame(forecast_data)
output_file = r"C:\Users\oliva\OneDrive\Documents\Excel doc\Quantum_Forecast_Output.xlsx"
forecasted_prices_df.to_excel(output_file, index=False, engine='openpyxl')
print(f"Final forecast saved to {output_file}")

# Trading Recommendation & Explanation
current_rsi = data['RSI'].iloc[-1]
current_atr = data['ATR'].iloc[-1]
forecasted_trend = "Bullish" if final_forecast[-1] > last_price else "Bearish"

if forecasted_trend == "Bullish" and current_rsi < 50:
    recommendation = "BUY"
    reason = "RSI is below 50, indicating an oversold market, and forecast suggests an uptrend."
elif forecasted_trend == "Bearish" and current_rsi > 50:
    recommendation = "SELL"
    reason = "RSI is above 50, indicating an overbought market, and forecast suggests a downtrend."
else:
    recommendation = "HOLD"
    reason = "Market conditions are neutral or mixed; best to wait for clearer signals."

print(f"Recommended Action: {recommendation}")
print(f"Reason: {reason}")

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(range(1, 8), final_forecast, label="Final Forecast", linestyle='dashed', color='blue')
plt.xlabel('Period')
plt.ylabel('Price')
plt.title('Final Quantum-Inspired Forecast (7-Period Horizon)')
plt.legend()
plt.grid()
plt.show()
