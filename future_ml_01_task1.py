import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
import os
print("Libraries imported")
print(" All Libraries imported")

# Upload 'mock_kaggle.csv' to the Colab files sidebar before running this cell
DATA_PATH = "/content/mock_kaggle.csv"  # In Colab, uploaded files appear at /content/
# If running locally and the file is at /mnt/data/mock_kaggle.csv, update DATA_PATH accordingly.
if os.path.exists("/mnt/data/mock_kaggle.csv"):
    DATA_PATH = "/mnt/data/mock_kaggle.csv"

df = pd.read_csv(DATA_PATH)
print("Loaded:", DATA_PATH)
df.head(5)

# Ensure common column names: Date, Category, Sales
print("Columns:", df.columns.tolist())

# Try to auto-detect likely column names if different
possible_date_cols = [c for c in df.columns if 'date' in c.lower()]
possible_sales_cols = [c for c in df.columns if c.lower() in ['sales','revenue','amount','sale']]

print("Possible date cols:", possible_date_cols)
print("Possible sales cols:", possible_sales_cols)
print("Data loaded successfully")

# If your dataset uses different names, rename them here:
df = df.rename(columns={'data':'Date','venda':'Sales'})

# Add a dummy 'Category' column for now to enable category-based plots
df['Category'] = 'General'

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
print(df.info())

# Aggregate daily -> monthly sums (you can change to weekly or keep daily)
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['dayofweek'] = df['Date'].dt.dayofweek

monthly = df.set_index('Date').resample('M').sum(numeric_only=True).reset_index()
monthly['year'] = monthly['Date'].dt.year
monthly['month'] = monthly['Date'].dt.month

# Create lag and rolling features (for ML models)
monthly['lag_1'] = monthly['Sales'].shift(1)
monthly['lag_12'] = monthly['Sales'].shift(12)
monthly['rmean_3'] = monthly['Sales'].rolling(3).mean()
monthly = monthly.dropna().reset_index(drop=True)
monthly.head(5)
print("lag and rolling features (for ML models)")

# Prepare pivot for category-year stacked area plot
cat_year = df.copy()
cat_year['year'] = cat_year['Date'].dt.year
pivot = cat_year.pivot_table(values='Sales', index='year', columns='Category', aggfunc='sum', fill_value=0)
pivot.head()

plt.figure(figsize=(14,6))
plt.stackplot(pivot.index, [pivot[col] for col in pivot.columns], labels=pivot.columns)
plt.legend(loc='upper left')
plt.title("Category Performance Trends (Year Over Year)")
plt.xlabel("Year")
plt.ylabel("Revenue ($)")
plt.show()
print("pivot for category-year stacked area plot")

# Donut chart for sales by category
cat_sum = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
plt.figure(figsize=(6,6))
wedges, texts, autotexts = plt.pie(cat_sum, labels=cat_sum.index, autopct='%1.1f%%', startangle=140)
centre = plt.Circle((0,0),0.70,fc='white')
plt.gca().add_artist(centre)
plt.title("Sales by Category")
plt.show()
print(" Donut chart for sales by category")

# Prophet requires columns 'ds' and 'y'
prophet_df = df[['Date','Sales']].rename(columns={'Date':'ds','Sales':'y'})
# Aggregate to daily sums in case of duplicates
prophet_df = prophet_df.groupby('ds').sum().reset_index()

m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
m.fit(prophet_df)

# Forecast 365 days into the future (adjust to 12 months)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
print("# Forecast 365 days into the future (adjust to 12 months)")

# Plot forecast (Prophet built-in)
fig1 = m.plot(forecast, xlabel='Date', ylabel='Sales')
fig1.set_size_inches(20, 10)

fig2 = m.plot_components(forecast)
fig2.set_size_inches(20, 10)
print("Plot forecast (Prophet built-in)")

# Merge actuals and forecast for plotting
pred = forecast[['ds','yhat']].rename(columns={'ds':'Date'})
actuals = prophet_df.rename(columns={'ds':'Date','y':'Sales'})

merged = pd.merge(actuals, pred, on='Date', how='right').fillna(method='ffill')

plt.figure(figsize=(14,6))
plt.plot(merged['Date'], merged['Sales'], label='Historical sales', linewidth=2)
plt.plot(merged['Date'], merged['yhat'], '--', label='AI Prediction', linewidth=1.5)
plt.title("Sales Trend: Actual v/s Forecasted")
plt.xlabel("Date")
plt.ylabel("Revenue ($)")
plt.legend()
plt.show()
print("Merge actuals and forecast for plotting")
# Calculate projected sales for next 12 months from the forecast
future_only = forecast[forecast['ds'] > prophet_df['ds'].max()]
projected_next_12m = future_only['yhat'].head(365).sum()
print("Projected Sales in next 12 months (raw):", projected_next_12m)
print("Projected Sales in next 12 months (in Millions):", round(projected_next_12m/1e6,2), "M")
print("Calculate projected sales for next 12 months from the forecast")
# Prepare ML dataset using monthly aggregated features from earlier
ml = monthly.copy().dropna().reset_index(drop=True)
features = ['month','lag_1','lag_12','rmean_3']
X = ml[features]
y = ml['Sales']
print("Prepare ML dataset using monthly aggregated features from earlier")
# TimeSeriesSplit
ts = TimeSeriesSplit(n_splits=3)
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
scores = cross_val_score(model_rf, X, y, cv=ts, scoring='neg_mean_absolute_error')
print("CV MAE (RandomForest):", -scores.mean())

# Fit final model and predict last 12 months (example)
model_rf.fit(X, y)
ml['pred_rf'] = model_rf.predict(X)
print("Fit final model and predict last 12 months")
# Export forecast and historical CSVs for Power BI
forecast[['ds','yhat','yhat_lower','yhat_upper']].to_csv('forecast_output.csv', index=False)
df.to_csv('historical_sales.csv', index=False)
print("Exported forecast_output.csv and historical_sales.csv to current directory.")
print(" Export forecast and historical CSVs for Power BI")