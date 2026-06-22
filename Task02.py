# -*- coding: utf-8 -*-
"""
Task 2: Predict Future Stock Prices (Short-Term)
AI/ML Engineering Internship – DevelopersHub Corporation
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# 1. Select and Load Stock Data
# -------------------------------
stock_symbol = 'AAPL'  # Change to 'TSLA', 'MSFT', 'GOOGL', etc.

print(f"=== Loading Historical Data for {stock_symbol} ===")

# Download historical data (last 2 years)
stock_data = yf.download(stock_symbol,
                         start='2024-01-01',
                         end='2026-06-18',
                         progress=False)

print(f"\nData shape: {stock_data.shape}")
print(f"Columns: {stock_data.columns.tolist()}")
print(f"\nFirst few rows:")
print(stock_data.head())

# -------------------------------
# 2. Feature Engineering
# -------------------------------
print("\n=== Feature Engineering ===")

df = stock_data.copy()

# Technical indicators
df['MA_5'] = df['Close'].rolling(window=5).mean()
df['MA_10'] = df['Close'].rolling(window=10).mean()
df['MA_20'] = df['Close'].rolling(window=20).mean()

# Price changes and volatility
df['Price_Change'] = df['Close'].pct_change()
df['High_Low_Ratio'] = df['High'] / df['Low']
df['Close_Open_Ratio'] = df['Close'] / df['Open']

# Volume indicators
df['Volume_Change'] = df['Volume'].pct_change()
df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()

# Date features
df['Day_of_Week'] = df.index.dayofweek
df['Month'] = df.index.month

# Remove rows with NaN values
df = df.dropna()

print(f"Shape after feature engineering: {df.shape}")

# -------------------------------
# 3. Prepare Features and Target
# -------------------------------
print("\n=== Preparing Features and Target ===")

features = ['Open', 'High', 'Low', 'Volume',
            'MA_5', 'MA_10', 'MA_20',
            'Price_Change', 'High_Low_Ratio', 'Close_Open_Ratio',
            'Volume_Change', 'Volume_MA_5',
            'Day_of_Week', 'Month']

# Target: Next day's closing price
df['Target_Close'] = df['Close'].shift(-1)
df = df.dropna()

X = df[features]
y = df['Target_Close']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# -------------------------------
# 4. Train-Test Split (Time Series)
# -------------------------------
print("\n=== Splitting Data (Time Series Split) ===")

split_idx = int(len(X) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# -------------------------------
# 5. Scale Features
# -------------------------------
print("\n=== Scaling Features ===")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 6. Train Models
# -------------------------------
print("\n=== Training Models ===")

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100,
                                           random_state=42,
                                           n_jobs=-1)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        'model': model,
        'predictions': y_pred,
        'metrics': {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }
    }

    print(f"  MAE: ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  R² Score: {r2:.4f}")

# -------------------------------
# 7. Feature Importance (Random Forest)
# -------------------------------
print("\n=== Feature Importance (Random Forest) ===")

if 'Random Forest' in results:
    rf_model = results['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

# -------------------------------
# 8. PLOTTING - All Visualizations
# -------------------------------
print("\n=== Creating Visualizations ===")

sns.set_style('whitegrid')

# Get test dates
test_dates = df.index[split_idx:].values[:len(y_test)]

# ============================================
# PLOT 1: Actual vs Predicted Prices (Full View)
# ============================================
fig, axes = plt.subplots(2, 1, figsize=(15, 12))

for idx, (name, result) in enumerate(results.items()):
    ax = axes[idx]

    ax.plot(test_dates, y_test.values, label='Actual Price',
            color='blue', linewidth=2, alpha=0.7)
    ax.plot(test_dates, result['predictions'], label='Predicted Price',
            color='red', linewidth=2, alpha=0.7, linestyle='--')

    ax.set_title(f'{name} - Actual vs Predicted Stock Prices', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Closing Price ($)', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    metrics = result['metrics']
    text_str = f"MAE: ${metrics['MAE']:.2f} | RMSE: ${metrics['RMSE']:.2f} | R²: {metrics['R2']:.4f}"
    ax.text(0.02, 0.95, text_str, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('1_stock_prediction_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# PLOT 2: Scatter Plot (Actual vs Predicted)
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for idx, (name, result) in enumerate(results.items()):
    ax = axes[idx]

    ax.scatter(y_test, result['predictions'], alpha=0.6, edgecolors='black', linewidth=0.5)

    min_val = min(y_test.min(), result['predictions'].min())
    max_val = max(y_test.max(), result['predictions'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax.set_title(f'{name} - Actual vs Predicted', fontsize=14, fontweight='bold')
    ax.set_xlabel('Actual Price ($)', fontsize=12)
    ax.set_ylabel('Predicted Price ($)', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('2_stock_prediction_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# PLOT 3: Feature Importance
# ============================================
if 'Random Forest' in results:
    plt.figure(figsize=(12, 8))

    top_features = feature_importance.head(10)
    plt.barh(top_features['Feature'], top_features['Importance'],
             color='steelblue', edgecolor='black', alpha=0.8)

    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Top 10 Most Important Features (Random Forest)',
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('3_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================
# PLOT 4: Zoomed 30-Day View
# ============================================
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

for idx, (name, result) in enumerate(results.items()):
    ax = axes[idx]

    n_days = 30
    test_dates_zoom = test_dates[-n_days:]
    y_test_zoom = y_test.values[-n_days:]
    y_pred_zoom = result['predictions'][-n_days:]

    ax.plot(test_dates_zoom, y_test_zoom, label='Actual Price',
            color='blue', linewidth=2, marker='o', markersize=4)
    ax.plot(test_dates_zoom, y_pred_zoom, label='Predicted Price',
            color='red', linewidth=2, marker='s', markersize=4, linestyle='--')

    ax.set_title(f'{name} - Last 30 Days (Zoomed View)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Closing Price ($)', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xticks(rotation=45)

plt.tight_layout()
plt.savefig('4_stock_prediction_zoomed.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# PLOT 5: Residual Analysis
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

for idx, (name, result) in enumerate(results.items()):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]

    residuals = y_test.values - result['predictions']

    # Residuals over time
    ax.plot(test_dates, residuals, color='green', linewidth=1, alpha=0.7)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_title(f'{name} - Residuals Over Time', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Residuals ($)', fontsize=10)
    ax.grid(True, alpha=0.3)

# Plot histogram of residuals (if space available)
if len(results) >= 2:
    ax = axes[1, 1]
    for name, result in results.items():
        residuals = y_test.values - result['predictions']
        ax.hist(residuals, bins=20, alpha=0.5, label=name, density=True)
    ax.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Residuals ($)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('5_residual_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# PLOT 6: Model Performance Comparison
# ============================================
fig, ax = plt.subplots(figsize=(10, 6))

metrics_df = pd.DataFrame({
    name: results[name]['metrics']
    for name in results.keys()
}).T

metrics_df[['MAE', 'RMSE']].plot(kind='bar', ax=ax, color=['steelblue', 'coral'], edgecolor='black')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Error ($)', fontsize=12)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# Add R² values on the plot
for i, name in enumerate(results.keys()):
    r2 = results[name]['metrics']['R2']
    ax.text(i, metrics_df.loc[name, 'RMSE'] + 0.5, f'R²={r2:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('6_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# PLOT 7: Price Movement Prediction (Up/Down)
# ============================================
fig, axes = plt.subplots(2, 1, figsize=(15, 8))

for idx, (name, result) in enumerate(results.items()):
    ax = axes[idx]

    # Calculate actual and predicted price movements
    actual_movement = np.sign(y_test.values[1:] - y_test.values[:-1])
    pred_movement = np.sign(result['predictions'][1:] - result['predictions'][:-1])

    # Create confusion-like visualization
    correct = actual_movement == pred_movement
    accuracy = np.mean(correct) * 100

    # Plot movements
    dates = test_dates[1:]
    ax.plot(dates, actual_movement, label='Actual Movement',
            color='blue', marker='o', markersize=3, linestyle='-', alpha=0.7)
    ax.plot(dates, pred_movement, label='Predicted Movement',
            color='red', marker='s', markersize=3, linestyle='--', alpha=0.7)

    ax.set_title(f'{name} - Price Movement Prediction (Accuracy: {accuracy:.1f}%)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Movement Direction (+1 Up, -1 Down)', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yticks([-1, 0, 1])

plt.tight_layout()
plt.savefig('7_movement_prediction.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------
# 9. Summary Report
# -------------------------------
print("\n" + "="*60)
print("=== SUMMARY REPORT ===")
print("="*60)

print(f"\nStock Symbol: {stock_symbol}")
print(f"Data Period: {df.index.min()} to {df.index.max()}")
print(f"Total Samples: {len(df)}")
print(f"Train/Test Split: {len(X_train)} / {len(X_test)}")

print("\nModel Performance:")
for name, result in results.items():
    metrics = result['metrics']
    print(f"\n  {name}:")
    print(f"    MAE:  ${metrics['MAE']:.2f}")
    print(f"    RMSE: ${metrics['RMSE']:.2f}")
    print(f"    R²:   {metrics['R2']:.4f}")

best_model = max(results.keys(), key=lambda x: results[x]['metrics']['R2'])
print(f"\n🏆 Best Model (based on R²): {best_model}")

print("\nTop 5 Most Important Features (Random Forest):")
print(feature_importance.head(5).to_string(index=False))

print("\n" + "="*60)
print("=== PLOTS GENERATED ===")
print("="*60)
print("1. 1_stock_prediction_comparison.png - Full prediction comparison")
print("2. 2_stock_prediction_scatter.png - Actual vs Predicted scatter")
print("3. 3_feature_importance.png - Feature importance visualization")
print("4. 4_stock_prediction_zoomed.png - 30-day zoomed view")
print("5. 5_residual_analysis.png - Residual analysis")
print("6. 6_model_comparison.png - Model comparison bar chart")
print("7. 7_movement_prediction.png - Price movement prediction")

print("\n" + "="*60)
print("=== END OF ANALYSIS ===")
print("="*60)