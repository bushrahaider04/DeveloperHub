# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 11:11:17 2026

@author: Waseem Haider
"""

# Task 3: Heart Disease Prediction
# AI/ML Engineering Internship – DevelopersHub Corporation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# 1. Load and Explore the Dataset
# -------------------------------
print("=" * 60)
print("HEART DISEASE PREDICTION MODEL")
print("=" * 60)

# Load the dataset (UCI Heart Disease dataset)
# You can download from Kaggle or use this URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

try:
    heart_data = pd.read_csv(url, names=column_names, na_values='?')
except:
    # Fallback: try to load from local file if available
    heart_data = pd.read_csv('heart.csv')
    if 'target' not in heart_data.columns:
        print("Warning: Target column not found. Please check your dataset.")
        exit()

print("\n=== Dataset Shape ===")
print(f"Rows: {heart_data.shape[0]}, Columns: {heart_data.shape[1]}")

print("\n=== Column Names ===")
print(heart_data.columns.tolist())

print("\n=== First 5 rows ===")
print(heart_data.head())

print("\n=== Data Types and Non-null Counts ===")
print(heart_data.info())

print("\n=== Descriptive Statistics ===")
print(heart_data.describe())

# Check for missing values
print("\n=== Missing Values ===")
missing_values = heart_data.isnull().sum()
print(missing_values[missing_values > 0] if any(missing_values > 0) else "No missing values found!")

# -------------------------------
# 2. Data Cleaning
# -------------------------------
print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

# Handle missing values (if any)
if heart_data.isnull().sum().sum() > 0:
    print("Handling missing values...")
    # For numerical columns, fill with median
    for column in heart_data.columns:
        if heart_data[column].dtype in ['float64', 'int64']:
            median_val = heart_data[column].median()
            heart_data[column].fillna(median_val, inplace=True)
            print(f"  Filled missing values in '{column}' with median: {median_val:.2f}")
    
    # Check if any missing values remain
    print(f"\nRemaining missing values: {heart_data.isnull().sum().sum()}")
else:
    print("No missing values to handle.")

# Convert target to binary (0 = no heart disease, 1 = heart disease)
# In some versions, target values are 0-4, we need to convert to binary
if heart_data['target'].nunique() > 2:
    print("\nConverting target to binary (0 = no disease, 1 = disease)...")
    # If target > 0, it indicates presence of heart disease
    heart_data['target'] = (heart_data['target'] > 0).astype(int)
    print(f"  Target values after conversion: {heart_data['target'].unique()}")

print(f"\nData shape after cleaning: {heart_data.shape}")
print(f"Class distribution:\n{heart_data['target'].value_counts()}")

# -------------------------------
# 3. Exploratory Data Analysis (EDA)
# -------------------------------
print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# --- a) Target Variable Distribution ---
plt.figure(figsize=(8, 6))
heart_data['target'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Heart Disease (0 = No Disease, 1 = Disease)', fontsize=14)
plt.xlabel('Target')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# --- b) Correlation Heatmap ---
plt.figure(figsize=(14, 12))
correlation_matrix = heart_data.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of Heart Disease Features', fontsize=16)
plt.tight_layout()
plt.show()

# --- c) Distribution of Numerical Features ---
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, feature in enumerate(numeric_features):
    # Histogram with KDE
    axes[i].hist(heart_data[heart_data['target']==0][feature], bins=20, alpha=0.5, label='No Disease', color='skyblue')
    axes[i].hist(heart_data[heart_data['target']==1][feature], bins=20, alpha=0.5, label='Disease', color='salmon')
    axes[i].set_title(f'Distribution of {feature}', fontsize=12)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')
    axes[i].legend()

# Remove empty subplot
fig.delaxes(axes[5])
plt.tight_layout()
plt.show()

# --- d) Box Plots for Key Features ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, feature in enumerate(numeric_features):
    if i < len(numeric_features):
        sns.boxplot(x='target', y=feature, data=heart_data, ax=axes[i], palette=['skyblue', 'salmon'])
        axes[i].set_title(f'{feature} by Target', fontsize=12)
        axes[i].set_xlabel('Target (0=No Disease, 1=Disease)')
        axes[i].set_ylabel(feature)

fig.delaxes(axes[5])
plt.tight_layout()
plt.show()

# --- e) Categorical Features Analysis ---
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, feature in enumerate(categorical_features):
    if i < len(categorical_features) and feature in heart_data.columns:
        # Create cross-tabulation
        cross_tab = pd.crosstab(heart_data[feature], heart_data['target'], normalize='index')
        cross_tab.plot(kind='bar', ax=axes[i], color=['skyblue', 'salmon'], stacked=True)
        axes[i].set_title(f'{feature} vs Target', fontsize=12)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Proportion')
        axes[i].legend(['No Disease', 'Disease'])
        axes[i].tick_params(axis='x', rotation=0)

# Remove empty subplot
fig.delaxes(axes[8])
plt.tight_layout()
plt.show()

# --- f) Pairplot for Key Features ---
sns.pairplot(heart_data[['age', 'thalach', 'chol', 'trestbps', 'target']], 
             hue='target', palette=['skyblue', 'salmon'], height=2.5)
plt.suptitle('Pairplot of Key Features', y=1.02, fontsize=14)
plt.show()

# -------------------------------
# 4. Feature Engineering
# -------------------------------
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Separate features and target
X = heart_data.drop('target', axis=1)
y = heart_data['target']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Feature scaling (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection using ANOVA F-test
selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Get selected feature indices and their scores
selected_indices = selector.get_support(indices=True)
feature_scores = selector.scores_
feature_names = X.columns

# Create feature importance DataFrame
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Score': feature_scores
}).sort_values('Score', ascending=False)

print("\n=== Top 10 Most Important Features (ANOVA F-test) ===")
print(feature_importance.head(10))

# -------------------------------
# 5. Model Training
# -------------------------------
print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=10)
}

results = {}
predictions = {}

# Train each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Store results
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    predictions[name] = y_pred
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy:.4f}")

# -------------------------------
# 6. Model Evaluation
# -------------------------------
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# --- a) Confusion Matrix ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (name, result) in enumerate(results.items()):
    cm = confusion_matrix(y_test, result['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                xticklabels=['No Disease', 'Disease'], 
                yticklabels=['No Disease', 'Disease'])
    axes[idx].set_title(f'Confusion Matrix - {name}', fontsize=12)
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# --- b) ROC Curves ---
plt.figure(figsize=(10, 8))

for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
    auc_score = roc_auc_score(y_test, result['y_pred_proba'])
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)

# Plot diagonal line
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)', linewidth=1, alpha=0.5)
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('ROC Curves for Heart Disease Prediction Models', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- c) Performance Metrics ---
print("\n=== Model Performance Metrics ===")
print("-" * 60)

for name, result in results.items():
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy_score(y_test, result['y_pred']):.4f}")
    print(f"  ROC-AUC Score: {roc_auc_score(y_test, result['y_pred_proba']):.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, result['y_pred'], target_names=['No Disease', 'Disease'], digits=4))

# -------------------------------
# 7. Feature Importance Analysis
# -------------------------------
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

# --- a) Logistic Regression Coefficients ---
logreg = results['Logistic Regression']['model']
coefficients = logreg.coef_[0]
feature_names = X.columns

# Create feature importance DataFrame for Logistic Regression
lr_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print("\n=== Logistic Regression Feature Importance (Absolute Coefficients) ===")
print(lr_importance[['Feature', 'Coefficient']].head(10))

# --- b) Decision Tree Feature Importance ---
dt = results['Decision Tree']['model']
dt_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': dt.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n=== Decision Tree Feature Importance ===")
print(dt_importance.head(10))

# --- c) Visualization of Feature Importance ---
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Logistic Regression - Top 10 features
top_lr = lr_importance.head(10)
axes[0].barh(top_lr['Feature'], top_lr['Coefficient'], color='steelblue')
axes[0].set_xlabel('Coefficient Value')
axes[0].set_title('Top 10 Features - Logistic Regression', fontsize=12)
axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
axes[0].grid(axis='x', alpha=0.3)

# Decision Tree - Top 10 features
top_dt = dt_importance.head(10)
axes[1].barh(top_dt['Feature'], top_dt['Importance'], color='forestgreen')
axes[1].set_xlabel('Importance Score')
axes[1].set_title('Top 10 Features - Decision Tree', fontsize=12)
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

# --- d) Summary of Important Features ---
print("\n=== Summary: Top Important Features Affecting Heart Disease Prediction ===")
print("-" * 60)

# Combine importance from both models
all_features = set(feature_names)
combined_importance = []

for feature in all_features:
    lr_score = np.abs(logreg.coef_[0][list(feature_names).index(feature)])
    dt_score = dt.feature_importances_[list(feature_names).index(feature)]
    # Normalize scores for fair comparison
    combined_importance.append({
        'Feature': feature,
        'LR_Coefficient_Abs': lr_score,
        'DT_Importance': dt_score,
        'Average_Score': (lr_score + dt_score) / 2
    })

importance_df = pd.DataFrame(combined_importance).sort_values('Average_Score', ascending=False)

print("\nCombined Feature Importance (Average of both models):")
print(importance_df[['Feature', 'Average_Score']].head(10))

# Interpret key features
print("\n=== Key Medical Insights ===")
print("-" * 60)
print("Based on feature importance analysis, the most critical factors for heart disease prediction are:")

important_features = importance_df.head(5)['Feature'].tolist()
feature_descriptions = {
    'thalach': 'Maximum heart rate achieved (lower values indicate higher risk)',
    'exang': 'Exercise induced angina (1 = yes, 0 = no)',
    'oldpeak': 'ST depression induced by exercise relative to rest (higher values = higher risk)',
    'age': 'Age of the patient (older = higher risk)',
    'cp': 'Chest pain type (higher types indicate more severe pain)',
    'ca': 'Number of major vessels colored by fluoroscopy (higher = higher risk)',
    'thal': 'Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)',
    'slope': 'Slope of peak exercise ST segment (2 = upsloping, 1 = flat, 0 = downsloping)',
    'chol': 'Serum cholesterol in mg/dl (higher = higher risk)',
    'trestbps': 'Resting blood pressure (higher = higher risk)'
}

for i, feature in enumerate(important_features, 1):
    desc = feature_descriptions.get(feature, "Important predictor of heart disease")
    print(f"{i}. {feature}: {desc}")

print("\n" + "=" * 60)
print("MODEL PREDICTION COMPLETE")
print("=" * 60)

# -------------------------------
# 8. Example Prediction
# -------------------------------
print("\n=== Example Predictions for Test Data ===")
print("-" * 60)

# Display some test predictions
n_samples = 5
test_indices = np.random.choice(len(X_test), n_samples, replace=False)

for idx in test_indices:
    actual = y_test.iloc[idx]
    lr_pred = results['Logistic Regression']['y_pred'][idx]
    dt_pred = results['Decision Tree']['y_pred'][idx]
    
    print(f"\nSample {idx}:")
    print(f"  Actual: {'Disease' if actual == 1 else 'No Disease'}")
    print(f"  Logistic Regression Prediction: {'Disease' if lr_pred == 1 else 'No Disease'}")
    print(f"  Decision Tree Prediction: {'Disease' if dt_pred == 1 else 'No Disease'}")

print("\n" + "=" * 60)
print("PROGRAM COMPLETED SUCCESSFULLY")
print("=" * 60)