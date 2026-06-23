# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 11:11:17 2026

@author: Waseem Haider
"""

# Task 1: Exploring and Visualizing the Iris Dataset
# AI/ML Engineering Internship – DevelopersHub Corporation

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load the dataset
# -------------------------------
# The Iris dataset is built into seaborn; alternatively, you can load it from a CSV file.
iris = sns.load_dataset('iris')   # returns a pandas DataFrame

# -------------------------------
# 2. Inspect the dataset
# -------------------------------
print("=== Shape of the dataset ===")
print(iris.shape)                # (150, 5)

print("\n=== Column names ===")
print(iris.columns.tolist())

print("\n=== First few rows (.head()) ===")
print(iris.head())

print("\n=== Data types and non-null counts (.info()) ===")
iris.info()   # prints directly to console

print("\n=== Descriptive statistics (.describe()) ===")
print(iris.describe())

# -------------------------------
# 3. Visualizations
# -------------------------------

# Set a clean visual style
sns.set_style('whitegrid')

# --- a) Scatter plot: relationship between sepal length and sepal width, colored by species ---
plt.figure(figsize=(8, 6))
sns.scatterplot(data=iris, x='sepal_length', y='sepal_width', hue='species', palette='deep', s=70)
plt.title('Sepal Length vs Sepal Width by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend(title='Species')
plt.tight_layout()
plt.show()

# Optional: pairplot to see all feature relationships at once
# sns.pairplot(iris, hue='species')
# plt.show()

# --- b) Histograms: distribution of each numeric feature ---
# Extract numeric columns (exclude the categorical 'species' column)
numeric_cols = iris.select_dtypes(include='number').columns.tolist()

plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 2, i)   # 2x2 grid of subplots
    sns.histplot(data=iris, x=col, kde=True, hue='species', palette='deep', alpha=0.5)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
plt.tight_layout()
plt.show()

# --- c) Box plots: identify outliers in each feature, split by species ---
plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data=iris, x='species', y=col, palette='deep')
    plt.title(f'Box plot of {col} by Species')
plt.tight_layout()
plt.show()