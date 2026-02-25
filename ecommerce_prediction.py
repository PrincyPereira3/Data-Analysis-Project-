"""
E-commerce Furniture Sales Prediction with EDA
Python 3.13 Compatible
------------------------------------------------
This script:
1. Loads the dataset
2. Cleans and preprocesses data
3. Performs EDA visualizations
4. Engineers features (discount %, TF-IDF for productTitle)
5. Splits data into train/test sets
6. Trains Linear Regression and Random Forest models
7. Evaluates models using MSE and R²
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# =============================
# Step 1: Load Dataset
# =============================
df = pd.read_csv("ecommerce_furniture_dataset_2024.csv")

# =============================
# Step 2: Data Preprocessing
# =============================
df['price'] = df['price'].replace(r'[\$,]', '', regex=True).astype(float)
df['originalPrice'] = df['originalPrice'].replace(r'[\$,]', '', regex=True)
df['originalPrice'] = df['originalPrice'].replace('', np.nan)
df['originalPrice'] = df['originalPrice'].astype(float)
df['originalPrice'].fillna(df['price'], inplace=True)
df.dropna(subset=['sold'], inplace=True)

# =============================
# Step 3: Feature Engineering
# =============================
df['discount_percentage'] = ((df['originalPrice'] - df['price']) / df['originalPrice']) * 100

# =============================
# Step 4: EDA Visualizations
# =============================
# Distribution of Price
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=30, kde=True)
plt.title('Distribution of Price')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Distribution of Sold
plt.figure(figsize=(10, 6))
sns.histplot(df['sold'], bins=30, kde=True)
plt.title('Distribution of Sold Items')
plt.xlabel('Sold')
plt.ylabel('Frequency')
plt.show()

# Scatter plot: Price vs Sold
plt.figure(figsize=(10, 6))
sns.scatterplot(x='price', y='sold', data=df)
plt.title('Price vs Sold')
plt.xlabel('Price')
plt.ylabel('Sold')
plt.show()

# Countplot for TagText
plt.figure(figsize=(12, 6))
sns.countplot(y='tagText', data=df, order=df['tagText'].value_counts().index)
plt.title('Count of TagText Categories')
plt.xlabel('Count')
plt.ylabel('TagText')
plt.show()

# =============================
# Step 5: Modeling
# =============================
X = df[['productTitle', 'price', 'originalPrice', 'discount_percentage', 'tagText']]
y = df['sold']

tfidf = TfidfVectorizer(max_features=100)
categorical_features = ['tagText']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('tfidf', tfidf, 'productTitle'),
        ('cat', categorical_transformer, categorical_features),
        ('num', 'passthrough', ['price', 'originalPrice', 'discount_percentage'])
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linreg_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', LinearRegression())])
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestRegressor(n_estimators=100, random_state=42))])

linreg_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)

y_pred_linreg = linreg_pipeline.predict(X_test)
y_pred_rf = rf_pipeline.predict(X_test)

linreg_mse = mean_squared_error(y_test, y_pred_linreg)
linreg_r2 = r2_score(y_test, y_pred_linreg)
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)

print("Linear Regression Performance:")
print(f"MSE: {linreg_mse:.2f}, R²: {linreg_r2:.4f}")

print("\nRandom Forest Performance:")
print(f"MSE: {rf_mse:.2f}, R²: {rf_r2:.4f}")
