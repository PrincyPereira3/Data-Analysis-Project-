# OCD Patient Dataset Analysis and Medication Prediction
# Python 3.13 Compatible

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# STEP 1: Load Dataset
# -----------------------------
file_path = "OCD Patient Dataset_ Demographics & Clinical Data.csv"
df = pd.read_csv(file_path)

print("Initial Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# -----------------------------
# STEP 2: Data Cleaning
# -----------------------------
# Fill missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# Preserve original Gender column for EDA
gender_original = df['Gender'].copy()

# Encode categorical variables for modeling
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("\nMissing Values After Cleaning:\n", df.isnull().sum())

# -----------------------------
# STEP 3: Exploratory Data Analysis
# -----------------------------
print("\nDescriptive Statistics:\n", df.describe())

# Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.show()

# Gender Distribution (using original labels)
plt.figure(figsize=(6, 4))
sns.countplot(x=gender_original)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Ethnicity Distribution (decode back for readability)
ethnicity_labels = label_encoders['Ethnicity'].inverse_transform(df['Ethnicity'])
plt.figure(figsize=(8, 5))
sns.countplot(y=ethnicity_labels)
plt.title('Ethnicity Distribution')
plt.xlabel('Count')
plt.ylabel('Ethnicity')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# -----------------------------
# STEP 4: Predictive Modeling
# -----------------------------
# Target: Medications
X = df.drop('Medications', axis=1)
y = df['Medications']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cv_score = cross_val_score(model, X_scaled, y, cv=5).mean()
    results[name] = {
        'Accuracy': round(acc * 100, 2),
        'CV Score': round(cv_score * 100, 2),
        'Confusion Matrix': cm.tolist()
    }

print("\nModel Performance:\n")
for model_name, metrics in results.items():
    print(f"{model_name}: Accuracy={metrics['Accuracy']}%, CV Score={metrics['CV Score']}%")
    print("Confusion Matrix:", metrics['Confusion Matrix'])
    print("-" * 50)

# -----------------------------
# STEP 5: Save Outputs
# -----------------------------
df.to_csv("cleaned_ocd_dataset.csv", index=False)
print("\nCleaned dataset saved as 'cleaned_ocd_dataset.csv'")
