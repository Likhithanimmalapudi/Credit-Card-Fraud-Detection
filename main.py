import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load Kaggle Credit Card Fraud Detection dataset
df = pd.read_csv("creditcard.csv")  # Ensure the dataset is downloaded

# Explore dataset
print("First 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Check class distribution
print("\nFraudulent vs Non-Fraudulent Transactions:")
print(df['Class'].value_counts())

# Data Preprocessing
# Handling missing values
df.fillna(method='ffill', inplace=True)

# Encoding categorical features (if applicable)
if 'user_id' in df.columns:
    df['user_id'] = LabelEncoder().fit_transform(df['user_id'])
if 'merchant' in df.columns:
    df['merchant'] = LabelEncoder().fit_transform(df['merchant'])

# Feature selection
selected_features = ['Amount', 'Time']  # Select relevant features
if 'user_id' in df.columns:
    selected_features.append('user_id')
if 'merchant' in df.columns:
    selected_features.append('merchant')

X = df[selected_features]
y = df['Class']

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model training
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', max_depth=10, min_samples_split=5)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
importances = model.feature_importances_
feature_names = selected_features
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# Visualizing Fraudulent vs Non-Fraudulent Transactions
plt.figure(figsize=(8, 6))
sns.histplot(df[df['Class'] == 0]['Amount'], bins=50, color='blue', label='Legitimate', kde=True)
sns.histplot(df[df['Class'] == 1]['Amount'], bins=50, color='red', label='Fraudulent', kde=True)
plt.legend()
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.show()

# Save Model
joblib.dump(model, 'fraud_detection_model.pkl')
print("\nModel saved as 'fraud_detection_model.pkl'")
