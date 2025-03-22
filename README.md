# Credit Card Fraud Detection

## Overview
This project aims to build a machine learning model that can distinguish fraudulent transactions from legitimate ones using the Kaggle Credit Card Fraud Detection dataset. The goal is to minimize false positives while maximizing fraud detection accuracy.

## Dataset
- The dataset contains anonymized transaction details such as `Amount`, `Time`, and a `Class` label (1 for fraud, 0 for legitimate transactions).
- Additional categorical features like `user_id` and `merchant` are encoded if available.

## Features Used
- `Amount`: The transaction amount.
- `Time`: The time elapsed since the first transaction in the dataset.
- `user_id` (if available): Encoded user identifier.
- `merchant` (if available): Encoded merchant information.

## Preprocessing Steps
1. Handle missing values using forward fill (`ffill`).
2. Encode categorical variables (`user_id` and `merchant`).
3. Scale numerical features (`Amount` and `Time`) using `StandardScaler`.
4. Split dataset into training (80%) and testing (20%) using stratified sampling.

## Model
- **Algorithm:** Random Forest Classifier
- **Hyperparameters:**
  - `n_estimators=200`
  - `max_depth=10`
  - `min_samples_split=5`
  - `class_weight='balanced'`

## Evaluation Metrics
- **Accuracy**: Measures overall correctness.
- **Precision**: Reduces false positives.
- **Recall**: Maximizes fraud detection.
- **F1 Score**: Balances precision and recall.

## Visualizations
- **Feature Importance**: Identifies key attributes affecting fraud detection.
- **Transaction Distribution**: Shows differences between fraudulent and non-fraudulent transaction amounts.

## How to Run
1. Install dependencies:  
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn joblib
   ```
2. Run the script:  
   ```bash
   python fraud_detection.py
   ```
3. The trained model is saved as `fraud_detection_model.pkl`.

## Project Structure
```
├── fraud_detection.py    # Main fraud detection script
├── creditcard.csv        # Kaggle dataset (not included in repo)
├── fraud_detection_model.pkl  # Saved model
├── README.md             # Project documentation
```

## Acknowledgment
- Dataset from Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)


