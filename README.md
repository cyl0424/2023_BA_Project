# 2023_BA_Project

This is team project of 2023 Fall Business Analytics

- [2023_BA_Project](#2023_ba_project)
  - [Usage](#usage)
    - [1. Load and Preprocess Data](#1-load-and-preprocess-data)
    - [2. Training and Model Creation](#2-training-and-model-creation)
    - [3. Ensemble Prediction](#3-ensemble-prediction)
    - [4. Prediction for New Data](#4-prediction-for-new-data)
    - [5. Results Storage](#5-results-storage)
  - [Execution Environment](#execution-environment)
  - [Notes](#notes)

## Usage

### 1. Load and Preprocess Data

- Install the required Python libraries:

  ```bash
  pip install lightgbm xgboost scikit-learn pandas numpy
  ```

- Open the Jupyter Notebook:

  ```bash
  jupyter notebook
  ```

- Load the data using the `merged_data.csv` file:

  ```python
  import pandas as pd

  # Load data
  data = pd.read_csv('merged_data.csv', encoding='utf-8')
  ```

- Select the necessary columns and convert categorical data into numerical format:

  ```python
  # Select relevant columns
  selected_features = ['stn_id', 'borrowed_hour', 'borrowed_day', ...]
  data = data[selected_features]

  # Convert categorical features to numerical
  categorical_features = ['stn_id', 'stn_gu', 'nearby_id']
  for feature in categorical_features:
      data[feature] = pd.factorize(data[feature])[0]
  ```

### 2. Training and Model Creation

- **Train LightGBM model:**

  ```python
  import lightgbm as lgb

  lgb_params = {
      'objective': 'regression',
      'metric': 'rmse',
      'boosting_type': 'gbdt',
      'num_leaves': 80,
      'learning_rate': 0.1,
      'feature_fraction': 1.0
  }

  # Dart: to improve the accuracy
  lgb_params = {
      'objective': 'regression',
      'metric': 'rmse',
      'boosting_type': 'dart',
      'num_leaves': 80,
      'learning_rate': 0.1,
      'feature_fraction': 1.0
  }

  train_data_lgb = lgb.Dataset(X_train, label=y_train)
  test_data_lgb = lgb.Dataset(X_test, label=y_test, reference=train_data_lgb)

  lgb_model = lgb.train(lgb_params, train_data_lgb, num_boost_round=100000, valid_sets=[test_data_lgb, train_data_lgb], callbacks=[
      lgb.early_stopping(stopping_rounds=3, verbose=100),
  ])
  ```

- **Train XGBoost model:**

  ```python
  import xgboost as xgb

  xgb_params = {
      'objective': 'reg:squarederror',
      'eval_metric': 'rmse',
      'booster': 'gbtree',
      'learning_rate': 0.1,
      'max_depth': 13,
      'subsample':0.8
  }

  train_data_xgb = xgb.DMatrix(X_train, label=y_train)
  test_data_xgb = xgb.DMatrix(X_test, label=y_test)

  xgb_model = xgb.train(xgb_params, train_data_xgb, num_boost_round=100000, evals=[(test_data_xgb, 'eval')], early_stopping_rounds=3, verbose_eval=100)
  ```

- **Evaluate the performance of the trained models:**

  ```python
  # LightGBM Performance Evaluation
  # ...

  # XGBoost Performance Evaluation
  # ...
  ```

### 3. Ensemble Prediction

- **Combine the predictions of both models to generate the final ensemble prediction:**

  ```python
  # Ensemble Prediction
  # ...
  ```

### 4. Prediction for New Data

- **Load and preprocess new data, such as using the `new_data.csv` file:**

  ```python
  # Load new data
  new_data = pd.read_csv('new_data.csv', encoding='utf-8')
  ```

- **Predict using both LightGBM and XGBoost models and create an ensemble prediction for the new data:**

  ```python
  # Prediction for New Data
  # ...
  ```

### 5. Results Storage

- **Save the predictions in the `new_data_with_predictions.csv` file:**

  ```python
  # Save Predictions
  new_data.to_csv('new_data_with_predictions.csv', index=False, encoding='utf-8')
  ```

## Execution Environment

- Python 3.11 or higher

## Notes

- The `new_data.csv` file must be provided by the user.
- If additional configurations or modifications are needed, feel free to adjust the code accordingly.

For more detailed information, refer to the code comments and consider exploring the documentation of LightGBM, XGBoost, and scikit-learn.
