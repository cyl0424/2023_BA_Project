# 2023_BA_Project

This repository is a team project for the 2023 Fall Business Analytics course in Seoul Nataional University of Science and Technology. The project aims to analyze the Seoul Public Bike(따릉이) project and develop strategies for deficit reduction. The analysis involves using machine learning models, specifically LightGBM and XGBoost, to predict public bicycle usage patterns and optimize system efficiency.

- [2023_BA_Project](#2023-ba-project)
  - [Project Overview](#project-overview)
  - [Objectives](#objectives)
  - [Execution Environment](#execution-environment)
  - [Model Execution](#model-execution)
    - [Model Versions](#model-versions)
    - [Create and Activate Conda Environment](#create-and-activate-conda-environment)
    - [Install Required Packages](#install-required-packages)
    - [Launch Jupyter Notebook](#launch-jupyter-notebook)
    - [1. Load and Preprocess Data](#1-load-and-preprocess-data)
    - [2. Training and Model Creation](#2-training-and-model-creation)
      - [2.1 Train LightGBM Model](#21-train-lightgbm-model)
      - [2.2 Train XGBoost Model](#22-train-xgboost-model)
    - [3. Ensemble Prediction](#3-ensemble-prediction)
    - [4. Deficit Reduction Strategies](#4-deficit-reduction-strategies)
      - [4.1 Analysis and Insights](#41-analysis-and-insights)
      - [4.2 Propose Strategies for Deficit Reduction](#42-propose-strategies-for-deficit-reduction)
    - [5. Results Storage](#5-results-storage)
    - [6. Model Performance](#6-model-performance)
      - [LightGBM Model](#lightgbm-model)
      - [XGBoost Model](#xgboost-model)
      - [Ensemble Model](#ensemble-model)
    - [7. Additional Notes and Considerations](#7-additional-notes-and-considerations)
      - [7.1 Hyperparameter Tuning Details](#71-hyperparameter-tuning-details)
    - [8. Dataset](#8-dataset)
      - [8.1 Preprocessing Data](#81-preprocessing-data)
      - [8.2 Training Dataset](#81-training-dataset)
      - [8.3 Collecting Real-time Rental Data (Optional)](#83-collecting-real-time-rental-data--optional-)

## Project Overview

The Seoul Public Bike project has faced financial challenges, with deficits increasing over the years. The goal of this project is to leverage business analytics to understand the usage patterns, optimize system efficiency, and propose strategies for deficit reduction.

## Objectives

1. **Financial Analysis:** Conduct a comprehensive analysis of the Seoul Public Bike project's financial status, identifying trends, and understanding the factors contributing to deficits.

2. **Usage Pattern Analysis:** Explore usage patterns of public bicycles, considering factors such as borrowed hour, borrowed day, and environmental conditions.

3. **Model Development:** Implement machine learning models, including LightGBM and XGBoost, to predict bicycle usage and optimize station-specific trends.

4. **Ensemble Model:** Combine the strengths of LightGBM and XGBoost through ensemble modeling to enhance prediction accuracy.

5. **Deficit Reduction Strategies:** Based on the analysis results, propose strategies to reduce the financial deficits associated with the public bicycle project.

## Execution Environment

- Python 3.11 or higher
- Conda (for managing the virtual environment)

## Model Execution

### Model Versions

- pandas==1.5.3
- numpy==1.24.3
- scikit-learn==1.3.0
- lightgbm==4.1.0
- xgboost==2.0.2

### Create and Activate Conda Environment

1. **Create Conda Environment:**

   ```bash
   conda create --name myenv python=3.11
   ```

   Replace `myenv` with the desired environment name.

2. **Activate Conda Environment:**
   ```bash
   conda activate myenv
   ```

### Install Required Packages

```bash
conda install --file requirements.txt
```

This command installs the necessary packages specified in the `requirements.txt` file within the Conda environment.

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Now, open the Jupyter Notebook and navigate to the `team7_ensemble_model.ipynb` notebook to run the code under the "1. Load and Preprocess Data" section.

Ensure that you are using Python 3.11 or a higher version and have activated your Conda environment before installing the required packages.

### 1. Load and Preprocess Data

1. **Load the Data:**

   ```python
   import pandas as pd

   # Load data
   data = pd.read_csv('merged_data.csv', encoding='utf-8')
   ```

2. **Select Relevant Features and Preprocess Data:**

   ```python
   # Selected Features
   selected_features = ['stn_id', 'borrowed_hour', 'borrowed_day', 'is_holiday', 'borrowed_num_nearby', '강수량(mm)', 'wind_chill', 'nearby_id', 'borrowed_date', 'borrowed_num']
   data = data[selected_features]

   # Label Encoding for Categorical Features
   categorical_features = ['stn_id', 'nearby_id']
   for feature in categorical_features:
       data[feature] = pd.factorize(data[feature])[0]
   ```

### 2. Training and Model Creation

#### 2.1 Train LightGBM Model

```python
import lightgbm as lgb

# LightGBM Parameters for Regression
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 80,
    'learning_rate': 0.05,
    'feature_fraction': 1.0,
    'device': 'gpu'
}

# Create training and test datasets
train_data_lgb = lgb.Dataset(X_train, label=y_train)
test_data_lgb = lgb.Dataset(X_test, label=y_test, reference=train_data_lgb)

# Train the LightGBM model
lgb_model = lgb.train(lgb_params, train_data_lgb, num_boost_round=10000, valid_sets=[test_data_lgb, train_data_lgb], callbacks=[
    lgb.early_stopping(stopping_rounds=3, verbose=100),
])
```

#### 2.2 Train XGBoost Model

```python
import xgboost as xgb

# XGBoost Parameters for Regression
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'booster': 'gbtree',
    'learning_rate': 0.1,
    'max_depth': 13,
    'subsample': 0.8,
    'device': 'gpu'
}

# Create training and test datasets
train_data_xgb = xgb.DMatrix(X_train, label=y_train)
test_data_xgb = xgb.DMatrix(X_test, label=y_test)

# Train the XGBoost model
xgb_model = xgb.train(xgb_params, train_data_xgb, num_boost_round=10000, evals=[(test_data_xgb, 'eval')], early_stopping_rounds=3, verbose_eval=100)
```

### 3. Ensemble Prediction

```python
# Combine predictions of both models for ensemble prediction
y_pred_ensemble = (lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) + xgb_model.predict(test_data_xgb)) / 2

# Evaluate the performance of the ensemble model
ensemble_rmse = mean_squared_error(y_test, y_pred_ensemble, squared=False)
ensemble_r2 = r2_score(y_test, y_pred_ensemble)

print(f'Ensemble Test RMSE: {ensemble_rmse}')
print(f'Ensemble Test R-squared: {ensemble_r2}')
```

### 4. Deficit Reduction Strategies

#### 4.1 Analysis and Insights

Based on the ensemble model results, analyze patterns and insights obtained from the predictions.

#### 4.2 Propose Strategies for Deficit Reduction

Considering the analysis, propose effective strategies to reduce the financial deficits associated with the Seoul Public Bike project.

### 5. Results Storage

Save the predictions in the `new_data_with_predictions.csv` file:

```python
# Save Predictions
new_data.to_csv('new_data_with_predictions.csv', index=False, encoding='utf-8')
```

### 6. Model Performance

After training and evaluating the LightGBM and XGBoost models, here are the key performance metrics:

#### LightGBM Model

| Metric    | Training Value | Test Value |
| --------- | -------------- | ---------- |
| RMSE      | 1.9139         | 1.9659     |
| R-squared | 0.5621         | 0.5377     |

#### XGBoost Model

| Metric    | Training Value | Test Value |
| --------- | -------------- | ---------- |
| RMSE      | 1.7220         | 1.9135     |
| R-squared | 0.6455         | 0.5620     |

#### Ensemble Model

| Metric    | Training Value | Test Value |
| --------- | -------------- | ---------- |
| RMSE      | 1.5199         | 1.7171     |
| R-squared | 0.7128         | 0.6473     |

These metrics provide insights into how well the models are performing, and users can quickly assess the quality of predictions.

### 7. Additional Notes and Considerations

Include any additional details, configurations, or modifications needed for the code. Clarify that the 'device' parameter is optional and can be adjusted based on the user's environment.

#### 7.1 Hyperparameter Tuning Details

For detailed information about the hyperparameter tuning process for XGBoost and LightGBM, including the configurations used and insights gained, please refer to the [23_BA_preprocessing](https://github.com/jeonghyeonee/23_BA_preprocessing) repository.

The hyperparameter tuning results and analysis can be found in the [Hyperparameter Tuning](https://github.com/jeonghyeonee/23_BA_preprocessing) section of the `23_BA_preprocessing` repository.

### 8. Dataset

#### 8.1 [Preprocessing Data](<(https://github.com/cyl0424/BA_Preprocessing)>)

For the preprocessing of Seoul Bike Rental Station Information, you can refer to the [BA_Preprocessing](https://github.com/cyl0424/BA_Preprocessing) repository. The preprocessing repository includes the following files:

- **seoul_bicycle_master.json**: Master data of Seoul Bike rental stations.
- **master_preprocessing.ipynb**: Jupyter Notebook for normalizing the coordinates in the master data where they are recorded as 0.0 using the Google API.
- **seoul_bicycle_maser_preprocessed.csv**: File containing data processed using master_preprocessing.ipynb.
- **master_info_with_nearby.ipynb**: Jupyter Notebook for adding columns of data for the nearest rental station and its distance using seoul_bicycle_maser_preprocessed.csv.
- **master_info_with_nearby.csv**: File containing data with added information about nearby rental stations using master_info_with_nearby.ipynb.
- **master_final.ipynb**: Jupyter Notebook for processing rows where the district data has not been correctly recorded due to differences in address formatting.
- **master_final.csv**: File containing data where 'stn_gu' has been appropriately added to all data using master_final.ipynb.

Columns in master_final.csv:

- **stn_id**: Represents the id of the rental station and is of object type.
- **stn_addr**: Represents the full address of the rental station and is of object type.
- **stn_lat**: Represents the latitude of the rental station and is of float64 type.
- **stn_lng**: Represents the longitude of the rental station and is of float64 type.
- **nearby_id**: Represents the id of the nearest rental station and is of object type.
- **nearby_km**: Represents the distance to the nearest rental station in km and is of float64 type.
- **stn_gu**: A district data column was added for analysis as the weather data classification is done by district. This is of object type.

[BA_Preprocessing Repository](https://github.com/cyl0424/BA_Preprocessing)

#### 8.2 Training Dataset

To replicate the analysis and run the code, you'll need the dataset file `merged_data.csv`. You can download it using the following link:

[Download merged_data.csv](https://drive.google.com/file/d/1xy6S4VpBw3NBsyTTBwaETZFjAlOM-tMc/view)

Place the downloaded file in the project's root directory before running the Jupyter Notebook.

#### 8.3 Collecting Real-time Rental Data (Optional)

If you want to collect real-time Seoul Public Bike rental data for testing purposes, you can use the provided Jupyter Notebook:

[따릉이 Real Data Collection Notebook](https://github.com/cyl0424/2023_BA_Project/blob/bc8f0e309b0da554236a749242fce6560b64c925/%EB%94%B0%EB%A6%89%EC%9D%B4%20Real%20Data.ipynb)

Follow the instructions in the notebook to collect real-time rental data. Note that this step is optional, and you can proceed with the analysis without real-time data collection.
