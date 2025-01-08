# The NASA Airport Throughput Prediction Challenge


## Overview
This repository is a submission for the 2024 NASA Throughput Challenge. Using provided data from ten US airports, the number of arrivals at airports in the US is forcasted over 15-minute intervals. Four CatBoost models are trained separately on each airport.

## Setup

Install the required python packages from the `requirements.txt`.
Next, download **NATPC - FUSER test** and all 10 **NATPC - FUSER train** files from https://bitgrit.net/competition/23. They should go in the raw_data/test and raw_data/train directories.

## Scripts
* **process_data.py** - extract tfm, tbfm, and runways FUSER data. Combine data from separate days into individual files.
* **prepare_features.py** - extract features.
* **train_models.py** - train 4 Catboost models on each airport.
* **infer.py** - create the submission.

To execute the entire pipeline, you can run **./run_all.sh**.

## Algorithm Summary
### Data & Features
Only the Runways, TFM, and TBFM datasets are used. The most important features are simply the number of arrivals predicted in each interval for TFM and TBFM. The features are shifted to capture the change in predictions over time in the input hours (first hour in each 4-hour interval).

### Models & Hyperparameters
Four Catboost models are trained for each airport. The final predictions for an airport are the average predictions of each of the four models.
Each model is trained with the same hyperparameters:
```
iterations=1000,
learning_rate=0.05,
l2_leaf_reg=3,
max_depth=7,
loss_function='RMSE',
eval_metric='RMSE',
early_stopping_rounds=50,
verbose=100
```
### Preventing Leakage
Leaked files (files with date ranges covering the test interval) are not ever used in the training of the model. During feature extraction, TFM & TBFM data is only kept if the timestamp is within one of the input hours and the prediction is for after the input hour. Finally, the training set is filtered to ensure the dates do not overlap with the testing set.

## Environment Details
Environment used to train the model.
* OS: macOS
* RAM: 8 GB
* CPU: 8 vCPU
* GPU: not used