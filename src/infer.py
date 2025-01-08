from datetime import datetime, timedelta
import pandas as pd
import joblib
import numpy as np


def extract_airport(id_string):
    return id_string.split('_')[0]


def convert_to_datetime(id_string):
    parts = id_string.split('_')
    date_str, hour_str, minute_str = parts[1:4]
    hr = datetime.strptime(f"{date_str}_{hour_str}", '%y%m%d_%H%M')
    minutes = int(minute_str) - 15
    dt = hr + timedelta(minutes=minutes)
    return dt


def match_and_replace_values(submission_df, data_dfs):
    # Create a copy to avoid modifying the original DataFrame
    result_df = submission_df.copy()
    
    result_df['airport'] = result_df['ID'].apply(extract_airport)
    result_df['timestamp'] = result_df['ID'].apply(convert_to_datetime)
    
    # Process each airport separately
    for airport in result_df['airport'].unique():
        # Get the data DataFrame for this airport
        if airport not in data_dfs:
            print(f"Warning: No data found for airport {airport}")
            continue
            
        airport_data = data_dfs[airport].copy()
        airport_data['timestamp'] = pd.to_datetime(airport_data['timestamp'])
        
        # Create a mapping from timestamp to num_arrivals for this airport
        timestamp_to_arrivals = dict(zip(airport_data['timestamp'], 
                                       airport_data['prediction']))
        
        # Get indices for this airport in the submission DataFrame
        airport_mask = result_df['airport'] == airport
        
        # Update values where timestamps match
        for idx in result_df[airport_mask].index:
            submission_timestamp = result_df.loc[idx, 'timestamp']
            if submission_timestamp in timestamp_to_arrivals:
                result_df.loc[idx, 'Value'] = timestamp_to_arrivals[submission_timestamp]
            else:
                print(f"Warning: No exact timestamp match found for {airport} at {submission_timestamp}")
    return result_df


def load_models(base_filename='catboost_model', n_folds=4):
    """
    Load all models from cross-validation
    """
    models = []
    for i in range(n_folds):
        filename = f'{base_filename}_fold_{i+1}.joblib'
        models.append(joblib.load(filename))
    return models


def predict(models, data):
    """
    Make predictions using ensemble of models from cross-validation
    """
    feature_columns = [col for col in data.columns if col not in ['timestamp', 'actual_num_arrivals']]
    
    if isinstance(data, pd.DataFrame):
        X = data[feature_columns]
    else:
        raise ValueError("Input must be a pandas DataFrame")
    
    # Make predictions with each model
    predictions = np.array([model.predict(X) for model in models])
    
    # Return average prediction
    return np.mean(predictions, axis=0)


def infer():
    airport_dfs = {}
    # airports = ['KATL', 'KCLT', 'KDEN', 'KDFW', 'KJFK', 'KMEM', 'KMIA', 'KORD', 'KPHX', 'KSEA']
    airports = ['KATL', 'KCLT', 'KDFW', 'KMEM']
    submission_format = pd.read_csv('raw_data/submission_format.csv')

    for airport in airports:
        models = load_models(f'models/{airport}')
        features = pd.read_csv(f'features/test/{airport}_features.csv')
        preds = predict(models, features)
        preds_df = features.copy()
        preds_df['prediction'] = preds
        preds_df.loc[(preds_df['total_floor_arrivals'].isna()) & (preds_df['total_floor_departures'].isna()), 'prediction'] = 0
        airport_dfs[airport] = preds_df[['timestamp', 'prediction']]

    to_submit = match_and_replace_values(submission_format, airport_dfs)
    to_submit['Value'] = to_submit['Value'].replace(99, 0)
    to_submit['Value'] = to_submit['Value'].clip(lower=0)
    to_submit = to_submit[['ID', 'Value']]
    to_submit.to_csv(f'submission.csv', index=False)


if __name__ == "__main__":
    infer()