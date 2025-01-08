import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def convert_to_datetime(id_string):
    parts = id_string.split('_')
    date_str, hour_str, minute_str = parts[1:4]
    hr = datetime.strptime(f"{date_str}_{hour_str}", '%y%m%d_%H%M')
    minutes = int(minute_str) - 15
    dt = hr + timedelta(minutes=minutes)
    return dt

def filter_tfm_tbfm(airport, output_file):
    print('1/4 READING INPUTS')
    print('Reading tfm')
    tfm_train_df = pd.read_parquet(f'processed_data/train/{airport}_tfm.parquet')
    tfm_test_df = pd.read_parquet(f'processed_data/test/{airport}_tfm.parquet')
    tfm_df = pd.concat([tfm_train_df, tfm_test_df])
    print('Concating tfm')
    del tfm_train_df, tfm_test_df

    print('Reading tbfm')
    tbfm_train_df = pd.read_parquet(f'processed_data/train/{airport}_tbfm.parquet')
    tbfm_test_df = pd.read_parquet(f'processed_data/test/{airport}_tbfm.parquet')
    print('Concating tbfm')
    tbfm_df = pd.concat([tbfm_train_df, tbfm_test_df])
    del tbfm_train_df, tbfm_test_df

    valid_hours = [0, 4, 8, 12, 16, 20]

    tfm_df['timestamp'] = pd.to_datetime(tfm_df['timestamp'])
    tfm_df = tfm_df[tfm_df['timestamp'].dt.hour.isin(valid_hours)]
    tfm_df = tfm_df.rename(columns={'timestamp': 'tfm_timestamp'})

    tbfm_df['timestamp'] = pd.to_datetime(tbfm_df['timestamp'])
    tbfm_df = tbfm_df[tbfm_df['timestamp'].dt.hour.isin(valid_hours)]
    tbfm_df = tbfm_df.rename(columns={'timestamp': 'tbfm_timestamp'})

    print('2/4 SORTING VALUES')
    latest_tfm = tfm_df.sort_values('tfm_timestamp').groupby('gufi').last().reset_index()
    latest_tbfm = tbfm_df.sort_values('tbfm_timestamp').groupby('gufi').last().reset_index()

    all_gufis = pd.concat([
            latest_tfm[['gufi']],
            latest_tbfm[['gufi']]
        ]).drop_duplicates()

    print('3/4 MERGING')
    result = (all_gufis
                 .merge(latest_tfm[['gufi', 'arrival_runway_estimated_time', 'tfm_timestamp']],
                       on=['gufi'],
                       how='left')
                 .merge(latest_tbfm[['gufi', 'arrival_runway_sta', 'tbfm_timestamp']],
                       on=['gufi'],
                       how='left'))

    result = result.rename(columns={
            'arrival_runway_estimated_time': 'tfm_arrival',
            'arrival_runway_sta': 'tbfm_arrival',
        })

    result['combined_tbfm_tfm'] = result['tbfm_arrival'].fillna(result['tfm_arrival'])

    print('4/4 SAVING')
    result.to_csv(output_file, index=False)

def resample_and_count(series):
        return pd.Series(1, index=pd.to_datetime(series)).resample('15T').count()

def get_arrival_departure_counts(runways_path, airport, mode):
    df = pd.read_parquet(runways_path)

    arrival_df = df.dropna(subset=['departure_runway_actual_time'])
    arrival_df = arrival_df.drop_duplicates(subset=['gufi', 'arrival_runway_actual_time'], keep='first')

    departure_df = df.dropna(subset=['arrival_runway_actual_time'])
    departure_df = departure_df.drop_duplicates(subset=['gufi', 'departure_runway_actual_time'], keep='first')

    arrival_counts = resample_and_count(arrival_df['arrival_runway_actual_time'].dropna())
    departure_counts = resample_and_count(departure_df['departure_runway_actual_time'].dropna())

    arrival_df = pd.DataFrame({
        'num_arrivals': arrival_counts,
    })
    arrival_df = arrival_df.reset_index()
    arrival_df = arrival_df.rename(columns={'arrival_runway_actual_time': 'timestamp'})

    departure_df = pd.DataFrame({
        'num_departures': departure_counts,
    })
    departure_df = departure_df.reset_index()
    departure_df = departure_df.rename(columns={'departure_runway_actual_time': 'timestamp'})

    arrival_df.to_csv(f'processed_data/{mode}/{airport}_combined_arrivals.csv', index=False)
    departure_df.to_csv(f'processed_data/{mode}/{airport}_combined_departures.csv', index=False)

def get_tfm_tbfm_counts(df, train_output_file, test_output_file):
    tfm_counts = resample_and_count(df['tfm_arrival'].dropna())
    tbfm_counts = resample_and_count(df['tbfm_arrival'].dropna())
    combined_counts = resample_and_count(df['combined_tbfm_tfm'].dropna())
    
    # Create final DataFrame with all counts
    result = pd.DataFrame({
        'tfm_arrivals': tfm_counts,
        'tbfm_arrivals': tbfm_counts,
        'combined_arrivals': combined_counts
    }).fillna(0)
    
    # Reset index to make timestamp a column
    result = result.reset_index()
    new_df = result.rename(columns={'index': 'timestamp'})

    new_df['hour'] = new_df['timestamp'].dt.hour
    new_df['minute_interval'] = (new_df['timestamp'].dt.minute // 15)

    # Mask hours
    hour = new_df['hour']
    mask = ~hour.isin([0, 4, 8, 12, 16, 20])
    new_df = new_df[mask]

    # Get all dates from the submission format
    submission_format = pd.read_csv('raw_data/submission_format.csv')
    submission_format['timestamp'] = submission_format['ID'].apply(convert_to_datetime)
    test_timestamps = submission_format['timestamp'].drop_duplicates()

    # Create train and test splits
    train_df = new_df[~new_df['timestamp'].isin(test_timestamps)]
    test_df = new_df[new_df['timestamp'].isin(test_timestamps)]

    # Print statistics for train set
    print("\nTrain Set Statistics:")
    print(f"TFM Arrivals - Mean: {train_df['tfm_arrivals'].mean():.2f}, Std: {train_df['tfm_arrivals'].std():.2f}")
    print(f"TBFM Arrivals - Mean: {train_df['tbfm_arrivals'].mean():.2f}, Std: {train_df['tbfm_arrivals'].std():.2f}")
    print(f"Combined Arrivals - Mean: {train_df['combined_arrivals'].mean():.2f}, Std: {train_df['combined_arrivals'].std():.2f}")
    
    # Print statistics for test set
    print("\nTest Set Statistics:")
    print(f"TFM Arrivals - Mean: {test_df['tfm_arrivals'].mean():.2f}, Std: {test_df['tfm_arrivals'].std():.2f}")
    print(f"TBFM Arrivals - Mean: {test_df['tbfm_arrivals'].mean():.2f}, Std: {test_df['tbfm_arrivals'].std():.2f}")
    print(f"Combined Arrivals - Mean: {test_df['combined_arrivals'].mean():.2f}, Std: {test_df['combined_arrivals'].std():.2f}")

    # Save the splits
    train_df.to_csv(train_output_file, index=False)
    test_df.to_csv(test_output_file, index=False)

def add_day_of_week(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    return df

def get_hour_floor(hour):
    target_hours = [1, 5, 9, 13, 17, 21]
    valid_values = [v for v in target_hours if v <= hour]
    return max(valid_values) if valid_values else max(target_hours)

def add_shifts(features):
    features['floor_hour'] = features['hour'].apply(get_hour_floor)
    features['interval'] = 4 * (features['hour'] - features['floor_hour']) + features['minute_interval']

    features['next_tfm_arrivals'] = features[features['interval'] != 11]['tfm_arrivals'].shift(-1)
    features['prev_tfm_arrivals'] = features[features['interval'] != 0]['tfm_arrivals'].shift(1)

    features['next_tbfm_arrivals'] = features[features['interval'] != 11]['tbfm_arrivals'].shift(-1)
    features['prev_tbfm_arrivals'] = features[features['interval'] != 0]['tbfm_arrivals'].shift(1)

    features['next_combined_arrivals'] = features[features['interval'] != 11]['combined_arrivals'].shift(-1)
    features['prev_combined_arrivals'] = features[features['interval'] != 0]['combined_arrivals'].shift(1)

    return features.drop(['hour', 'minute_interval'], axis=1)

def add_arrivals_departures(features, mode):
    temp = features.copy()
    temp['date'] = pd.to_datetime(temp['timestamp']).dt.date
    temp['floor_timestamp4'] = pd.to_datetime(temp['date']) + pd.to_timedelta(temp['floor_hour'], unit='h') - pd.to_timedelta(15, unit='m')
    temp['floor_timestamp3'] = temp['floor_timestamp4'] - pd.to_timedelta(15, unit='m')
    temp['floor_timestamp2'] = temp['floor_timestamp3'] - pd.to_timedelta(15, unit='m')
    temp['floor_timestamp1'] = temp['floor_timestamp2'] - pd.to_timedelta(15, unit='m')

    # add arrivals data
    arrivals = pd.read_csv(f'processed_data/{mode}/{airport}_combined_arrivals.csv')
    arrivals = arrivals.rename(columns={'timestamp': 'floor_timestamp', 'num_arrivals': 'floor_num_arrivals'})
    arrivals['floor_timestamp'] = pd.to_datetime(arrivals['floor_timestamp'])
    for i in range(1,5):
        temp_merge = temp.merge(
            arrivals[['floor_timestamp', 'floor_num_arrivals']], 
            how='left', 
            left_on=[f'floor_timestamp{i}'], 
            right_on=['floor_timestamp']
        )
        temp[f'floor_num_arrivals_{i}'] = temp_merge['floor_num_arrivals']
    columns = ['floor_num_arrivals_1', 'floor_num_arrivals_2', 
          'floor_num_arrivals_3', 'floor_num_arrivals_4']
    temp['total_floor_arrivals'] = temp[columns].fillna(0).sum(axis=1)
    # If all source columns were NaN, make the sum NaN
    all_nan_mask = temp[columns].isna().all(axis=1)
    temp.loc[all_nan_mask, 'total_floor_arrivals'] = np.nan

    # add departure data
    departures = pd.read_csv(f'processed_data/{mode}/{airport}_combined_departures.csv')
    departures = departures.rename(columns={'timestamp': 'floor_timestamp', 'num_departures': 'floor_num_departures'})
    departures['floor_timestamp'] = pd.to_datetime(departures['floor_timestamp'])
    for i in range(1,5):
        temp_merge = temp.merge(
            departures[['floor_timestamp', 'floor_num_departures']], 
            how='left', 
            left_on=[f'floor_timestamp{i}'], 
            right_on=['floor_timestamp']
        )
        temp[f'floor_num_departures_{i}'] = temp_merge['floor_num_departures']
    columns = ['floor_num_departures_1', 'floor_num_departures_2', 
          'floor_num_departures_3', 'floor_num_departures_4']
    temp['total_floor_departures'] = temp[columns].fillna(0).sum(axis=1)
    # If all source columns were NaN, make the sum NaN
    all_nan_mask = temp[columns].isna().all(axis=1)
    temp.loc[all_nan_mask, 'total_floor_departures'] = np.nan

    to_keep = list(features.columns) + ['floor_num_departures_1', 'floor_num_departures_2', 
          'floor_num_departures_3', 'floor_num_departures_4'] + ['floor_num_arrivals_1', 'floor_num_arrivals_2', 
          'floor_num_arrivals_3', 'floor_num_arrivals_4'] + ['total_floor_arrivals', 'total_floor_departures']
    
    return temp[to_keep]

def add_labels(features, airport):
    arrivals = pd.read_csv(f'processed_data/train/{airport}_combined_arrivals.csv')
    arrivals = arrivals.rename(columns={'num_arrivals': 'actual_num_arrivals'})
    features['timestamp'] = pd.to_datetime(features['timestamp'])
    arrivals['timestamp'] = pd.to_datetime(arrivals['timestamp'])
    features = features.merge(arrivals, on='timestamp', how='left')
    return features

if __name__ == "__main__":
    os.makedirs('features/train', exist_ok=True)
    os.makedirs('features/test', exist_ok=True)
    airports = ['KATL', 'KCLT', 'KDEN', 'KDFW', 'KJFK', 'KMEM', 'KMIA', 'KORD', 'KPHX', 'KSEA']

    # Add most important features - tfm and tbfm preds
    for airport in airports:
        print('Processing airport: ', airport)
        filter_tfm_tbfm(airport, f'processed_data/filtered_tfm_tbfm_{airport}.csv')
        get_tfm_tbfm_counts(pd.read_csv(f'processed_data/filtered_tfm_tbfm_{airport}.csv'), f'features/train/{airport}_features.csv', f'features/test/{airport}_features.csv')
    
    # Add secondary features
    for airport in airports:
        for mode in ('test', 'train'):
            get_arrival_departure_counts(f'processed_data/{mode}/{airport}_runways.parquet', airport, mode)
            features = pd.read_csv(f'features/{mode}/{airport}_features.csv')
            features = add_day_of_week(features)
            features = add_shifts(features)
            features = add_arrivals_departures(features, mode)
            if mode == 'train':
                features = add_labels(features, airport)
            features.to_csv(f'features/{mode}/{airport}_features.csv',index=False)