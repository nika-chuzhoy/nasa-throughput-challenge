import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
import joblib
import os

def prepare_data(df):
    """
    Prepare data for training by separating features and target
    """
    df = df.copy()
    df = df.dropna(subset=['actual_num_arrivals'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    feature_columns = [col for col in df.columns if col not in ['timestamp', 'actual_num_arrivals']]
    target_column = 'actual_num_arrivals'
    
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    return X, y, df['timestamp']

def create_time_based_folds(timestamps, n_splits=4):
    """
    Create time-based folds where each fold's test set contains 7 days out of every 28 days
    """
    # Convert timestamps to numpy array for faster operations
    dates = timestamps.values
    min_date = dates.min()
    max_date = dates.max()
    
    # Calculate period length
    period_length = np.timedelta64(28, 'D')
    test_length = np.timedelta64(7, 'D')
    
    # Initialize folds
    folds = []
    
    # For each fold, select different 7-day periods within each 28-day chunk
    for fold in range(n_splits):
        test_indices = []
        current_date = min_date
        
        # Calculate the start day for this fold's test periods
        test_start_offset = np.timedelta64(fold * 7, 'D')
        
        while current_date <= max_date:
            # Define test period for this chunk
            test_start = current_date + test_start_offset
            test_end = test_start + test_length
            
            # Get indices for this test period
            period_test_mask = (dates >= test_start) & (dates < test_end)
            period_test_indices = np.where(period_test_mask)[0]
            
            test_indices.extend(period_test_indices)
            
            # Move to next 28-day period
            current_date += period_length
        
        # Convert to numpy array
        test_indices = np.array(test_indices)
        
        # Create train indices as all indices not in test
        all_indices = np.arange(len(timestamps))
        train_indices = np.setdiff1d(all_indices, test_indices)
        
        folds.append((train_indices, test_indices))
    
    return folds

def train_catboost_model(X, y, timestamps, random_state=42):
    """
    Train a CatBoost model using 4-fold time-based cross-validation
    """
    # Create folds
    folds = create_time_based_folds(timestamps, n_splits=4)
    
    # Initialize lists to store results
    models = []
    all_metrics = []
    all_feature_importance = []
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"\nTraining Fold {fold_idx + 1}/4")
        
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Initialize and train model
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            l2_leaf_reg=3,
            max_depth=7,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=random_state,
            early_stopping_rounds=50,
            verbose=100
        )
        
        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            plot=False
        )
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'fold': fold_idx + 1,
            'train': {
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'mae': mean_absolute_error(y_train, y_pred_train),
                'r2': r2_score(y_train, y_pred_train)
            },
            'test': {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'mae': mean_absolute_error(y_test, y_pred_test),
                'r2': r2_score(y_test, y_pred_test)
            }
        }
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store results
        models.append(model)
        all_metrics.append(metrics)
        all_feature_importance.append(feature_importance)
        fold_results.append((X_test, y_test, y_pred_test))
    
    # Calculate average metrics across folds
    avg_metrics = {
        'train': {
            metric: np.mean([fold['train'][metric] for fold in all_metrics])
            for metric in ['rmse', 'mae', 'r2']
        },
        'test': {
            metric: np.mean([fold['test'][metric] for fold in all_metrics])
            for metric in ['rmse', 'mae', 'r2']
        }
    }
    
    # Average feature importance across folds
    avg_feature_importance = pd.concat(all_feature_importance).groupby('feature').mean().reset_index()
    avg_feature_importance = avg_feature_importance.sort_values('importance', ascending=False)
    
    return models, avg_metrics, avg_feature_importance, fold_results

def save_models(models, base_filename='catboost_model'):
    """
    Save all models from cross-validation
    """
    for i, model in enumerate(models):
        filename = f'{base_filename}_fold_{i+1}.joblib'
        joblib.dump(model, filename)
        print(f"Model {i+1} saved as {filename}")

def main(df):
    # Prepare the data
    X, y, timestamps = prepare_data(df)
    
    # Train the models and get results
    models, metrics, feature_importance, fold_results = train_catboost_model(X, y, timestamps)
    
    # Print average metrics
    print("\nAverage Model Performance Metrics:")
    print("\nTraining Set:")
    for metric, value in metrics['train'].items():
        print(f"{metric.upper()}: {value:.4f}")
    
    print("\nTest Set:")
    for metric, value in metrics['test'].items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Print feature importance
    print("\nAverage Feature Importance:")
    print(feature_importance)
    
    return models, metrics, feature_importance

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    airports = ['KATL', 'KCLT', 'KDEN', 'KDFW', 'KMEM', 'KMIA', 'KORD', 'KPHX', 'KSEA', 'KJFK']

    for airport in airports:
        features = pd.read_csv(f'features/train/{airport}_features.csv')
        models, metrics, feature_importance = main(features)
        save_models(models, f'models/{airport}')
