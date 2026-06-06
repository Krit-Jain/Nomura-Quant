import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
from typing import List, Dict

# Load data globally for the required functions
df = pd.read_csv('trade_data.csv')

# Preprocess and feature engineering
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['time'])
df = df.sort_values('datetime').reset_index(drop=True)

# Feature 1: eta (fraction of trading day)
start_time = pd.to_datetime(df['Date'] + ' 09:30:00')
end_time = pd.to_datetime(df['Date'] + ' 16:00:00')
total_seconds = (end_time - start_time).dt.total_seconds()
elapsed_seconds = (df['datetime'] - start_time).dt.total_seconds()
df['eta'] = (elapsed_seconds / total_seconds).clip(0, 1)

# Feature 2: realized volatility (sigma)
N = 20
df['r'] = df['M0'].pct_change()
df['r_sq'] = df['r']**2
df['sigma'] = np.sqrt(df['r_sq'].rolling(window=N, min_periods=1).mean())
df['sigma'] = df['sigma'].fillna(0)

# Feature 3: Order Imbalance over last N trades
df['signed_volume'] = df['Side'] * df['Volume']
df['imbalance'] = df['signed_volume'].rolling(window=N, min_periods=1).sum()

# Feature 4: Client categorical encoding
encoder = OrdinalEncoder()
df['client_idx'] = encoder.fit_transform(df[['Name']])

# Define features
features = ['eta', 'sigma', 'Volume', 'Spread', 'Side', 'imbalance', 'client_idx']
categorical_features = [6]

# Split 60/20/20 chronologically by date
dates = sorted(df['Date'].unique())
train_cutoff = int(len(dates) * 0.6)
val_cutoff = int(len(dates) * 0.8)

train_dates = dates[:train_cutoff]
val_dates = dates[train_cutoff:val_cutoff]
test_dates = dates[val_cutoff:]

train_mask = df['Date'].isin(train_dates)
val_mask = df['Date'].isin(val_dates)
test_mask = df['Date'].isin(test_dates)

X_train = df.loc[train_mask, features]

# Dictionary to hold models for each tau
models = {}
taus = [5, 10, 15, 20, 25, 30]

print("Training models...")
for tau in taus:
    y_train = (df.loc[train_mask, 'Side'] * (df.loc[train_mask, f'M{tau}'] - df.loc[train_mask, 'Trade Price']) < 0).astype(int)
    model = HistGradientBoostingClassifier(categorical_features=categorical_features, random_state=42)
    model.fit(X_train, y_train)
    models[tau] = model
print("Training complete.")

def predict_adversity(*args, **kwargs) -> float:
    """
    Returns probability that trade is adverse at given horizon.
    Expected kwargs: 'row' (pd.Series or dict of the trade), 'tau' (int)
    """
    row = kwargs.get('row')
    tau = kwargs.get('tau')
    
    if row is None or tau is None:
        raise ValueError("Must provide 'row' and 'tau'")
        
    client_idx = encoder.transform([[row['Name']]])[0][0]
    
    X = pd.DataFrame([{
        'eta': row.get('eta', 0.5), 
        'sigma': row.get('sigma', 0.0), 
        'Volume': row['Volume'], 
        'Spread': row['Spread'], 
        'Side': row['Side'], 
        'imbalance': row.get('imbalance', 0.0), 
        'client_idx': client_idx
    }])
    
    model = models[tau]
    prob = model.predict_proba(X)[0, 1]
    return float(prob)

def compute_metrics(*args, **kwargs) -> pd.DataFrame:
    metrics_list = []
    
    for split_name, mask in [('train', train_mask), ('validation', val_mask), ('test', test_mask)]:
        X_split = df.loc[mask, features]
        split_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'log_loss': []}
        
        for tau in taus:
            y_true = (df.loc[mask, 'Side'] * (df.loc[mask, f'M{tau}'] - df.loc[mask, 'Trade Price']) < 0).astype(int)
            y_pred_prob = models[tau].predict_proba(X_split)[:, 1]
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            split_metrics['accuracy'].append(accuracy_score(y_true, y_pred))
            split_metrics['precision'].append(precision_score(y_true, y_pred, zero_division=0))
            split_metrics['recall'].append(recall_score(y_true, y_pred, zero_division=0))
            split_metrics['log_loss'].append(log_loss(y_true, y_pred_prob))
            
        metrics_list.append({
            'split': split_name,
            'accuracy': np.mean(split_metrics['accuracy']),
            'precision': np.mean(split_metrics['precision']),
            'recall': np.mean(split_metrics['recall']),
            'log_loss': np.mean(split_metrics['log_loss'])
        })
        
    return pd.DataFrame(metrics_list).set_index('split')

if __name__ == '__main__':
    res_df = compute_metrics()
    res_df.to_csv('task3_results.csv')
    print(res_df)
