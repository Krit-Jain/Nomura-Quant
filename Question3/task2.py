import pandas as pd
import numpy as np
from typing import List

df = pd.read_csv('trade_data.csv')

def expected_pnl(client: str, tau: List[int]) -> dict:
    """
    Parameters:
        client: Client identifier
        tau: List of horizons e.g. [5, 10, 15, 20, 25, 30]
    Returns:
        Dictionary with keys:
            'per_horizon': List[float]
            'aggregate': float
    """
    client_df = df[df['Name'] == client]
    if client_df.empty:
        return {'per_horizon': [0.0]*len(tau), 'aggregate': 0.0}
    
    per_horizon = []
    
    for t in tau:
        pnl = client_df['Side'] * client_df['Volume'] * (client_df[f'M{t}'] - client_df['Trade Price'])
        per_horizon.append(float(pnl.mean()))
        
    sum_M = sum(client_df[f'M{t}'] for t in tau)
    agg_pnl = client_df['Side'] * client_df['Volume'] * (sum_M / 6.0 - client_df['Trade Price'])
    aggregate = float(agg_pnl.mean())
    
    return {
        'per_horizon': per_horizon,
        'aggregate': aggregate
    }

def classify_client(client: str) -> str:
    """
    Parameters:
        client: Client identifier
    Returns:
        'profitable' or 'costly'
    """
    res = expected_pnl(client, [5, 10, 15, 20, 25, 30])
    return 'profitable' if res['aggregate'] >= 0 else 'costly'

def min_half_spread(client: str) -> float:
    """
    Parameters:
        client: Client identifier
    Returns:
        Minimum half-spread (in data units) such that
        expected aggregate PnL >= 0
    """
    client_df = df[df['Name'] == client]
    if client_df.empty:
        return 0.0
        
    taus = [5, 10, 15, 20, 25, 30]
    sum_M = sum(client_df[f'M{t}'] for t in taus)
    mean_M_tau = sum_M / 6.0
    
    intrinsic_pnl = client_df['Side'] * client_df['Volume'] * (mean_M_tau - client_df['M0'])
    expected_intrinsic = intrinsic_pnl.mean()
    expected_volume = client_df['Volume'].mean()
    
    delta_star = -expected_intrinsic / expected_volume
    return float(delta_star)

if __name__ == '__main__':
    clients = sorted(df['Name'].unique())
    taus = [5, 10, 15, 20, 25, 30]
    
    results = []
    for client in clients:
        pnl_res = expected_pnl(client, taus)
        delta_star = min_half_spread(client)
        
        row = {'client': client}
        for i, t in enumerate(taus):
            row[f'tau = {t}'] = pnl_res['per_horizon'][i]
        row['agg_pnl'] = pnl_res['aggregate']
        row['delta*'] = delta_star
        results.append(row)
        
    res_df = pd.DataFrame(results)
    res_df.to_csv('task2_results.csv', index=False)
    print(res_df.to_string())
    
    print("\nClient Classifications:")
    for client in clients:
        print(f"Client {client}: {classify_client(client)}")
