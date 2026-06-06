import pandas as pd
import numpy as np
from typing import List

# Load the data
df = pd.read_csv('trade_data.csv')

def adversity_profile(client: str, tau: List[int]) -> List[float]:
    """
    Parameters:
        client: Client identifier (single character or string)
        tau: List of horizons e.g. [5, 10, 15, 20, 25, 30]
    Returns:
        List of floats representing adversity percentage at each horizon
    """
    client_df = df[df['Name'] == client]
    if client_df.empty:
        return [0.0] * len(tau)
    
    results = []
    for t in tau:
        # Trade is adverse if PnL < 0
        # PnL = Side * Volume * (M_tau - Trade Price)
        pnl = client_df['Side'] * client_df['Volume'] * (client_df[f'M{t}'] - client_df['Trade Price'])
        adverse_count = (pnl < 0).sum()
        total_count = len(client_df)
        adversity_pct = (adverse_count / total_count) * 100
        results.append(adversity_pct)
        
    return results

if __name__ == '__main__':
    clients = sorted(df['Name'].unique())
    taus = [5, 10, 15, 20, 25, 30]
    
    results = []
    for client in clients:
        adv = adversity_profile(client, taus)
        results.append({'client': client, **{f'tau = {t}': a for t, a in zip(taus, adv)}})
        
    res_df = pd.DataFrame(results)
    # The output format requests: client, tau = 5, tau = 10, tau = 15, tau = 20, tau = 25, tau = 30
    res_df.to_csv('task1_results.csv', index=False)
    print(res_df.to_string())
