import numpy as np
import pandas as pd
from typing import Tuple

def quote(inventory: float, sigma: float, alpha: float, eta: float) -> Tuple[float, float]:
    """
    Returns (delta_bid, delta_ask) dynamically adjusted based on inventory, volatility, 
    adversity (alpha), and time of day (eta).
    """
    # Baseline spread depends on volatility and adversity
    # The more adverse the client (alpha -> 1), the wider the spread
    # c_min * sigma is the absolute floor
    base_spread = sigma * (0.5 + 2.0 * alpha)
    
    # Inventory skew: shifts the reservation price to attract flow that flattens inventory
    # Skew becomes more aggressive as eta -> 1 (end of day) to avoid the quadratic penalty
    skew_factor = 0.1 * (1.0 + 5.0 * eta)
    skew = inventory * sigma * skew_factor
    
    delta_bid = base_spread + skew
    delta_ask = base_spread - skew
    
    # Constraints
    c_min = 0.5
    delta_min = c_min * sigma
    delta_max = 0.5  # approx 50 bps of mid (assuming mid ~100)
    
    delta_bid = max(delta_min, min(delta_max, delta_bid))
    delta_ask = max(delta_min, min(delta_max, delta_ask))
    
    return float(delta_bid), float(delta_ask)

def validate_quote(df: pd.DataFrame) -> None:
    """
    Runs a simple backtest simulation over the dataset.
    Since true (lambda, gamma, phi) are hidden, we use estimated bounds 
    to demonstrate the backtesting methodology.
    """
    print("Starting validation backtest...")
    
    # Assumed hidden parameters for simulation
    lam = 0.5
    gamma = 1.0
    phi = 0.1
    
    days = df['Date'].unique()
    
    total_net_pnl = 0.0
    daily_pnls = []
    
    np.random.seed(42)
    
    for date in days:
        day_df = df[df['Date'] == date].copy()
        
        # Calculate eta
        start_time = pd.to_datetime(date + ' 09:30:00')
        end_time = pd.to_datetime(date + ' 16:00:00')
        total_sec = (end_time - start_time).total_seconds()
        
        day_df['datetime'] = pd.to_datetime(day_df['Date'] + ' ' + day_df['time'])
        elapsed = (day_df['datetime'] - start_time).dt.total_seconds()
        day_df['eta'] = (elapsed / total_sec).clip(0, 1)
        
        # Calculate sigma
        N = 20
        r = day_df['M0'].pct_change().fillna(0)
        sigma_series = np.sqrt((r**2).rolling(N, min_periods=1).mean()).fillna(0.0001)
        sigma_series = np.maximum(sigma_series, 0.0001)
        
        alphas = np.random.uniform(0, 1, len(day_df))
        
        inventory = 0.0
        day_pnl = 0.0
        
        for i in range(len(day_df)):
            row = day_df.iloc[i]
            sigma = sigma_series.iloc[i]
            eta = row['eta']
            alpha = alphas[i]
            vol = row['Volume']
            side = row['Side'] 
            
            db, da = quote(inventory, sigma, alpha, eta)
            
            delta_side = db if side == 1 else da
            
            p_fill = lam * np.exp(-gamma * (delta_side / sigma))
            
            if np.random.rand() < p_fill:
                avg_m = np.mean([row[f'M{t}'] for t in [5, 10, 15, 20, 25, 30]])
                tp = row['M0'] - side * delta_side
                
                trade_pnl = side * vol * (avg_m - tp)
                day_pnl += trade_pnl
                inventory += side * vol
                
        sigma_D = sigma_series.mean()
        penalty = phi * (inventory**2) * sigma_D
        
        net_day_pnl = day_pnl - penalty
        daily_pnls.append(net_day_pnl)
        total_net_pnl += net_day_pnl
        
    print(f"Validation complete over {len(days)} days.")
    print(f"Total Net PnL: {total_net_pnl:.2f}")
    
    std_pnl = np.std(daily_pnls)
    if std_pnl > 0:
        sharpe = np.mean(daily_pnls) / std_pnl
        print(f"Daily Sharpe Ratio: {sharpe:.4f}")
    
if __name__ == '__main__':
    df = pd.read_csv('trade_data.csv')
    validate_quote(df)
