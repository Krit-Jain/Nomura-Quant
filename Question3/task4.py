import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Union

# Import necessary data and models from task3
from task3 import models, df, val_mask, test_mask, features, taus

def optimal_threshold(tau: int, use_client_specific: bool = True) -> dict:
    """
    Returns optimal threshold configuration for a given horizon tau.
    """
    X_val = df.loc[val_mask, features]
    y_prob_val = models[tau].predict_proba(X_val)[:, 1]
    pnl_val = df.loc[val_mask, 'Side'] * df.loc[val_mask, 'Volume'] * (df.loc[val_mask, f'M{tau}'] - df.loc[val_mask, 'Trade Price'])
    clients_val = df.loc[val_mask, 'Name']
    
    X_test = df.loc[test_mask, features]
    y_prob_test = models[tau].predict_proba(X_test)[:, 1]
    pnl_test = df.loc[test_mask, 'Side'] * df.loc[test_mask, 'Volume'] * (df.loc[test_mask, f'M{tau}'] - df.loc[test_mask, 'Trade Price'])
    clients_test = df.loc[test_mask, 'Name']
    
    thetas = np.linspace(0, 1, 101)
    
    if use_client_specific:
        theta_dict = {}
        val_pnl_total = 0.0
        test_pnl_total = 0.0
        
        for c in clients_val.unique():
            mask_v = (clients_val == c)
            pnl_v = pnl_val[mask_v].values
            prob_v = y_prob_val[mask_v]
            
            best_th = 0.0
            best_pnl = -np.inf
            for th in thetas:
                v_pnl = np.sum(pnl_v[prob_v <= th])
                if v_pnl > best_pnl:
                    best_pnl = v_pnl
                    best_th = th
            
            theta_dict[c] = best_th
            val_pnl_total += best_pnl
            
            mask_t = (clients_test == c)
            t_pnl = np.sum(pnl_test[mask_t].values[y_prob_test[mask_t] <= best_th])
            test_pnl_total += t_pnl
            
        return {
            'theta': theta_dict,
            'validation_pnl': val_pnl_total,
            'test_pnl': test_pnl_total
        }
    else:
        best_th = 0.0
        best_pnl = -np.inf
        for th in thetas:
            v_pnl = np.sum(pnl_val.values[y_prob_val <= th])
            if v_pnl > best_pnl:
                best_pnl = v_pnl
                best_th = th
                
        t_pnl = np.sum(pnl_test.values[y_prob_test <= best_th])
        return {
            'theta': best_th,
            'validation_pnl': best_pnl,
            'test_pnl': t_pnl
        }

def plot_pnl_vs_theta(tau: int, client: str = None) -> None:
    """
    Plots PnL_validation(theta) for theta in [0, 1].
    Saves figure to 'pnl_vs_theta.png'.
    """
    X_val = df.loc[val_mask, features]
    y_prob_val = models[tau].predict_proba(X_val)[:, 1]
    pnl_val = df.loc[val_mask, 'Side'] * df.loc[val_mask, 'Volume'] * (df.loc[val_mask, f'M{tau}'] - df.loc[val_mask, 'Trade Price'])
    
    if client:
        mask = (df.loc[val_mask, 'Name'] == client)
        pnl_val_c = pnl_val[mask].values
        prob_val_c = y_prob_val[mask]
    else:
        pnl_val_c = pnl_val.values
        prob_val_c = y_prob_val
        
    thetas = np.linspace(0, 1, 101)
    pnls = []
    for theta in thetas:
        internalized = (prob_val_c <= theta)
        pnls.append(np.sum(pnl_val_c[internalized]))
        
    plt.figure()
    plt.plot(thetas, pnls, label='Validation PnL')
    plt.title(f'Validation PnL vs Theta (tau={tau})')
    plt.xlabel('Theta (Cutoff Probability)')
    plt.ylabel('Total Validation PnL')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'pnl_vs_theta_tau{tau}.png')
    plt.close()

if __name__ == '__main__':
    results = []
    
    # We choose client-specific because different clients have different adversity baselines.
    for tau in taus:
        print(f"Optimizing threshold for tau={tau}...")
        res = optimal_threshold(tau, use_client_specific=True)
        theta_dict = res['theta']
        
        # Calculate per-client test pnl
        X_test = df.loc[test_mask, features]
        y_prob_test = models[tau].predict_proba(X_test)[:, 1]
        pnl_test = df.loc[test_mask, 'Side'] * df.loc[test_mask, 'Volume'] * (df.loc[test_mask, f'M{tau}'] - df.loc[test_mask, 'Trade Price'])
        clients_test = df.loc[test_mask, 'Name']
        
        for client, th in theta_dict.items():
            mask_t = (clients_test == client)
            final_pnl = np.sum(pnl_test[mask_t].values[y_prob_test[mask_t] <= th])
            results.append({
                'client': client,
                'tau': tau,
                'theta*': th,
                'final_pnl': final_pnl
            })
            
        # Save a global plot for the PDF
        plot_pnl_vs_theta(tau, client=None)
        
    res_df = pd.DataFrame(results)
    # Sort for cleaner CSV
    res_df = res_df.sort_values(['client', 'tau'])
    res_df.to_csv('task4_results.csv', index=False)
    print("Optimization complete. task4_results.csv generated.")
