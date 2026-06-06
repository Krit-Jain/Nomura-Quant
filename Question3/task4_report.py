import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import textwrap

res_df = pd.read_csv('task4_results.csv')
final_pnl_total = res_df['final_pnl'].sum()

fig, ax = plt.subplots(figsize=(8.5, 11))

text = f"""
Task 4: Optimal Externalization Threshold

1. Over-Externalization and Under-Externalization Trade-offs
- Over-Externalization (theta is too low): 
If the threshold is too low, the LP is overly conservative and externalizes too many trades. While this minimizes adverse selection risk, it drastically reduces the volume internalized. Since the LP earns the bid-ask spread on internalized trades, over-externalization results in significant lost revenue. The LP effectively gives away safe, profitable trades to third parties.

- Under-Externalization (theta is too high):
If the threshold is too high, the LP internalizes too much flow, including highly toxic trades. While revenue from the spread increases, it is overwhelmed by the severe adverse price drift that follows these informed trades. The LP absorbs toxic flow and suffers heavy inventory mark-to-market losses.
  
The optimal theta* perfectly balances spread revenue from safe trades against the adverse selection costs of toxic trades.

2. Global vs. Client-Specific Thresholds
A client-specific threshold is mathematically and practically superior to a global threshold. As demonstrated in Tasks 1 and 2, clients exhibit vastly different baseline toxicities. 
Using a global threshold would under-externalize toxic clients (like Client F) and over-externalize benign clients (like Client A). By computing theta* per client, the model adapts to the specific flow profile of each participant, maximizing internalization of retail flow while aggressively filtering institutional/informed flow. The evidence lies in the superior aggregate PnL achieved on the validation set when sweeping theta per client compared to a single global sweep.

3. Final Test Set PnL
The final aggregate test set PnL across all clients and horizons using the client-specific optimal thresholds is: {final_pnl_total:.2f}.
"""

wrapped_lines = []
for line in text.split('\n'):
    if line.strip() == '':
        wrapped_lines.append('')
    else:
        wrapped_lines.append(textwrap.fill(line, width=90))

wrapped_text = '\n'.join(wrapped_lines)

ax.axis('off')
ax.text(0.05, 0.5, wrapped_text, fontsize=11, verticalalignment='center', fontfamily='sans-serif')

with PdfPages('task4_report.pdf') as pdf:
    pdf.savefig(fig)
    
    # Add the plots generated from task4
    for tau in [5, 10, 15, 20, 25, 30]:
        img_fig, img_ax = plt.subplots(figsize=(8.5, 6))
        try:
            img = plt.imread(f'pnl_vs_theta_tau{tau}.png')
            img_ax.imshow(img)
            img_ax.axis('off')
            pdf.savefig(img_fig)
        except Exception as e:
            print(f"Could not load image for tau={tau}: {e}")
        plt.close(img_fig)
