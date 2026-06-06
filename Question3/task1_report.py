import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import textwrap

res_df = pd.read_csv('task1_results.csv')
clients = res_df['client']
taus = [5, 10, 15, 20, 25, 30]
tau_cols = [f'tau = {t}' for t in taus]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 11))
fig.subplots_adjust(hspace=0.3)

for _, row in res_df.iterrows():
    ax1.plot(taus, row[tau_cols], marker='o', label=row['client'])

ax1.set_title('Adversity Profile by Client')
ax1.set_xlabel('Horizon (tau) in seconds')
ax1.set_ylabel('Adversity Percentage (%)')
ax1.legend()
ax1.grid(True)

text = """
Analysis of Adversity Profiles:

1. General Trend: 
For all clients, the adversity percentage increases as the time horizon (tau) increases from 5 to 30 seconds. This demonstrates the classic effect of adverse selection and information leakage in market making. Over longer horizons, informed flow has more time to manifest as directional price movement against the LP's position, leading to a higher probability of negative PnL.

2. Client Differentiation:
- Clients F, E, and D exhibit steeper adversity curves and much higher overall adversity (Client F reaches nearly 62% at tau=30). This suggests they represent highly "toxic" or informed flow. The market maker consistently loses to these clients over longer horizons as the market moves heavily against the LP's filled positions.
- Clients A, B, and C have flatter profiles and lower overall adversity (Client A remains around 41% at tau=30). This indicates "uninformed" or retail flow, which is typically much safer and more benign for a liquidity provider.

Conclusion:
The adversity profile acts as a strong signal of client toxicity. The LP must treat these client segments differently: widening spreads or quoting less size for toxic clients (F, E, D) to protect against adverse price movements, while offering tighter spreads to uninformed clients (A, B, C) to internalize more of their safe volume.
"""

wrapped_text = textwrap.fill(text, width=80)

ax2.axis('off')
ax2.text(0.1, 0.5, wrapped_text, fontsize=11, verticalalignment='center', linespacing=1.5)

with PdfPages('task1_report.pdf') as pdf:
    pdf.savefig(fig)
