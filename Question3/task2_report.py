import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import textwrap

fig, ax = plt.subplots(figsize=(8.5, 11))

text = """
Task 2: Client Profitability and Spread Recommendation

1. Client Classification and Quantitative Evidence
Based on the aggregate expected PnL (eq. 6) across the six time horizons:
- Client A: Profitable (Agg PnL = 1.89)
- Client B: Profitable (Agg PnL = 1.63)
- Client C: Profitable (Agg PnL = 1.17)
- Client D: Profitable (Agg PnL = 0.30)
- Client E: Costly (Agg PnL = -0.40)
- Client F: Costly (Agg PnL = -1.44)

Clients A, B, C, and D generate positive expected PnL on average, meaning the LP's bid-ask spread captures enough margin to overcome any adverse price drift from their trades. Clients E and F, however, are highly toxic; the market moves against the LP so severely after trading with them that the standard spread is insufficient, resulting in a net loss.

2. Minimum Half-Spread (delta*) and Adversity Profile
The recommended minimum half-spread delta* required to break even on aggregate flow is:
- Client A: -0.0117
- Client B: -0.0079
- Client C: -0.0020
- Client D: 0.0096
- Client E: 0.0189
- Client F: 0.0328

(Note: A negative delta* theoretically implies the LP could quote inside the mid-price and still break even, reflecting highly benign flow. In practice, the LP would quote positive spreads to maximize profit).

Relationship to Adversity Profile from Task 1:
The required delta* directly correlates with the adversity profiles established in Task 1. 
- Client F had the highest adversity percentage (~62% at tau=30), leading to severe adverse selection costs. Thus, Client F requires the widest minimum half-spread (0.0328) to compensate for these expected losses.
- Conversely, Client A had the lowest adversity profile (~41%), meaning their flow does not systematically predict adverse market movements. Consequently, they require the smallest (even negative) delta* to break even.

This demonstrates that adversity profiles are a reliable proxy for client toxicity and expected profitability, directly dictating the risk premium (wider spreads) the market maker must charge.
"""

# Manual wrapping for each paragraph to preserve newlines
wrapped_lines = []
for line in text.split('\n'):
    if line.strip() == '':
        wrapped_lines.append('')
    else:
        wrapped_lines.append(textwrap.fill(line, width=90))

wrapped_text = '\n'.join(wrapped_lines)

ax.axis('off')
ax.text(0.05, 0.5, wrapped_text, fontsize=11, verticalalignment='center', fontfamily='sans-serif')

with PdfPages('task2_report.pdf') as pdf:
    pdf.savefig(fig)
