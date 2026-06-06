import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import textwrap

fig, ax = plt.subplots(figsize=(8.5, 11))

text = """
Task 3: Adversity Prediction Model

1. Choice of Model and Rationale
We selected the `HistGradientBoostingClassifier` (a scikit-learn implementation equivalent to LightGBM) as our prediction model M. 
Rationale:
- Robust Tabular Performance: Gradient boosted trees consistently achieve state-of-the-art performance on tabular financial data, effectively capturing complex, non-linear interactions between features (e.g., between Volume and Spread).
- Categorical Support: The model natively handles categorical variables (like Client ID) without requiring explosive one-hot encoding, preserving the tree split efficiency.
- Speed and Scalability: Histogram-based binning makes training highly efficient over the large trade dataset.
- Probabilistic Output: The model outputs well-calibrated probabilities via softmax/logistic loss, directly fulfilling the requirement to predict P(Adverse).

2. Feature Selection and Justification
The feature vector ordering is: [eta, sigma, Volume, Spread, Side, imbalance, client_idx]

Justification for each feature:
- eta (Time of Day Fraction): The probability of adverse flow often follows intraday patterns (e.g., market open and close see higher institutional informed volume).
- sigma (Realized Volatility): Computed over the past 20 trades. High volatility periods tend to cluster with informed trading and adverse price drift.
- Volume: Larger trade sizes frequently indicate informed institutional participants seeking to absorb available liquidity before a price move.
- Spread: Current bid-ask spread reflects market uncertainty and the LP's existing adversity premium.
- Side: Whether the LP is buying or selling. Certain sides may exhibit asymmetric adversity depending on market momentum.
- imbalance (Rolling Signed Volume): The sum of signed volume over the past 20 trades. Strong directional flow imbalance is a classic predictor of short-term price drift (microstructure momentum).
- client_idx (Categorical Client ID): As proven in Tasks 1 and 2, clients exhibit drastically different baseline toxicities. Identifying the client is paramount for accurate adversity prediction.

These features comprehensively capture the microstructural state (Spread, Volume, Side), the market regime (sigma, eta, imbalance), and the specific participant (client), allowing deep mathematical learning of the adversity likelihood.
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

with PdfPages('task3_report.pdf') as pdf:
    pdf.savefig(fig)
