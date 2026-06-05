# Nomura Quant Challenge - System Design & Implementation Report

## Architecture Overview
The system is designed as a robust, production-grade fixed income pricing engine. The primary challenge was calculating exact analytical risk sensitivities (PV01) directly from market quotes without relying on finite-difference bumping.

We solved this using an architecture centered around the **Implicit Function Theorem (IFT)** for propagating derivatives during curve bootstrapping.

The core abstractions are:
1. `IInterpolator`: A strategy pattern interface defining interpolation rules. Crucially, it provides both `interpolate` and `interpolateDerivatives`, allowing exact computation of $\frac{\partial \ln DF(t)}{\partial \ln DF(t_k)}$.
2. `DiscountCurve`: A container for nodes $(t, DF)$ which manages state and computes forward rates. It wraps the interpolator and computes the chain rule $\frac{\partial DF(t)}{\partial DF_k}$.
3. **Sequential Bootstrapping with Forward Jacobian**: As each node $k$ is calibrated, we accumulate the Jacobian $\frac{\partial DF_k}{\partial q_j}$. Since later nodes depend on earlier nodes, this forms a lower-triangular system that we solve sequentially via direct substitution during the bootstrapping process.

## Mathematical Formulation

### 1. Curve Bootstrapping Equations
- **Cash Rates ($T \le 180$)**: 
  $DF(T) = \frac{1}{1 + c \times \tau}$
  The derivative is directly $\frac{\partial DF_k}{\partial c_k} = -\frac{\tau}{(1+c\tau)^2}$.
  
- **Swap Rates ($T > 180$)**:
  The parity equation for a par swap is $f(DF_k) = 1 - DF_k - p_k \sum_{i} DF(t_i) \Delta_i = 0$.
  Because the intermediate payment dates $t_i$ may not align with market nodes, we interpolate $DF(t_i)$. This introduces non-linearity even for "Linear" interpolation (which is linear on $\ln(DF)$). 
  We use a **1D Newton-Raphson Solver** to iteratively find $DF_k$ that roots the parity equation. The analytical derivative $f'(DF_k)$ is cleanly obtained from our interpolator's `interpolateDerivatives` method.

### 2. Analytical Risk (Implicit Function Theorem)
Rather than bumping input quotes and re-running the solver, we compute the sensitivities analytically.
By differentiating the parity equation $1 - DF_k - p_k \sum_{i} DF(t_i) \Delta_i = 0$ with respect to market quote $q_j$, we isolate $\frac{\partial DF_k}{\partial q_j}$:

$$ \frac{\partial DF_k}{\partial q_j} \left[ 1 + p_k \sum_i \Delta_i \frac{\partial DF(t_i)}{\partial DF_k} \right] = - \delta_{kj} \sum_i DF(t_i) \Delta_i - p_k \sum_i \Delta_i \sum_{l=0}^{k-1} \frac{\partial DF(t_i)}{\partial DF_l} \frac{\partial DF_l}{\partial q_j} $$

This yields a closed-form recursive update for the Jacobian $\frac{\partial DF_k}{\partial q_j}$ directly from the previously computed derivatives $\frac{\partial DF_l}{\partial q_j}$.

### 3. Final PV Sensitivities (Chain Rule)
For the target 25Y Swap, we calculate its PV as $PV = PV_{fixed} - PV_{float}$.
The sensitivity to node $DF_k$ is:
$$ V_k = \frac{\partial PV}{\partial DF_k} = r_{fixed} N \sum_i \Delta_i \frac{\partial DF(t_i)}{\partial DF_k} + N \frac{\partial DF(T)}{\partial DF_k} $$

Finally, we chain this with the curve Jacobian to obtain the risk to the market quotes:
$$ Risk_j = \left( \sum_k V_k \frac{\partial DF_k}{\partial q_j} \right) \times 0.0001 $$

## Implementation Details & Robustness
1. **No External Dependencies**: Implemented in purely standard C++17.
2. **UTF-8 and BOM Handling**: The CSV parser automatically detects and strips UTF-8 BOM headers, ensuring cross-platform stability.
3. **Performance**: Analytical derivatives execute in $O(N^2)$ time compared to the $O(N^3)$ computational cost of bumping, rendering the risk engine computationally superior.
4. **Code Quality**: Strict typing, const-correctness, object-oriented segregation of duties, and rigorous Newton iteration safeguards.

## Results Summary
- **DF(784)**:
  - Cash/Linear: 0.8908
  - Cash/AQ:     0.8954
  - Swap/Linear: 0.9158
  - Swap/AQ:     0.9159

- **25Y Swap Pricing**:
  - Par Rate (Swap/AQ): 4.0286%
  - The PV for the fixed receiver is ~$30.93 per $100 Notional, reflective of receiving a 6.00% fixed rate against a ~4.02% par environment.
