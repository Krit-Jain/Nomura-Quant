# Nomura Quant Challenge - Implementation & System Design Report

## 1. Comparison of Implementations (`solution.cpp` vs `solution_v2.cpp`)
Both versions mathematically succeed in producing the exact same accurate output (e.g., PV = 30.93, Par Rate = 4.028%). However, **`solution_v2.cpp` is significantly better and represents a true production-grade quantitative finance architecture.**

Here is the comparison:
* **Data Structures (Computational Efficiency)**: In v1, analytical derivatives were tracked using `std::map<int, double>`. Since `std::map` is a node-based Red-Black Tree, it forces dynamic memory allocation inside the innermost solver loop, which is a massive performance bottleneck. In v2, this was replaced with a custom `SparseVector` backed by contiguous memory, providing $O(1)$ cache locality. 
* **Matrix Operations**: v1 used `std::vector<std::vector<double>>` for the Jacobian, which fragments memory. v2 uses a `DenseMatrix` (a 1D flat array mapping) which is the industry standard for mathematical libraries (like BLAS) to ensure vectorization and cache pre-fetching.
* **Separation of Concerns**: v1 mixed mathematical logic (Newton solver) directly into the financial logic. v2 isolates the generic `newtonRaphson` solver into a separate `math` namespace, making the code vastly more maintainable. 

---

## 2. Mathematical Derivations & Implementation Approach
The core requirement of this challenge was to compute risk sensitivities analytically (without numerical bumping). 

### Curve Bootstrapping (Implicit Function Theorem)
For any market instrument $k$ with maturity $T_k$, the pricing parity equation must equal zero:
$$ F(\text{quotes}, \text{DiscountFactors}) = PV_{float} - PV_{fixed} = 0 $$

For Cash ($T \le 180$), the relationship is linear: $DF(T) = \frac{1}{1 + c \tau}$.
For Swaps ($T > 180$), the intermediate cashflows require interpolation, creating a non-linear dependency on $DF(T_k)$. We use a 1D **Newton-Raphson Solver** to iteratively find $\ln DF(T_k)$.

To obtain the exact risk matrix (Jacobian) $\frac{\partial DF_k}{\partial q_j}$, we differentiate the parity equation $F=0$ with respect to the market quote $q_j$ using the **Implicit Function Theorem (IFT)**:
$$ \frac{\partial F}{\partial DF_k} \frac{\partial DF_k}{\partial q_j} + \sum_{l=0}^{k-1} \frac{\partial F}{\partial DF_l} \frac{\partial DF_l}{\partial q_j} + \frac{\partial F}{\partial q_j} = 0 $$
By keeping track of $\frac{\partial DF_l}{\partial q_j}$ for all previously bootstrapped nodes $l < k$, we can sequentially solve for the current node's derivatives $\frac{\partial DF_k}{\partial q_j}$ via forward substitution.

### Pricing and Risk Engine
Once the full Jacobian is constructed, the PV of the new 25Y swap is computed. The exact 1bp Risk Sensitivity is then extracted via the Chain Rule:
$$ Risk_j = \left( \sum_{k} \frac{\partial PV}{\partial DF_k} \frac{\partial DF_k}{\partial q_j} \right) \times 0.0001 $$

---

## 3. Answer to Q3: Designing a Generic System Architecture
To satisfy Q3 (designing a robust, extensible system for any new instrument and interpolation method), the `solution_v2.cpp` architecture was designed specifically around **SOLID Object-Oriented Design** principles. 

If this system were deployed to a production pricing desk, the generic architecture would be structured as follows:

### A. The Instrument Abstraction (`IInstrument`)
To accept any new instrument with its own cashflows, we define an interface that mandates a present value function and an analytical derivative function:
```cpp
class IInstrument {
public:
    virtual ~IInstrument() = default;
    virtual double calculatePV(const IYieldCurve& curve) const = 0;
    virtual math::SparseVector calculatePVDerivatives(const IYieldCurve& curve) const = 0;
    virtual double getMaturityTime() const = 0;
};
```
* **Why this works**: A developer can add a `FRA`, `OIS`, or `TenorBasisSwap` by simply inheriting from `IInstrument`. The generic `CurveBuilder` will never need to know the specific cashflow structure; it just calls `calculatePV()`.

### B. The Interpolation Abstraction (`IInterpolator`)
The system accepts any new interpolation method (e.g., Cubic Spline) by adhering to the `IInterpolator` interface:
```cpp
class IInterpolator {
public:
    virtual double interpolate(double t, const std::vector<double>& times, const std::vector<double>& lnDFs) const = 0;
    virtual math::SparseVector interpolateDerivatives(double t, const std::vector<double>& times) const = 0;
};
```
* **Why this works**: The `DiscountCurve` holds a `std::unique_ptr<IInterpolator>`. At runtime, we can inject a `CubicSplineInterpolator`. Because the interface forces the implementation of `interpolateDerivatives`, the Newton solver and IFT Jacobian generator will continue to function flawlessly without any modifications.

### C. The Generic Calibrator (`CurveBuilder` & `RootFinder`)
The calibration engine is decoupled from the instruments. It iterates through the sorted array of `IInstrument` pointers. For each instrument, it defines a generic lambda function:
```cpp
auto parityFunc = [&](double lnDF_k) {
    curve.updateNode(k, std::exp(lnDF_k));
    double f = instrument->calculatePV(curve);
    double df_dx = instrument->calculatePVDerivatives(curve).get(k);
    return std::pair{f, df_dx};
};
double root = math::newtonRaphson(guess, parityFunc);
```
* **Scalability**: This generic structure means adding 50 new instruments requires **zero** changes to the solver or bootstrapping logic. The architecture is fully modular, highly cohesive, and loosely coupled.

## 4. Conclusion & Output Validation
The implementation strictly adheres to the prompt's mandate of analytical derivation (no bump-and-revalue). The system successfully loaded the CSV quotes, bootstrapped the cash and swap curves, and priced the out-of-sample 25Y Swap.
* **Validation**: The PV of the 25Y swap (receiver of 6.00% fixed) in a 4.028% market environment evaluates to an exact positive valuation of ~30.93 per 100 Notional. The risk sensitivities cleanly offset the market par curve, validating the structural integrity of the Jacobian.
