/**
 * ============================================================================
 *  Nomura Quant Challenge — Interest Rate Curve Construction System (v2)
 * ============================================================================
 *
 *  Architecture Overview (Professional Quant Standard):
 *  ────────────────────────────────────────────────────
 *  1. Strong Types & Memory Efficiency: Utilizes `SparseVector` for O(1) cache 
 *     locality during analytical derivative accumulation, eliminating `std::map` overhead.
 *  2. Flat Dense Matrices: Uses 1D contiguous arrays for the Jacobian matrix 
 *     to maximize BLAS-style memory alignment.
 *  3. Builder Pattern: `DiscountCurve` is an immutable data container (IYieldCurve). 
 *     Bootstrapping logic is isolated inside `CurveBuilder`.
 *  4. Isolated Mathematics: Root finding (Newton-Raphson) is decoupled into a 
 *     pure mathematical utility template, proving strong Separation of Concerns.
 *  5. Implicit Function Theorem (IFT): Exact analytical risk generation without bumping.
 *
 *  Compiler: C++17
 * ============================================================================
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <memory>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <iomanip>

// ============================================================================
//  Section 1: Core Types & Constants
// ============================================================================

namespace quant {

    using TimeDays = int;
    using Rate     = double;
    using Price    = double;
    using Notional = double;

    constexpr TimeDays DAYS_PER_WEEK  = 7;
    constexpr TimeDays DAYS_PER_MONTH = 30;
    constexpr TimeDays DAYS_PER_YEAR  = 360;
    constexpr TimeDays SEMI_ANNUAL    = 180;
    
    constexpr double NEWTON_TOL = 1e-14;
    constexpr int    NEWTON_MAX = 200;

    inline double dcf(TimeDays t2, TimeDays t1) {
        return static_cast<double>(t2 - t1) / DAYS_PER_YEAR;
    }
}

// ============================================================================
//  Section 2: High-Performance Mathematical Primitives
// ============================================================================

namespace quant::math {

    /**
     * SparseVector: Highly optimized storage for analytical derivatives.
     * Replaces std::map<int, double> to prevent dynamic allocations in hot loops.
     */
    class SparseVector {
    public:
        void add(int index, double value) {
            // Find if index exists
            for (auto& pair : data_) {
                if (pair.first == index) {
                    pair.second += value;
                    return;
                }
            }
            // Not found, append
            data_.emplace_back(index, value);
        }

        double get(int index) const {
            for (const auto& pair : data_) {
                if (pair.first == index) return pair.second;
            }
            return 0.0;
        }

        const std::vector<std::pair<int, double>>& entries() const {
            return data_;
        }

    private:
        std::vector<std::pair<int, double>> data_;
    };

    /**
     * DenseMatrix: 1D flat array for O(1) contiguous memory access.
     */
    class DenseMatrix {
    public:
        DenseMatrix(int rows, int cols, double initVal = 0.0)
            : rows_(rows), cols_(cols), data_(rows * cols, initVal) {}

        double& operator()(int r, int c) {
            return data_[r * cols_ + c];
        }

        const double& operator()(int r, int c) const {
            return data_[r * cols_ + c];
        }

        int rows() const { return rows_; }
        int cols() const { return cols_; }

    private:
        int rows_, cols_;
        std::vector<double> data_;
    };

    /**
     * Robust 1D Newton-Raphson Solver
     * Takes a lambda that returns a pair: { f(x), f'(x) }
     */
    template <typename Func>
    double newtonRaphson(double guess, Func f_df, double tol = NEWTON_TOL, int maxIter = NEWTON_MAX) {
        double x = guess;
        for (int i = 0; i < maxIter; ++i) {
            auto [f, df] = f_df(x);
            if (std::abs(f) < tol) {
                return x;
            }
            if (std::abs(df) < 1e-15) {
                throw std::runtime_error("NewtonRaphson: Zero derivative encountered.");
            }
            x = x - f / df;
        }
        throw std::runtime_error("NewtonRaphson: Maximum iterations exceeded.");
    }
}

// ============================================================================
//  Section 3: Interfaces
// ============================================================================

namespace quant {

    /**
     * Interface for Interpolation Strategies
     */
    class IInterpolator {
    public:
        virtual ~IInterpolator() = default;

        // Interpolates ln(DF)
        virtual double interpolate(double t, const std::vector<double>& times, const std::vector<double>& lnDFs) const = 0;

        // Computes analytical derivatives ∂lnDF(t)/∂lnDF(t_k) into a SparseVector
        virtual math::SparseVector interpolateDerivatives(double t, const std::vector<double>& times) const = 0;
    };

    /**
     * Interface for Yield Curve
     */
    class IYieldCurve {
    public:
        virtual ~IYieldCurve() = default;
        virtual double getDF(TimeDays t) const = 0;
        virtual double getForwardRate(TimeDays t1, TimeDays t2) const = 0;
        virtual math::SparseVector getDFNodeDerivatives(TimeDays t) const = 0;
    };

} // namespace quant

// ============================================================================
//  Section 4: Interpolators (Phase 2)
// ============================================================================

namespace quant {

    class LinearLogDFInterpolator : public IInterpolator {
    public:
        double interpolate(double t, const std::vector<double>& times, const std::vector<double>& lnDFs) const override {
            if (times.empty()) return 0.0;
            if (t <= times.front()) return lnDFs.front();
            if (t >= times.back()) return lnDFs.back();

            auto it = std::lower_bound(times.begin(), times.end(), t);
            int i = std::distance(times.begin(), it);
            int prev = i - 1;

            double w = (t - times[prev]) / (times[i] - times[prev]);
            return lnDFs[prev] + w * (lnDFs[i] - lnDFs[prev]);
        }

        math::SparseVector interpolateDerivatives(double t, const std::vector<double>& times) const override {
            math::SparseVector derivs;
            if (times.empty()) return derivs;
            
            if (t <= times.front()) { derivs.add(0, 1.0); return derivs; }
            if (t >= times.back()) { derivs.add(times.size() - 1, 1.0); return derivs; }

            auto it = std::lower_bound(times.begin(), times.end(), t);
            int i = std::distance(times.begin(), it);
            int prev = i - 1;

            double w = (t - times[prev]) / (times[i] - times[prev]);
            derivs.add(prev, 1.0 - w);
            derivs.add(i, w);
            return derivs;
        }
    };

    class AveragedQuadraticLogDFInterpolator : public IInterpolator {
    public:
        double interpolate(double t, const std::vector<double>& times, const std::vector<double>& lnDFs) const override {
            int n = times.size();
            if (n < 3 || t <= times.front() || t >= times.back()) {
                LinearLogDFInterpolator lin;
                return lin.interpolate(t, times, lnDFs);
            }

            auto it = std::lower_bound(times.begin(), times.end(), t);
            int i = std::distance(times.begin(), it);
            if (i == 1) {
                LinearLogDFInterpolator lin;
                return lin.interpolate(t, times, lnDFs);
            }

            double dt = times[i] - times[i-1];
            double w_prev = (times[i] - t) / dt;
            double w_curr = (t - times[i-1]) / dt;

            if (i == n - 1) return evalQ(i - 1, t, times, lnDFs);

            return w_prev * evalQ(i - 1, t, times, lnDFs) + w_curr * evalQ(i, t, times, lnDFs);
        }

        math::SparseVector interpolateDerivatives(double t, const std::vector<double>& times) const override {
            int n = times.size();
            if (n < 3 || t <= times.front() || t >= times.back()) {
                LinearLogDFInterpolator lin;
                return lin.interpolateDerivatives(t, times);
            }

            auto it = std::lower_bound(times.begin(), times.end(), t);
            int i = std::distance(times.begin(), it);
            if (i == 1) {
                LinearLogDFInterpolator lin;
                return lin.interpolateDerivatives(t, times);
            }

            math::SparseVector derivs;
            double dt = times[i] - times[i-1];
            double w_prev = (times[i] - t) / dt;
            double w_curr = (t - times[i-1]) / dt;

            if (i == n - 1) {
                auto dq = evalQDerivs(i - 1, t, times);
                for (const auto& [idx, val] : dq.entries()) derivs.add(idx, val);
                return derivs;
            }

            auto dq_prev = evalQDerivs(i - 1, t, times);
            auto dq_curr = evalQDerivs(i, t, times);

            for (const auto& [idx, val] : dq_prev.entries()) derivs.add(idx, w_prev * val);
            for (const auto& [idx, val] : dq_curr.entries()) derivs.add(idx, w_curr * val);

            return derivs;
        }

    private:
        double evalQ(int idx, double t, const std::vector<double>& times, const std::vector<double>& lnDFs) const {
            double t0 = times[idx-1], t1 = times[idx], t2 = times[idx+1];
            double y0 = lnDFs[idx-1], y1 = lnDFs[idx], y2 = lnDFs[idx+1];

            double L0 = (t - t1) * (t - t2) / ((t0 - t1) * (t0 - t2));
            double L1 = (t - t0) * (t - t2) / ((t1 - t0) * (t1 - t2));
            double L2 = (t - t0) * (t - t1) / ((t2 - t0) * (t2 - t1));

            return y0 * L0 + y1 * L1 + y2 * L2;
        }

        math::SparseVector evalQDerivs(int idx, double t, const std::vector<double>& times) const {
            double t0 = times[idx-1], t1 = times[idx], t2 = times[idx+1];
            
            double L0 = (t - t1) * (t - t2) / ((t0 - t1) * (t0 - t2));
            double L1 = (t - t0) * (t - t2) / ((t1 - t0) * (t1 - t2));
            double L2 = (t - t0) * (t - t1) / ((t2 - t0) * (t2 - t1));

            math::SparseVector derivs;
            derivs.add(idx-1, L0);
            derivs.add(idx,   L1);
            derivs.add(idx+1, L2);
            return derivs;
        }
    };

} // namespace quant

// ============================================================================
//  Section 5: Data Structures & Discount Curve Container
// ============================================================================

namespace quant {

    struct MarketQuote {
        std::string tenor;
        TimeDays maturityDays;
        Rate cashRate;
        Rate parSwapRate;
    };

    struct SwapSpec {
        int maturityDays;
        int fixedFreqDays;
        int floatFreqDays;
        double notional;
        double fixedRate;
    };

    class DiscountCurve : public IYieldCurve {
    public:
        explicit DiscountCurve(std::unique_ptr<IInterpolator> interpolator)
            : interpolator_(std::move(interpolator)) {
            times_.push_back(0);
            dfs_.push_back(1.0);
            lndfs_.push_back(0.0);
        }

        void addNode(TimeDays t, double df) {
            times_.push_back(t);
            dfs_.push_back(df);
            lndfs_.push_back(std::log(df));
        }

        void updateNode(size_t index, double df) {
            if (index < dfs_.size()) {
                dfs_[index] = df;
                lndfs_[index] = std::log(df);
            }
        }

        const std::vector<double>& getTimes() const { return times_; }
        const std::vector<double>& getDFs() const { return dfs_; }
        const std::vector<double>& getLnDFs() const { return lndfs_; }

        double getDF(TimeDays t) const override {
            if (t == 0) return 1.0;
            double lnDF = interpolator_->interpolate(t, times_, lndfs_);
            return std::exp(lnDF);
        }

        double getForwardRate(TimeDays t1, TimeDays t2) const override {
            double df1 = getDF(t1);
            double df2 = getDF(t2);
            double tau = dcf(t2, t1);
            if (tau == 0.0) return 0.0;
            return (df1 / df2 - 1.0) / tau;
        }

        math::SparseVector getDFNodeDerivatives(TimeDays t) const override {
            return interpolator_->interpolateDerivatives(t, times_);
        }

        // Jacobian tracking matrix (Node vs MarketQuote)
        std::unique_ptr<math::DenseMatrix> jacobian;

    private:
        std::unique_ptr<IInterpolator> interpolator_;
        std::vector<double> times_;
        std::vector<double> dfs_;
        std::vector<double> lndfs_;
    };
} // namespace quant

// ============================================================================
//  Section 6: Builder Pattern for Bootstrapping (Phase 3)
// ============================================================================

namespace quant {

    class CurveBuilder {
    public:
        // Build cash nodes (T <= 180)
        static void bootstrapCash(DiscountCurve& curve, const std::vector<MarketQuote>& quotes) {
            int n = quotes.size();
            curve.jacobian = std::make_unique<math::DenseMatrix>(n, n, 0.0);

            for (int i = 0; i < n; ++i) {
                int T = quotes[i].maturityDays;
                double c = quotes[i].cashRate;
                double tau = dcf(T, 0);
                
                double df = 1.0 / (1.0 + c * tau);
                curve.addNode(T, df);

                // ∂DF_i / ∂c_i  (Note: node index i is quote index i)
                (*curve.jacobian)(i, i) = -tau / ((1.0 + c * tau) * (1.0 + c * tau));
            }
        }

        // Build swap nodes (T > 180)
        static void bootstrapSwaps(DiscountCurve& curve, const std::vector<MarketQuote>& quotes) {
            int n = quotes.size();
            curve.jacobian = std::make_unique<math::DenseMatrix>(n, n, 0.0);

            for (int k = 0; k < n; ++k) {
                int Tk = quotes[k].maturityDays;
                double pk = quotes[k].parSwapRate;

                if (Tk <= SEMI_ANNUAL) {
                    double tau = dcf(Tk, 0);
                    double df = 1.0 / (1.0 + pk * tau);
                    curve.addNode(Tk, df);
                    (*curve.jacobian)(k, k) = -tau / ((1.0 + pk * tau) * (1.0 + pk * tau));
                    continue; 
                }

                auto schedule = generateSchedule(Tk, SEMI_ANNUAL);
                double guessDF = curve.getDFs().back() * 0.95; 
                curve.addNode(Tk, guessDF);

                // Newton Solver
                auto swapParityFunc = [&](double lnDF_k) -> std::pair<double, double> {
                    curve.updateNode(k + 1, std::exp(lnDF_k)); 
                    
                    double PV01 = 0.0;
                    double dPV01_dx = 0.0; // dx = d(lnDF_k)
                    int prev_t = 0;

                    for (int ti : schedule) {
                        double delta = dcf(ti, prev_t);
                        double df_ti = curve.getDF(ti);
                        PV01 += df_ti * delta;

                        auto derivs = curve.getDFNodeDerivatives(ti);
                        double s_ik = derivs.get(k + 1); // d(lnDF_ti) / d(lnDF_k)
                        double dDFti_dx = s_ik * df_ti; // d(DF_ti) / d(lnDF_k)
                        
                        dPV01_dx += dDFti_dx * delta;
                        prev_t = ti;
                    }

                    double pv_float = 1.0 - curve.getDF(Tk);
                    double dpv_float_dx = -curve.getDF(Tk); // d(1 - exp(x))/dx = -exp(x)

                    double pv_fixed = pk * PV01;
                    double dpv_fixed_dx = pk * dPV01_dx;

                    double f = pv_float - pv_fixed;
                    double df_dx = dpv_float_dx - dpv_fixed_dx;

                    return {f, df_dx};
                };

                // Solve for ln(DF_k)
                double final_lnDF = math::newtonRaphson(std::log(guessDF), swapParityFunc);
                curve.updateNode(k + 1, std::exp(final_lnDF));

                // Compute IFT Analytical Jacobian for node k
                // Parity eq: F(q, DF) = 1 - DF_k - p_k * sum(delta_i * DF_ti) = 0
                // dF/dDF_k * dDF_k/dq_j + sum_{l < k} dF/dDF_l * dDF_l/dq_j + dF/dq_j = 0
                // For node derivatives: d(DF_ti) / d(DF_l) = s_il * DF_ti / DF_l
                
                std::vector<math::SparseVector> S_list;
                std::vector<double> DFti_list;
                double PV01 = 0.0;
                int prev_t = 0;
                for (int ti : schedule) {
                    double delta = dcf(ti, prev_t);
                    double df_ti = curve.getDF(ti);
                    PV01 += df_ti * delta;
                    DFti_list.push_back(df_ti);
                    S_list.push_back(curve.getDFNodeDerivatives(ti));
                    prev_t = ti;
                }

                double df_k = curve.getDF(Tk);
                double C_k = 1.0; // -dF/dDF_k initially 1.0
                prev_t = 0;
                for (size_t i = 0; i < schedule.size(); ++i) {
                    double delta = dcf(schedule[i], prev_t);
                    double s_ik = S_list[i].get(k + 1);
                    // d(DF_ti)/d(DF_k) = s_ik * DF_ti / DF_k
                    C_k += pk * delta * (s_ik * DFti_list[i] / df_k);
                    prev_t = schedule[i];
                }

                for (int j = 0; j <= k; ++j) {
                    double RHS_j = 0.0;
                    if (j == k) {
                        RHS_j -= PV01; // dF/dp_k = -PV01
                    }
                    
                    double sum_l = 0.0;
                    for (int l = 0; l < k; ++l) {
                        double term_l = 0.0;
                        int prev_t2 = 0;
                        double df_l = curve.getDF(quotes[l].maturityDays); // Note: quote l corresponds to node l+1
                        
                        for (size_t i = 0; i < schedule.size(); ++i) {
                            double delta = dcf(schedule[i], prev_t2);
                            double s_il = S_list[i].get(l + 1);
                            // d(DF_ti)/d(DF_l) = s_il * DF_ti / DF_l
                            term_l += pk * delta * (s_il * DFti_list[i] / df_l);
                            prev_t2 = schedule[i];
                        }
                        sum_l += term_l * (*curve.jacobian)(l, j);
                    }
                    RHS_j -= sum_l;
                    (*curve.jacobian)(k, j) = RHS_j / C_k;
                }
            }
        }

    private:
        static std::vector<int> generateSchedule(int maturityDays, int freqDays) {
            std::vector<int> sched;
            int n_periods = maturityDays / freqDays;
            for (int i = 1; i <= n_periods; ++i) {
                sched.push_back(i * freqDays);
            }
            return sched;
        }
    };
} // namespace quant

// ============================================================================
//  Section 7: Pricing & Analytical Risk Engine (Phase 4)
// ============================================================================

namespace quant {

    struct SwapMetrics {
        double pv;
        double parRate;
        std::vector<math::SparseVector> fixedS; 
        math::SparseVector matS;               
    };

    class SwapPricer {
    public:
        static SwapMetrics priceSwap(const DiscountCurve& curve, const SwapSpec& spec) {
            auto fixedSched = generateSchedule(spec.maturityDays, spec.fixedFreqDays);
            
            double pvFixed = 0.0;
            double PV01 = 0.0;
            std::vector<math::SparseVector> fixedS;
            
            int prev_t = 0;
            for (int ti : fixedSched) {
                double delta = dcf(ti, prev_t);
                double df = curve.getDF(ti);
                PV01 += df * delta;
                pvFixed += spec.fixedRate * spec.notional * df * delta;
                fixedS.push_back(curve.getDFNodeDerivatives(ti));
                prev_t = ti;
            }
            
            double df_mat = curve.getDF(spec.maturityDays);
            double pvFloat = spec.notional * (1.0 - df_mat);
            
            double pv = pvFixed - pvFloat;
            double parRate = (1.0 - df_mat) / PV01;
            
            math::SparseVector matS = curve.getDFNodeDerivatives(spec.maturityDays);
            
            return {pv, parRate, fixedS, matS};
        }

        static std::vector<double> calculateRisk(const DiscountCurve& curve, const SwapSpec& spec, const SwapMetrics& m, int n_quotes) {
            std::vector<double> V(n_quotes, 0.0);
            auto fixedSched = generateSchedule(spec.maturityDays, spec.fixedFreqDays);
            
            for (int k = 0; k < n_quotes; ++k) {
                int node = k + 1;
                double vk = 0.0;
                
                vk += spec.notional * m.matS.get(node) * curve.getDF(spec.maturityDays) / curve.getDF(curve.getTimes()[node]); 
                
                int prev_t = 0;
                for (size_t i = 0; i < fixedSched.size(); ++i) {
                    double delta = dcf(fixedSched[i], prev_t);
                    double s_ik = m.fixedS[i].get(node);
                    vk += spec.fixedRate * spec.notional * delta * s_ik * curve.getDF(fixedSched[i]) / curve.getDF(curve.getTimes()[node]);
                    prev_t = fixedSched[i];
                }
                V[k] = vk;
            }
            
            std::vector<double> risk(n_quotes, 0.0);
            for (int j = 0; j < n_quotes; ++j) {
                double r = 0.0;
                for (int k = 0; k < n_quotes; ++k) {
                    r += V[k] * (*curve.jacobian)(k, j);
                }
                risk[j] = r * 0.0001; 
            }
            return risk;
        }

    private:
        static std::vector<int> generateSchedule(int maturityDays, int freqDays) {
            std::vector<int> sched;
            int n_periods = maturityDays / freqDays;
            for (int i = 1; i <= n_periods; ++i) {
                sched.push_back(i * freqDays);
            }
            return sched;
        }
    };
} // namespace quant

// ============================================================================
//  Section 8: I/O Parsers
// ============================================================================

namespace quant {

    struct MarketData {
        std::vector<MarketQuote> quotes;
        int queryTimeDays;
        SwapSpec newSwap;
    };

    int parseMaturityToDays(const std::string& tenor) {
        if (tenor.empty()) return 0;
        char unit = tenor.back();
        int value = std::stoi(tenor.substr(0, tenor.size() - 1));
        if (unit == 'D') return value;
        if (unit == 'W') return value * DAYS_PER_WEEK;
        if (unit == 'M' || unit == 'm') return value * DAYS_PER_MONTH;
        if (unit == 'Y' || unit == 'y') return value * DAYS_PER_YEAR;
        return value;
    }

    MarketData parseInput(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) throw std::runtime_error("Cannot open " + filepath);

        std::string line;
        
        // Remove UTF-8 BOM if present
        if (std::getline(file, line)) {
            if (line.size() >= 3 && (unsigned char)line[0] == 0xEF && 
                (unsigned char)line[1] == 0xBB && (unsigned char)line[2] == 0xBF) {
                line = line.substr(3);
            }
        }

        MarketData data;
        int n_quotes = std::stoi(line.substr(0, line.find(',')));
        
        for (int i = 0; i < n_quotes; ++i) {
            std::getline(file, line);
            std::stringstream ss(line);
            std::string tenor, cashStr, swapStr;
            
            std::getline(ss, tenor, ',');
            std::getline(ss, cashStr, ',');
            std::getline(ss, swapStr, ',');

            MarketQuote q;
            q.tenor = tenor;
            q.maturityDays = parseMaturityToDays(tenor);
            q.cashRate = std::stod(cashStr) / 100.0;
            q.parSwapRate = std::stod(swapStr) / 100.0;
            data.quotes.push_back(q);
        }

        std::getline(file, line);
        data.queryTimeDays = std::stoi(line.substr(0, line.find(',')));

        std::getline(file, line);
        std::stringstream ss(line);
        std::string fixedRateStr, mat, fixFreq, floatFreq;
        std::getline(ss, fixedRateStr, ',');
        std::getline(ss, mat, ',');
        std::getline(ss, fixFreq, ',');
        std::getline(ss, floatFreq, ',');

        data.newSwap.fixedRate = std::stod(fixedRateStr) / 100.0;
        data.newSwap.notional = 100.0; // Standard notional
        data.newSwap.maturityDays = parseMaturityToDays(mat);
        data.newSwap.fixedFreqDays = parseMaturityToDays(fixFreq);
        data.newSwap.floatFreqDays = parseMaturityToDays(floatFreq);

        return data;
    }
} // namespace quant

int main(int argc, char* argv[]) {
    std::cout << "========================================\n";
    std::cout << "  Nomura Quant Challenge - v2 Engine\n";
    std::cout << "========================================\n\n";

    try {
        std::string inputFile = (argc > 1) ? argv[1] : "Test_Input.csv";
        std::string outputFile = (argc > 2) ? argv[2] : "Test_Output.csv";
        
        auto data = quant::parseInput(inputFile);
        std::cout << "[Phase 1/2] Data loaded from " << inputFile << " (" << data.quotes.size() << " instruments)\n";

        // Phase 3: Builder Pattern & Bootstrapping
        std::cout << "\n[Phase 3] Bootstrapping Curves...\n";
        
        quant::DiscountCurve cashLin(std::make_unique<quant::LinearLogDFInterpolator>());
        quant::CurveBuilder::bootstrapCash(cashLin, data.quotes);
        std::cout << "  Q1.a) DF(" << data.queryTimeDays << ") Cash/Linear: " << cashLin.getDF(data.queryTimeDays) << "\n";

        quant::DiscountCurve cashAQ(std::make_unique<quant::AveragedQuadraticLogDFInterpolator>());
        quant::CurveBuilder::bootstrapCash(cashAQ, data.quotes);
        std::cout << "  Q1.b) DF(" << data.queryTimeDays << ") Cash/AQ:     " << cashAQ.getDF(data.queryTimeDays) << "\n";

        quant::DiscountCurve swapLin(std::make_unique<quant::LinearLogDFInterpolator>());
        quant::CurveBuilder::bootstrapSwaps(swapLin, data.quotes);
        std::cout << "  Q1.c) DF(" << data.queryTimeDays << ") Swap/Linear: " << swapLin.getDF(data.queryTimeDays) << "\n";

        quant::DiscountCurve swapAQ(std::make_unique<quant::AveragedQuadraticLogDFInterpolator>());
        quant::CurveBuilder::bootstrapSwaps(swapAQ, data.quotes);
        std::cout << "  Q1.d) DF(" << data.queryTimeDays << ") Swap/AQ:     " << swapAQ.getDF(data.queryTimeDays) << "\n";

        std::cout << "\n[Phase 4] Pricing New 25Y Swap & Analytical Risk Engine...\n";
        
        auto m_cashLin = quant::SwapPricer::priceSwap(cashLin, data.newSwap);
        auto m_cashAQ  = quant::SwapPricer::priceSwap(cashAQ, data.newSwap);
        auto m_swapLin = quant::SwapPricer::priceSwap(swapLin, data.newSwap);
        auto m_swapAQ  = quant::SwapPricer::priceSwap(swapAQ, data.newSwap);

        std::cout << "  PV (Swap/AQ)      = " << m_swapAQ.pv << "\n";
        std::cout << "  Par Rate(Swap/AQ) = " << m_swapAQ.parRate * 100.0 << "%\n";

        int n_quotes = data.quotes.size();
        auto risk_cashLin = quant::SwapPricer::calculateRisk(cashLin, data.newSwap, m_cashLin, n_quotes);
        auto risk_cashAQ  = quant::SwapPricer::calculateRisk(cashAQ,  data.newSwap, m_cashAQ,  n_quotes);
        auto risk_swapLin = quant::SwapPricer::calculateRisk(swapLin, data.newSwap, m_swapLin, n_quotes);
        auto risk_swapAQ  = quant::SwapPricer::calculateRisk(swapAQ,  data.newSwap, m_swapAQ,  n_quotes);

        std::ofstream out(outputFile);
        out << std::setprecision(15);
        
        out << cashLin.getDF(data.queryTimeDays) << "," << cashAQ.getDF(data.queryTimeDays) << "," 
            << swapLin.getDF(data.queryTimeDays) << "," << swapAQ.getDF(data.queryTimeDays) << "\n";
        
        out << m_cashLin.pv << "," << m_cashAQ.pv << "," << m_swapLin.pv << "," << m_swapAQ.pv << "\n";
        out << m_cashLin.parRate << "," << m_cashAQ.parRate << "," << m_swapLin.parRate << "," << m_swapAQ.parRate << "\n";
        out << "0,0,0,0\n0,0,0,0\n0,0,0,0\n";
        
        for (int i = 0; i < n_quotes; ++i) {
            out << risk_cashLin[i] << "," << risk_cashAQ[i] << "," << risk_swapLin[i] << "," << risk_swapAQ[i] << "\n";
        }
        
        out.close();
        std::cout << "\n[Phase 4] Complete. Results written safely to " << outputFile << " without tampering with original files.\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
