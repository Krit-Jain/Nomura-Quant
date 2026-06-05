/**
 * ============================================================================
 *  Nomura Quant Challenge — Interest Rate Curve Construction System
 * ============================================================================
 *
 *  A production-grade discount curve bootstrapper and swap pricing engine.
 *
 *  Architecture Overview:
 *  ─────────────────────
 *  The system is built around three extensible abstractions:
 *
 *  1. IInterpolator — Strategy pattern for interpolation methods.
 *     Provides both value interpolation and analytical derivative
 *     computation (∂lnDF/∂lnDFₖ) required for risk sensitivities.
 *
 *  2. IInstrument  — Represents any calibration instrument (cash deposit,
 *     swap, FRA, etc.) that can imply a discount factor from its market
 *     quote, enabling generic curve bootstrapping.
 *
 *  3. DiscountCurve — Owns the (time, DF) node set and an IInterpolator.
 *     Supports calibration, interpolation, forward rate computation,
 *     and Jacobian extraction for analytical risk propagation.
 *
 *  Risk sensitivities are computed analytically via the chain rule:
 *     ∂PV/∂qₖ = Σⱼ (∂PV/∂DFⱼ) × (∂DFⱼ/∂qₖ)
 *  where the calibration Jacobian ∂DFⱼ/∂qₖ is derived using the
 *  Implicit Function Theorem on the bootstrapping equations.
 *
 *  Compiler: C++17 or later (clang compatible)
 *  Usage:    ./solution Input.csv Output.csv
 *
 * ============================================================================
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <cmath>
#include <memory>
#include <algorithm>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <cassert>
#include <iomanip>
#include <optional>
#include <variant>

// ============================================================================
//  Section 1: Constants and Day-Count Conventions
// ============================================================================

namespace constants {
    constexpr int    DAYS_PER_WEEK  = 7;
    constexpr int    DAYS_PER_MONTH = 30;
    constexpr int    DAYS_PER_YEAR  = 360;
    constexpr double NOTIONAL       = 100.0;
    constexpr double DF_AT_ZERO     = 1.0;      // DF(t=0) is always 1
    constexpr int    SEMI_ANNUAL    = 180;       // Calibration swap frequency (days)
    constexpr double NEWTON_TOL     = 1e-14;     // Newton solver convergence
    constexpr int    NEWTON_MAX     = 200;       // Newton solver max iterations
    constexpr double BUMP_SIZE      = 1e-6;      // For validation only (not used in analytical risk)
}

// ============================================================================
//  Section 2: Maturity and Frequency Convention Parsing
// ============================================================================

/**
 * Converts a maturity string (e.g., "1D", "2W", "3M", "25Y") to days.
 *
 * Convention:
 *   nD → n days
 *   nW → n × 7 days
 *   nM → n × 30 days
 *   nY → n × 360 days
 *
 * Throws std::invalid_argument for unrecognized formats.
 */
int parseTenorToDays(const std::string& tenor) {
    if (tenor.empty()) {
        throw std::invalid_argument("Empty tenor string");
    }

    char unit = tenor.back();
    std::string numStr = tenor.substr(0, tenor.size() - 1);

    // Handle case-insensitive unit
    unit = static_cast<char>(std::toupper(static_cast<unsigned char>(unit)));

    int n = 0;
    try {
        n = std::stoi(numStr);
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid numeric part in tenor: " + tenor);
    }

    if (n <= 0) {
        throw std::invalid_argument("Tenor multiplier must be positive: " + tenor);
    }

    switch (unit) {
        case 'D': return n;
        case 'W': return n * constants::DAYS_PER_WEEK;
        case 'M': return n * constants::DAYS_PER_MONTH;
        case 'Y': return n * constants::DAYS_PER_YEAR;
        default:
            throw std::invalid_argument("Unknown tenor unit '" + std::string(1, unit) + "' in: " + tenor);
    }
}

/**
 * Converts a frequency string (e.g., "1m", "3m", "6m", "12m") to days
 * between payments.
 *
 * Convention:
 *   "1m"  → 30 days  (monthly)
 *   "3m"  → 90 days  (quarterly)
 *   "6m"  → 180 days (semi-annual)
 *   "12m" → 360 days (annual)
 */
int parseFrequencyToDays(const std::string& freq) {
    if (freq.empty()) {
        throw std::invalid_argument("Empty frequency string");
    }

    // Frequency is always in months, suffix 'm'
    char unit = freq.back();
    if (unit != 'm' && unit != 'M') {
        throw std::invalid_argument("Frequency must end with 'm': " + freq);
    }

    std::string numStr = freq.substr(0, freq.size() - 1);
    int months = 0;
    try {
        months = std::stoi(numStr);
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid numeric part in frequency: " + freq);
    }

    if (months <= 0 || months > 12) {
        throw std::invalid_argument("Frequency months must be in [1,12]: " + freq);
    }

    return months * constants::DAYS_PER_MONTH;
}

/**
 * Day count fraction: DCF(t2, t1) = (t2 - t1) / 360
 * t1, t2 in days.
 */
inline double dcf(int t2, int t1) {
    return static_cast<double>(t2 - t1) / constants::DAYS_PER_YEAR;
}

// ============================================================================
//  Section 3: Market Data Structures
// ============================================================================

/**
 * A single market instrument quote — maturity + cash rate + par swap rate.
 */
struct MarketQuote {
    std::string tenorLabel;   // Original label (e.g., "2Y")
    int         maturityDays; // Maturity in days
    double      cashRate;     // Cash rate (as decimal, e.g., 0.0364 for 3.64%)
    double      parSwapRate;  // Par swap rate (as decimal)
};

/**
 * Parameters for a new swap to be priced.
 */
struct SwapSpec {
    double      fixedRate;        // Fixed rate (as decimal, e.g., 0.06 for 6%)
    int         maturityDays;     // Maturity in days
    int         fixedFreqDays;    // Fixed leg payment frequency in days
    int         floatFreqDays;    // Floating leg payment frequency in days
    double      notional;         // Notional amount
};

/**
 * Complete input data parsed from the CSV.
 */
struct MarketData {
    std::vector<MarketQuote> quotes;
    int                      queryTimeDays;  // Time t for Q1 (discount factor query)
    SwapSpec                 newSwap;        // New swap for Q2
};

/**
 * All computed results to be written to Output.csv.
 */
struct Results {
    // Q1: DF(t) — 4 values: Cash/Linear, Cash/AQ, Swap/Linear, Swap/AQ
    std::array<double, 4> df;

    // Q2.1: PV and Par Rate — 4 values each
    std::array<double, 4> pv;
    std::array<double, 4> parRate;

    // Q2.2: Risk vectors — 4 columns × N rows (N = number of market quotes)
    std::vector<std::array<double, 4>> risk;
};

// ============================================================================
//  Section 4: CSV I/O
// ============================================================================

/**
 * Trims whitespace and carriage returns from a string.
 */
std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

/**
 * Splits a CSV line into fields.
 */
std::vector<std::string> splitCSV(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(trim(field));
    }
    return fields;
}

/**
 * Reads and parses Input.csv into a MarketData structure.
 *
 * Format:
 *   Row 1:       N (number of maturities)
 *   Rows 2..N+1: Tenor, CashRate(%), ParSwapRate(%)
 *   Row N+2:     t (query time in days)
 *   Row N+3:     FixedRate(%), Maturity, FixedFreq, FloatFreq
 */
MarketData readInput(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open input file: " + filename);
    }

    MarketData data;
    std::string line;

    // Strip UTF-8 BOM if present (EF BB BF)
    {
        char bom[3] = {};
        file.read(bom, 3);
        if (!(static_cast<unsigned char>(bom[0]) == 0xEF &&
              static_cast<unsigned char>(bom[1]) == 0xBB &&
              static_cast<unsigned char>(bom[2]) == 0xBF)) {
            file.seekg(0); // No BOM, rewind
        }
    }

    // Row 1: Number of maturities
    if (!std::getline(file, line)) {
        throw std::runtime_error("Input file is empty");
    }
    auto fields = splitCSV(line);
    int numMaturities = std::stoi(fields[0]);

    if (numMaturities <= 0 || numMaturities > 100) {
        throw std::runtime_error("Invalid number of maturities: " + std::to_string(numMaturities));
    }

    // Rows 2..N+1: Market quotes
    data.quotes.reserve(numMaturities);
    for (int i = 0; i < numMaturities; ++i) {
        if (!std::getline(file, line)) {
            throw std::runtime_error("Unexpected end of file at quote row " + std::to_string(i + 1));
        }
        fields = splitCSV(line);
        if (fields.size() < 3) {
            throw std::runtime_error("Insufficient fields at quote row " + std::to_string(i + 1));
        }

        MarketQuote q;
        q.tenorLabel   = fields[0];
        q.maturityDays = parseTenorToDays(fields[0]);
        q.cashRate     = std::stod(fields[1]) / 100.0;   // Convert from % to decimal
        q.parSwapRate  = std::stod(fields[2]) / 100.0;
        data.quotes.push_back(q);
    }

    // Verify maturities are in ascending order
    for (size_t i = 1; i < data.quotes.size(); ++i) {
        if (data.quotes[i].maturityDays <= data.quotes[i - 1].maturityDays) {
            throw std::runtime_error("Maturities must be in strictly ascending order. "
                "Violation at index " + std::to_string(i) + ": " +
                data.quotes[i].tenorLabel + " (" + std::to_string(data.quotes[i].maturityDays) +
                " days) <= " + data.quotes[i-1].tenorLabel + " (" +
                std::to_string(data.quotes[i-1].maturityDays) + " days)");
        }
    }

    // Row N+2: Query time t
    if (!std::getline(file, line)) {
        throw std::runtime_error("Missing query time row");
    }
    fields = splitCSV(line);
    data.queryTimeDays = std::stoi(fields[0]);

    // Row N+3: New swap spec
    if (!std::getline(file, line)) {
        throw std::runtime_error("Missing new swap specification row");
    }
    fields = splitCSV(line);
    if (fields.size() < 4) {
        throw std::runtime_error("Insufficient fields for swap specification");
    }

    data.newSwap.fixedRate     = std::stod(fields[0]) / 100.0;
    data.newSwap.maturityDays  = parseTenorToDays(fields[1]);
    data.newSwap.fixedFreqDays = parseFrequencyToDays(fields[2]);
    data.newSwap.floatFreqDays = parseFrequencyToDays(fields[3]);
    data.newSwap.notional      = constants::NOTIONAL;

    return data;
}

/**
 * Writes Results to Output.csv in the exact required format.
 *
 * Format:
 *   Row 1:       Q1 results (4 DFs)
 *   Row 2:       Q2.1 PV (4 values)
 *   Row 3:       Q2.1 Par Rates (4 values)
 *   Rows 4..N+3: Q2.2 Risk vectors (4 columns × N rows)
 */
void writeOutput(const std::string& filename, const Results& results) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open output file: " + filename);
    }

    // Use sufficient precision for financial computations
    file << std::setprecision(15);

    // Row 1: Q1 — DF values
    file << results.df[0] << "," << results.df[1] << ","
         << results.df[2] << "," << results.df[3] << "\n";

    // Row 2: Q2.1 — PV values
    file << results.pv[0] << "," << results.pv[1] << ","
         << results.pv[2] << "," << results.pv[3] << "\n";

    // Row 3: Q2.1 — Par swap rates
    file << results.parRate[0] << "," << results.parRate[1] << ","
         << results.parRate[2] << "," << results.parRate[3] << "\n";

    // Rows 4..N+3: Q2.2 — Risk vectors
    for (const auto& row : results.risk) {
        file << row[0] << "," << row[1] << ","
             << row[2] << "," << row[3] << "\n";
    }
}

// ============================================================================
//  Section 5: Schedule Generation
// ============================================================================

/**
 * Generates a payment schedule: evenly-spaced dates from freqDays to maturityDays.
 *
 * Returns a vector of payment times in days: {freq, 2*freq, ..., maturity}.
 * The last element is always maturityDays (capped if not perfectly divisible).
 *
 * Example: generateSchedule(9000, 180) → {180, 360, ..., 8820, 9000}
 */
std::vector<int> generateSchedule(int maturityDays, int freqDays) {
    if (freqDays <= 0) {
        throw std::invalid_argument("Payment frequency must be positive");
    }
    if (maturityDays <= 0) {
        throw std::invalid_argument("Maturity must be positive");
    }

    std::vector<int> schedule;
    for (int t = freqDays; t < maturityDays; t += freqDays) {
        schedule.push_back(t);
    }
    // Always include maturity as last payment
    if (schedule.empty() || schedule.back() != maturityDays) {
        schedule.push_back(maturityDays);
    }
    return schedule;
}


// ============================================================================
//  Section 6: Interpolation (Strategy Pattern)
// ============================================================================

/**
 * Interface for interpolation strategies.
 * Operates on the natural logarithm of the discount factor: ln(DF).
 */
class IInterpolator {
public:
    virtual ~IInterpolator() = default;

    /**
     * Interpolates ln(DF) at time t.
     * @param t Target time in days.
     * @param times Sorted vector of node times.
     * @param lnDFs Vector of ln(DF) corresponding to node times.
     * @return Interpolated ln(DF(t)).
     */
    virtual double interpolate(double t, const std::vector<double>& times, const std::vector<double>& lnDFs) const = 0;

    /**
     * Computes analytical partial derivatives ∂lnDF(t)/∂lnDF(t_k) for each node k.
     * @param t Target time in days.
     * @param times Sorted vector of node times.
     * @return A map of node_index -> derivative value.
     */
    virtual std::map<int, double> interpolateDerivatives(double t, const std::vector<double>& times) const = 0;
};

/**
 * Linear Interpolation on ln(DF).
 * Corresponds to piecewise-constant forward rates.
 */
class LinearLogDFInterpolator : public IInterpolator {
public:
    double interpolate(double t, const std::vector<double>& times, const std::vector<double>& lnDFs) const override {
        if (times.empty()) return 0.0;
        if (t <= times.front()) return lnDFs.front();
        if (t >= times.back()) return lnDFs.back(); // Flat extrapolation

        auto it = std::lower_bound(times.begin(), times.end(), t);
        int i = std::distance(times.begin(), it);
        int prev = i - 1;

        double w = (t - times[prev]) / (times[i] - times[prev]);
        return lnDFs[prev] + w * (lnDFs[i] - lnDFs[prev]);
    }

    std::map<int, double> interpolateDerivatives(double t, const std::vector<double>& times) const override {
        std::map<int, double> derivs;
        if (times.empty()) return derivs;
        
        if (t <= times.front()) {
            derivs[0] = 1.0;
            return derivs;
        }
        if (t >= times.back()) {
            derivs[times.size() - 1] = 1.0;
            return derivs;
        }

        auto it = std::lower_bound(times.begin(), times.end(), t);
        int i = std::distance(times.begin(), it);
        int prev = i - 1;

        double w = (t - times[prev]) / (times[i] - times[prev]);
        derivs[prev] = 1.0 - w;
        derivs[i]    = w;

        return derivs;
    }
};

/**
 * Averaged Quadratic Interpolation on ln(DF).
 */
class AveragedQuadraticLogDFInterpolator : public IInterpolator {
public:
    double interpolate(double t, const std::vector<double>& times, const std::vector<double>& lnDFs) const override {
        int n = times.size();
        if (n < 3) {
            LinearLogDFInterpolator lin;
            return lin.interpolate(t, times, lnDFs);
        }
        if (t <= times.front()) return lnDFs.front();
        if (t >= times.back()) return lnDFs.back();

        auto it = std::lower_bound(times.begin(), times.end(), t);
        int i = std::distance(times.begin(), it);
        
        if (i == 1) {
            LinearLogDFInterpolator lin;
            return lin.interpolate(t, times, lnDFs);
        }

        double dt = times[i] - times[i-1];
        double w_prev = (times[i] - t) / dt;
        double w_curr = (t - times[i-1]) / dt;

        if (i == n - 1) {
            return evalQ(i - 1, t, times, lnDFs);
        }

        double q_prev = evalQ(i - 1, t, times, lnDFs);
        double q_curr = evalQ(i, t, times, lnDFs);

        return w_prev * q_prev + w_curr * q_curr;
    }

    std::map<int, double> interpolateDerivatives(double t, const std::vector<double>& times) const override {
        int n = times.size();
        if (n < 3) {
            LinearLogDFInterpolator lin;
            return lin.interpolateDerivatives(t, times);
        }
        
        std::map<int, double> derivs;
        if (t <= times.front()) { derivs[0] = 1.0; return derivs; }
        if (t >= times.back()) { derivs[n - 1] = 1.0; return derivs; }

        auto it = std::lower_bound(times.begin(), times.end(), t);
        int i = std::distance(times.begin(), it);

        if (i == 1) {
            LinearLogDFInterpolator lin;
            return lin.interpolateDerivatives(t, times);
        }

        double dt = times[i] - times[i-1];
        double w_prev = (times[i] - t) / dt;
        double w_curr = (t - times[i-1]) / dt;

        if (i == n - 1) {
            auto dQ_prev = evalQDerivs(i - 1, t, times);
            for (auto const& [k, v] : dQ_prev) derivs[k] += v;
            return derivs;
        }

        auto dQ_prev = evalQDerivs(i - 1, t, times);
        auto dQ_curr = evalQDerivs(i, t, times);

        for (auto const& [k, v] : dQ_prev) derivs[k] += w_prev * v;
        for (auto const& [k, v] : dQ_curr) derivs[k] += w_curr * v;

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

    std::map<int, double> evalQDerivs(int idx, double t, const std::vector<double>& times) const {
        double t0 = times[idx-1], t1 = times[idx], t2 = times[idx+1];
        
        double L0 = (t - t1) * (t - t2) / ((t0 - t1) * (t0 - t2));
        double L1 = (t - t0) * (t - t2) / ((t1 - t0) * (t1 - t2));
        double L2 = (t - t0) * (t - t1) / ((t2 - t0) * (t2 - t1));

        std::map<int, double> derivs;
        derivs[idx-1] = L0;
        derivs[idx]   = L1;
        derivs[idx+1] = L2;
        return derivs;
    }
};

// ============================================================================
//  Section 7: Discount Curve
// ============================================================================

class DiscountCurve {
public:
    DiscountCurve(std::unique_ptr<IInterpolator> interpolator)
        : interpolator_(std::move(interpolator)) {
        // Curve starts with DF(0) = 1.0
        addNode(0, 1.0);
    }

    void addNode(int timeDays, double df) {
        if (!times_.empty() && timeDays <= times_.back()) {
            throw std::invalid_argument("Nodes must be added in strictly increasing time order");
        }
        times_.push_back(timeDays);
        dfs_.push_back(df);
        lnDFs_.push_back(std::log(df));
    }

    void updateNode(int index, double df) {
        if (index <= 0 || index >= static_cast<int>(times_.size())) {
            throw std::out_of_range("Invalid node index for update");
        }
        dfs_[index] = df;
        lnDFs_[index] = std::log(df);
    }

    double getDF(int timeDays) const {
        if (timeDays == 0) return 1.0;
        double lnDF = interpolator_->interpolate(timeDays, times_, lnDFs_);
        return std::exp(lnDF);
    }

    double getForwardRate(int t1, int t2) const {
        if (t1 >= t2) throw std::invalid_argument("Forward rate requires t1 < t2");
        double df1 = getDF(t1);
        double df2 = getDF(t2);
        return (df1 / df2 - 1.0) / dcf(t2, t1);
    }

    std::map<int, double> getDFNodeDerivatives(int timeDays) const {
        std::map<int, double> dfDerivs;
        if (timeDays == 0) return dfDerivs;

        double currentDF = getDF(timeDays);
        auto lnDFDerivs = interpolator_->interpolateDerivatives(timeDays, times_);

        for (const auto& kv : lnDFDerivs) {
            int nodeIdx = kv.first;
            double d_lnDF = kv.second;
            if (nodeIdx == 0) continue; 
            
            double dDF_k = dfs_[nodeIdx];
            dfDerivs[nodeIdx] = currentDF * d_lnDF / dDF_k;
        }
        return dfDerivs;
    }

    const std::vector<double>& getTimes() const { return times_; }
    const std::vector<double>& getDFs() const { return dfs_; }

    std::vector<std::vector<double>> jacobian;

private:
    std::vector<double> times_;
    std::vector<double> dfs_;
    std::vector<double> lnDFs_;
    std::unique_ptr<IInterpolator> interpolator_;
};

// ============================================================================
//  Section 8: Bootstrapping & Calibration
// ============================================================================

/**
 * Bootstraps a cash curve. Direct computation.
 * DF(T) = 1 / (1 + cashRate * (T/360))
 */
void bootstrapCashCurve(DiscountCurve& curve, const std::vector<MarketQuote>& quotes) {
    int n = quotes.size();
    curve.jacobian.assign(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i) {
        int T = quotes[i].maturityDays;
        double c = quotes[i].cashRate;
        double tau = dcf(T, 0);
        
        double df = 1.0 / (1.0 + c * tau);
        curve.addNode(T, df);

        // ∂DF_i / ∂c_i  (Note: Jacobian index i corresponds to curve node i+1 since node 0 is t=0)
        curve.jacobian[i][i] = -tau / ((1.0 + c * tau) * (1.0 + c * tau));
    }
}

/**
 * Bootstraps a swap curve. Solves for DFs sequentially using Newton's method.
 * Computes analytical Jacobian via Implicit Function Theorem.
 */
void bootstrapSwapCurve(DiscountCurve& curve, const std::vector<MarketQuote>& quotes) {
    int n = quotes.size();
    curve.jacobian.assign(n, std::vector<double>(n, 0.0));

    for (int k = 0; k < n; ++k) {
        int Tk = quotes[k].maturityDays;
        double pk = quotes[k].parSwapRate;

        if (Tk <= constants::SEMI_ANNUAL) {
            // Single period instrument (<= 6 months)
            double tau = dcf(Tk, 0);
            double df = 1.0 / (1.0 + pk * tau);
            curve.addNode(Tk, df);
            curve.jacobian[k][k] = -tau / ((1.0 + pk * tau) * (1.0 + pk * tau));
        } else {
            // Swap with semi-annual payments
            auto schedule = generateSchedule(Tk, constants::SEMI_ANNUAL);
            
            // Initial guess for DF_k
            double guessDF = (k > 0) ? curve.getDFs().back() * 0.95 : 0.95; 
            curve.addNode(Tk, guessDF);

            int iter = 0;
            double f = 0.0, df_dx = 0.0;
            do {
                double PV01 = 0.0;
                double dPV01_dx = 0.0; // derivative w.r.t ln(DF_k)

                int prev_t = 0;
                for (int ti : schedule) {
                    double delta = dcf(ti, prev_t);
                    double df_ti = curve.getDF(ti);
                    PV01 += df_ti * delta;

                    auto derivs = curve.getDFNodeDerivatives(ti);
                    double dDFti_dDFk = derivs.count(k+1) ? derivs[k+1] : 0.0;
                    double dDFti_dx = dDFti_dDFk * curve.getDF(Tk); // since dx = d(lnDF_k)
                    
                    dPV01_dx += dDFti_dx * delta;
                    prev_t = ti;
                }

                double pv_float = 1.0 - curve.getDF(Tk);
                double dpv_float_dx = -curve.getDF(Tk);

                double pv_fixed = pk * PV01;
                double dpv_fixed_dx = pk * dPV01_dx;

                f = pv_float - pv_fixed;
                df_dx = dpv_float_dx - dpv_fixed_dx;

                if (std::abs(f) < constants::NEWTON_TOL) break;

                double dx = -f / df_dx;
                double new_lnDF = std::log(curve.getDF(Tk)) + dx;
                
                curve.updateNode(k+1, std::exp(new_lnDF)); 

                iter++;
                if (iter > constants::NEWTON_MAX) {
                    throw std::runtime_error("Newton solver failed to converge for swap maturity " + std::to_string(Tk));
                }
            } while (true);

            // Compute Jacobian for node k using IFT
            std::vector<std::map<int, double>> S_list;
            double PV01 = 0.0;
            int prev_t = 0;
            for (int ti : schedule) {
                double delta = dcf(ti, prev_t);
                PV01 += curve.getDF(ti) * delta;
                S_list.push_back(curve.getDFNodeDerivatives(ti));
                prev_t = ti;
            }

            double C_k = 1.0;
            prev_t = 0;
            for (size_t i = 0; i < schedule.size(); ++i) {
                double delta = dcf(schedule[i], prev_t);
                double s_ik = S_list[i].count(k+1) ? S_list[i][k+1] : 0.0;
                C_k += pk * delta * s_ik;
                prev_t = schedule[i];
            }

            for (int j = 0; j <= k; ++j) {
                double RHS_j = 0.0;
                if (j == k) RHS_j -= PV01;
                
                double sum_l = 0.0;
                for (int l = 0; l < k; ++l) {
                    double term_l = 0.0;
                    int prev_t2 = 0;
                    for (size_t i = 0; i < schedule.size(); ++i) {
                        double delta = dcf(schedule[i], prev_t2);
                        double s_il = S_list[i].count(l+1) ? S_list[i][l+1] : 0.0;
                        term_l += pk * delta * s_il;
                        prev_t2 = schedule[i];
                    }
                    sum_l += term_l * curve.jacobian[l][j];
                }
                RHS_j -= sum_l;

                curve.jacobian[k][j] = RHS_j / C_k;
            }
        }
    }
}


struct SwapMetrics {
    double pv;
    double parRate;
    std::vector<std::map<int, double>> fixedS; // d(DF_ti) / d(DF_node) for fixed leg
    std::map<int, double> matS;                // d(DF_T) / d(DF_node) for maturity
};

SwapMetrics priceSwap(const DiscountCurve& curve, const SwapSpec& spec) {
    auto fixedSched = generateSchedule(spec.maturityDays, spec.fixedFreqDays);
    auto floatSched = generateSchedule(spec.maturityDays, spec.floatFreqDays);
    
    double pvFixed = 0.0;
    double PV01 = 0.0;
    std::vector<std::map<int, double>> fixedS;
    
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
    
    std::map<int, double> matS = curve.getDFNodeDerivatives(spec.maturityDays);
    
    return {pv, parRate, fixedS, matS};
}

/**
 * Calculates the analytical risk sensitivities (PV01) for a 1bp shift in market quotes.
 * Uses the chain rule: dPV/dq_j = sum_k (dPV/dDF_k * dDF_k/dq_j).
 */
std::vector<double> calculateRisk(const DiscountCurve& curve, const SwapSpec& spec, const SwapMetrics& m, int n_quotes) {
    std::vector<double> V(n_quotes, 0.0);
    auto fixedSched = generateSchedule(spec.maturityDays, spec.fixedFreqDays);
    
    // V_k = dPV / dDF_{k+1}
    for (int k = 0; k < n_quotes; ++k) {
        int node = k + 1; // curve node 0 is t=0, so quote k matches node k+1
        double vk = 0.0;
        
        if (m.matS.count(node)) {
            vk += spec.notional * m.matS.at(node); 
        }
        
        int prev_t = 0;
        for (size_t i = 0; i < fixedSched.size(); ++i) {
            double delta = dcf(fixedSched[i], prev_t);
            if (m.fixedS[i].count(node)) {
                vk += spec.fixedRate * spec.notional * delta * m.fixedS[i].at(node);
            }
            prev_t = fixedSched[i];
        }
        V[k] = vk;
    }
    
    std::vector<double> risk(n_quotes, 0.0);
    for (int j = 0; j < n_quotes; ++j) {
        double r = 0.0;
        for (int k = 0; k < n_quotes; ++k) {
            r += V[k] * curve.jacobian[k][j];
        }
        // Multiply by 1e-4 because Risk is for a 1bp (0.0001) shift in quote
        risk[j] = r * 0.0001; 
    }
    return risk;
}

// ============================================================================
//  Section 10: Main Driver (Phase 1 — I/O validation)
// ============================================================================

int main(int argc, char* argv[]) {
    try {
        // Default filenames
        std::string inputFile  = "Input.csv";
        std::string outputFile = "Output.csv";

        if (argc >= 2) inputFile  = argv[1];
        if (argc >= 3) outputFile = argv[2];


        std::cout << "========================================\n";
        std::cout << "  Interest Rate Curve Construction System\n";
        std::cout << "========================================\n\n";
        std::cout.flush();

        // -- Phase 1: Parse input --
        std::cout << "[Phase 1] Reading market data from: " << inputFile << "\n";
        MarketData data = readInput(inputFile);

        std::cout << "  Parsed " << data.quotes.size() << " market quotes:\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  " << std::setw(6) << "Tenor"
                  << std::setw(10) << "Days"
                  << std::setw(12) << "Cash(%)"
                  << std::setw(12) << "Swap(%)" << "\n";
        std::cout << "  " << std::string(40, '-') << "\n";
        for (const auto& q : data.quotes) {
            std::cout << "  " << std::setw(6) << q.tenorLabel
                      << std::setw(10) << q.maturityDays
                      << std::setw(12) << (q.cashRate * 100.0)
                      << std::setw(12) << (q.parSwapRate * 100.0) << "\n";
        }

        std::cout << "\n  Query time t = " << data.queryTimeDays << " days\n";
        std::cout << "\n  New swap specification:\n";
        std::cout << "    Fixed Rate:     " << (data.newSwap.fixedRate * 100.0) << "%\n";
        std::cout << "    Maturity:       " << data.newSwap.maturityDays << " days ("
                  << data.newSwap.maturityDays / constants::DAYS_PER_YEAR << "Y)\n";
        std::cout << "    Fixed Freq:     " << data.newSwap.fixedFreqDays << " days ("
                  << data.newSwap.fixedFreqDays / constants::DAYS_PER_MONTH << "m)\n";
        std::cout << "    Float Freq:     " << data.newSwap.floatFreqDays << " days ("
                  << data.newSwap.floatFreqDays / constants::DAYS_PER_MONTH << "m)\n";
        std::cout << "    Notional:       " << data.newSwap.notional << "\n";

        // ── Phase 2: Cash Curve Bootstrapping ──
        std::cout << "\n[Phase 2] Bootstrapping Cash Curves...\n";
        
        Results results;
        results.df = {0, 0, 0, 0};
        results.pv = {0, 0, 0, 0};
        results.parRate = {0, 0, 0, 0};
        results.risk.resize(data.quotes.size(), {0, 0, 0, 0});
        
        DiscountCurve cashCurveLin(std::make_unique<LinearLogDFInterpolator>());
        bootstrapCashCurve(cashCurveLin, data.quotes);
        double q1a = cashCurveLin.getDF(data.queryTimeDays);

        DiscountCurve cashCurveAQ(std::make_unique<AveragedQuadraticLogDFInterpolator>());
        bootstrapCashCurve(cashCurveAQ, data.quotes);
        double q1b = cashCurveAQ.getDF(data.queryTimeDays);

        results.df[0] = q1a;
        results.df[1] = q1b;

        std::cout << "  Q1.a) DF(" << data.queryTimeDays << ") Cash/Linear: " << q1a << "\n";
        std::cout << "  Q1.b) DF(" << data.queryTimeDays << ") Cash/AQ:     " << q1b << "\n";




        std::cout << "\n[Phase 3] Bootstrapping Swap Curves...\n";
        
        DiscountCurve swapCurveLin(std::make_unique<LinearLogDFInterpolator>());
        bootstrapSwapCurve(swapCurveLin, data.quotes);
        double q1c = swapCurveLin.getDF(data.queryTimeDays);

        DiscountCurve swapCurveAQ(std::make_unique<AveragedQuadraticLogDFInterpolator>());
        bootstrapSwapCurve(swapCurveAQ, data.quotes);
        double q1d = swapCurveAQ.getDF(data.queryTimeDays);

        results.df[2] = q1c;
        results.df[3] = q1d;

        std::cout << "  Q1.c) DF(" << data.queryTimeDays << ") Swap/Linear: " << q1c << "\n";
        std::cout << "  Q1.d) DF(" << data.queryTimeDays << ") Swap/AQ:     " << q1d << "\n";

        std::cout << "\n[Phase 3] DONE - Swap curve bootstrapped successfully.\n";

        // ── Phase 4: Pricing New Swap (Q2.1) ──
        std::cout << "\n[Phase 4] Pricing 25Y Swap...\n";

        auto m_cashLin = priceSwap(cashCurveLin, data.newSwap);
        auto m_cashAQ  = priceSwap(cashCurveAQ, data.newSwap);
        auto m_swapLin = priceSwap(swapCurveLin, data.newSwap);
        auto m_swapAQ  = priceSwap(swapCurveAQ, data.newSwap);

        results.pv = {m_cashLin.pv, m_cashAQ.pv, m_swapLin.pv, m_swapAQ.pv};
        results.parRate = {m_cashLin.parRate, m_cashAQ.parRate, m_swapLin.parRate, m_swapAQ.parRate};

        std::cout << "  PVs (Cash/Lin, Cash/AQ, Swap/Lin, Swap/AQ): \n";
        std::cout << "    " << m_cashLin.pv << ", " << m_cashAQ.pv << ", " 
                  << m_swapLin.pv << ", " << m_swapAQ.pv << "\n";
                  
        std::cout << "  Par Rates (%): \n";
        std::cout << "    " << (m_cashLin.parRate * 100.0) << ", " 
                  << (m_cashAQ.parRate * 100.0) << ", " 
                  << (m_swapLin.parRate * 100.0) << ", " 
                  << (m_swapAQ.parRate * 100.0) << "\n";

        // ── Phase 5: Analytical Risk (Q2.2) ──
        std::cout << "\n[Phase 5] Computing Analytical Risk Sensitivities (1bp)...\n";
        
        int n_quotes = data.quotes.size();
        auto risk_cashLin = calculateRisk(cashCurveLin, data.newSwap, m_cashLin, n_quotes);
        auto risk_cashAQ  = calculateRisk(cashCurveAQ,  data.newSwap, m_cashAQ,  n_quotes);
        auto risk_swapLin = calculateRisk(swapCurveLin, data.newSwap, m_swapLin, n_quotes);
        auto risk_swapAQ  = calculateRisk(swapCurveAQ,  data.newSwap, m_swapAQ,  n_quotes);
        
        for (int i = 0; i < n_quotes; ++i) {
            results.risk[i][0] = risk_cashLin[i];
            results.risk[i][1] = risk_cashAQ[i];
            results.risk[i][2] = risk_swapLin[i];
            results.risk[i][3] = risk_swapAQ[i];
        }

        std::cout << "  Analytical Risk computed for " << n_quotes << " market instruments.\n";
        
        // Output CSV writing
        writeOutput(outputFile, results);
        std::cout << "\n[Phase 6] DONE - Results written to " << outputFile << ".\n";


    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
