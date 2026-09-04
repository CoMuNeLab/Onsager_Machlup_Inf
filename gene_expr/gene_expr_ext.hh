#ifndef GENE_EXPR_EXT_HH
#define GENE_EXPR_EXT_HH

#include <inference/uncertain/dynamics_continuous.hh>
#include <cmath>

namespace graph_tool
{

class GeneExprState
    : public ContinuousStateBase<GeneExprState, false, true>
{
public:
    using typename ContinuousStateBase<GeneExprState, false, true>::base_t;

    template <class... Args>
    GeneExprState(python::dict p, Args&&... s)
        : ContinuousStateBase<GeneExprState, false, true>(s...)
    {
        set_params(p);
    };

    virtual void set_params(python::dict p)
    {
        _dt = python::extract<double>(p.get("dt", 1.0));
        _ldt = log(_dt);
    }

    [[gnu::flatten]] [[gnu::always_inline]]
    double log_P(const std::vector<double>& theta, double m, double x, double nx)
    {
        double lgamma = (theta.size() > 0) ? std::max(-15.0, std::min(15.0, theta[0])) : 0.0;
        double lV     = (theta.size() > 1) ? std::max(-15.0, std::min(15.0, theta[1])) : 0.0;
        double lD     = (theta.size() > 2) ? std::max(-15.0, std::min(15.0, theta[2])) : -3.0;
        double gamma  = exp(lgamma);
        double V      = exp(lV);
        double D      = exp(lD);

        if (!std::isfinite(m)) m = 0.0;
        if (!std::isfinite(x)) x = 1.0;
        if (!std::isfinite(nx)) nx = 1.0;

        // Ensure x is strictly positive to prevent log(0) and division by zero
        double x_safe = std::max(0.01, x); 

        double sig_m = 1.0 / (1.0 + exp(-m));
        double dt = std::max(1e-6, _dt);
        // UPGRADE: Analytical CIR integration
        double mu_inf = (V * sig_m) / gamma;
        
        // Exact Mean (Analytical Solution)
        double exp_g = exp(-gamma * dt);
        double mu = mu_inf + (x_safe - mu_inf) * exp_g;
        
        // Exact Variance (Ito Isometry Solution)
        double exp_2g = exp(-2.0 * gamma * dt);
        double exact_var = (D / gamma) * (x_safe - mu_inf) * (exp_g - exp_2g) 
                         + (D * mu_inf) / (2.0 * gamma) * (1.0 - exp_2g);
        
        // Standard deviation for the Gaussian likelihood
        double s = sqrt(std::max(1e-10, exact_var)); 
        double log_s = log(s);

        double res = norm_lpdf(nx, mu, s, log_s);
        if (!std::isfinite(res) || std::abs(res) > 1e10)
        {
            std::cerr << "[GeneExprState WARNING] bad log_P: res=" << res 
                      << " x=" << x << " nx=" << nx << " m=" << m 
                      << " theta.size=" << theta.size() << "\n";
            return -1e10;
        }
        return res;
    }

    double _dt = 1.0;
    double _ldt = 0.0;
};

} // namespace graph_tool

#endif // GENE_EXPR_EXT_HH
