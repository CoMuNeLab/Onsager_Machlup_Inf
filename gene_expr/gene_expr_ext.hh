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

    [[gnu::pure]] [[gnu::flatten]] [[gnu::always_inline]]
    double log_P(const std::vector<double>& theta, double m, double x, double nx)
    {
        // Clamp log parameters to safe numerical range [-15.0, 15.0]
        // to prevent overflow/underflow during unrestricted MCMC proposals
        double lgamma = std::max(-15.0, std::min(15.0, theta[0]));
        double lV     = std::max(-15.0, std::min(15.0, theta[1]));
        double lD     = std::max(-15.0, std::min(15.0, theta[2]));
        double gamma  = exp(lgamma);
        double V      = exp(lV);
        double D      = exp(lD);

        // Sigmoid activation function: sigma(m) = 1 / (1 + exp(-m))
        // where m = sum_j W_{i,j} * x_j
        double sig_m = 1.0 / (1.0 + exp(-m));
        
        // Euler-Maruyama drift: mu = x + dt * (-gamma * x + V * sigma(m))
        double mu = x + _dt * (-gamma * x + V * sig_m);
        
        // Normal standard deviation: s = sqrt(D * dt)
        double s = sqrt(D * _dt);
        double log_s = 0.5 * (lD + _ldt);

        // Conditional log-probability density (Onsager-Machlup approximation)
        return norm_lpdf(nx, mu, s, log_s);
    }

    double _dt = 1.0;
    double _ldt = 0.0;
};

} // namespace graph_tool

#endif // GENE_EXPR_EXT_HH
