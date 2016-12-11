using Distributions
using Turing
using Turing: get_gradient_dict
using ForwardDiff
using ForwardDiff: Dual
using Base.Test

# Define model
@model ad_test begin
  @assume s ~ InverseGamma(2,3)
  @assume m ~ Normal(0,sqrt(s))
  @observe 1.5 ~ Normal(m, sqrt(s))
  @observe 2.0 ~ Normal(m, sqrt(s))
  @predict s m
end

# Run HMC to gen GradientInfo, model, etc
chain = sample(ad_test, HMC(1, 0.1, 1))
chain[:s]

# Get s and m from the result
gi = Turing.sampler.values
vars = [k for k in keys(gi)]
v1 = vars[1]
v2 = vars[2]
s = gi[v1][1].value
m = gi[v2][1].value

# Call Turing's AD
∇E = get_gradient_dict(gi, ad_test)
∇E[v1][1]
∇E[v2][1]

# Hand-written logjoint
function logjoint(x::Vector)
  s = x[1]
  m = x[2]
  lik_dist = Normal(m, sqrt(s))
  lp = logpdf(InverseGamma(2,3), s) + logpdf(Normal(0,sqrt(s)), m)
  lp += logpdf(lik_dist, 1.5) + logpdf(lik_dist, 2.0)
  lp
end

# Call ForwardDiff's AD
g = x -> ForwardDiff.gradient(logjoint, x);
x = [s, m]
logjoint(x)
∇E_ad = -g(x)
∇E_ad[1]
∇E_ad[2]

# Compare result
@test_approx_eq_eps ∇E[v1][1] ∇E_ad[1] 1e-9
@test_approx_eq_eps ∇E[v2][1] ∇E_ad[2] 1e-9
