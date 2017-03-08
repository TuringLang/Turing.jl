using Distributions
using Turing
using Turing: gradient, invlink, link
using ForwardDiff
using ForwardDiff: Dual
using Base.Test

# Define model
@model ad_test begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  1.5 ~ Normal(m, sqrt(s))
  2.0 ~ Normal(m, sqrt(s))
  s, m
end
Turing.TURING[:modelex]
# Call Turing's AD
# The result out is the gradient information on R
gi = ad_test()
_s = realpart(gi.values[Var(:s)][1])
_m = realpart(gi.values[Var(:m)][1])
∇E = gradient(gi, ad_test, Dict(), nothing)
grad_Turing = sort([∇E[v][1] for v in keys(gi)])

# Hand-written logjoint
function logjoint(x::Vector)
  s = x[2]
  dist_s = InverseGamma(2,3)
  s = invlink(dist_s, s)        # as we now work in R, we need to do R -> X for s
  m = x[1]
  lik_dist = Normal(m, sqrt(s))
  lp = logpdf(dist_s, s, true) + logpdf(Normal(0,sqrt(s)), m, true)
  lp += logpdf(lik_dist, 1.5) + logpdf(lik_dist, 2.0)
  lp
end

# Call ForwardDiff's AD
g = x -> ForwardDiff.gradient(logjoint, x);
_x = [_m, _s]
grad_FWAD = sort(-g(_x))

# Compare result
@test_approx_eq_eps grad_Turing grad_FWAD 1e-9
