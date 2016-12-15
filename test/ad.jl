using Distributions
using Turing
using Turing: get_gradient_dict, invlink, link
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

# Get current working variables in Turing
s = chain[:s][1]
s = link(InverseGamma(2,3), s)  # the AD in HMC work in R, so we need do X -> R
m = chain[:m][1]

# Call Turing's AD
# The result out is the gradient information on R
gi = Turing.sampler.values
∇E = get_gradient_dict(gi, ad_test)
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
x = [m, s]
grad_FWAD = sort(-g(x))

# Compare result
@test_approx_eq_eps grad_Turing grad_FWAD 1e-9
