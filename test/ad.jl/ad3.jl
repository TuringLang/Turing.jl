using Distributions
using Turing
using Turing: gradient, invlink, link, realpart
using ForwardDiff
using ForwardDiff: Dual
using Base.Test


# Define model
@model ad_test_3() = begin
  v ~ Wishart(7, [1 0.5; 0.5 1])
  v
end
# Turing.TURING[:modelex]
# Call Turing's AD
# The result out is the gradient information on R
ad_test_3_f = ad_test_3()
vi = ad_test_3_f()
vvn = collect(filter(vn -> vn.sym == :v, keys(vi)))[1]
_v = realpart(vi[vvn])
grad_Turing = gradient(vi, ad_test_3_f)

dist_v = Wishart(7, [1 0.5; 0.5 1])

# Hand-written logp
function logp3(x::Vector)
  v = [x[1] x[3]; x[2] x[4]]
  lp = logpdf(dist_v, v, false)
  lp
end

# Call ForwardDiff's AD
g = x -> ForwardDiff.gradient(logp3, vec(x));
# _s = link(dist_v, _s)
grad_FWAD = -g(_v)

# Compare result
@test_approx_eq_eps grad_Turing grad_FWAD 1e-9
