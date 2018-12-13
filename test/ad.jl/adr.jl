using Turing
using Turing: gradient_reverse, invlink, link, getval
using ForwardDiff
using ForwardDiff: Dual
using Test

# Define model
@model ad_test() = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  1.5 ~ Normal(m, sqrt(s))
  2.0  ~ Normal(m, sqrt(s))
  return s, m
end
# Turing.TURING[:modelex]
# Call Turing's AD
# The result out is the gradient information on R
ad_test_f = ad_test()
vi = Turing.VarInfo()
ad_test_f(vi, nothing)
svn = collect(Iterators.filter(vn -> vn.sym == :s, keys(vi)))[1]
mvn = collect(Iterators.filter(vn -> vn.sym == :m, keys(vi)))[1]
_s = getval(vi, svn)[1]
_m = getval(vi, mvn)[1]

x = map(x->Float64(x), vi[nothing])
∇E = gradient_reverse(x, vi, ad_test_f)[2]
# println(vi.vns)
# println(∇E)
grad_Turing = sort(∇E)

dist_s = InverseGamma(2,3)

# Hand-written logp
function logp(x::Vector)
  s = x[2]
  # s = invlink(dist_s, s)
  m = x[1]
  lik_dist = Normal(m, sqrt(s))
  lp = logpdf(dist_s, s) + logpdf(Normal(0,sqrt(s)), m)
  lp += logpdf(lik_dist, 1.5) + logpdf(lik_dist, 2.0)
  lp
end

# Call ForwardDiff's AD
g = x -> ForwardDiff.gradient(logp, x);
# _s = link(dist_s, _s)
_x = [_m, _s]
grad_FWAD = sort(-g(_x))

# Compare result
println(grad_Turing)
println(grad_FWAD)
@test grad_Turing ≈ grad_FWAD atol=1e-9
