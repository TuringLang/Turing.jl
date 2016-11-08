# Test if we can pass Dual types to Distributions

using Distributions
using ForwardDiff: Dual

d1 = Dual(1.1)
d2 = Dual(2.3)

logpdf(Normal(0, 1), d1)
logpdf(Gamma(2, 3), d2)
logpdf(Beta(2, 3), (d2 - d1) / 2)
