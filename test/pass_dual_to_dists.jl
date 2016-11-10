# Test if we can pass Dual types to Distributions

using Turing
using Distributions
using ForwardDiff: Dual
using Base.Test

f1 = 1.1
f2 = 1.1
d1 = Dual(1.1)
d2 = Dual(2.3)

@test logpdf(Normal(0, 1), d1).value == logpdf(Normal(0, 1), f1)
@test logpdf(Gamma(2, 3), d2).value == logpdf(Gamma(2, 3), f2)
@test logpdf(Beta(2, 3), (d2 - d1) / 2).value == logpdf(Beta(2, 3), (f2 - f1) / 2)

@test pdf(Normal(0, 1), d1).value == pdf(Normal(0, 1), f1)
@test pdf(Gamma(2, 3), d2).value == pdf(Gamma(2, 3), f2)
@test pdf(Beta(2, 3), (d2 - d1) / 2).value == pdf(Beta(2, 3), (f2 - f1) / 2)
