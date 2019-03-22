# Test if we can pass Dual types to Distributions

using Turing
using ForwardDiff: Dual
using Test

float1 = 1.1
float2 = 2.3
d1 = Dual(1.1)
d2 = Dual(2.3)

@test logpdf(Normal(0, 1), d1).value ≈ logpdf(Normal(0, 1), float1) atol=0.001
@test logpdf(Gamma(2, 3), d2).value ≈ logpdf(Gamma(2, 3), float2) atol=0.001
@test logpdf(Beta(2, 3), (d2 - d1) / 2).value ≈ logpdf(Beta(2, 3), (float2 - float1) / 2) atol=0.001

@test pdf(Normal(0, 1), d1).value ≈ pdf(Normal(0, 1), float1) atol=0.001
@test pdf(Gamma(2, 3), d2).value ≈ pdf(Gamma(2, 3), float2) atol=0.001
@test pdf(Beta(2, 3), (d2 - d1) / 2).value ≈ pdf(Beta(2, 3), (float2 - float1) / 2) atol=0.001
