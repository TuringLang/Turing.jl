# Test file for dDistribution wrapper w.r.t to the pdf function and gradient returned by AD.

using Turing, DualNumbers, Distributions, Base.Test, ForwardDiff

# InverseGamma
ddIG = dInverseGamma(2, 3)
@test pdf(ddIG, 1) ≈ realpart(pdf(ddIG, Dual(1)))
@test ForwardDiff.gradient(x::Vector -> hmcInverseGamma(2.0, 3.0)(x[1]), [1])[1] ≈ gradient(ddIG, 1)


# Normal
ddN = dNormal(0, 1)
@test pdf(ddN, 1) ≈ realpart(pdf(ddN, Dual(1)))
@test ForwardDiff.gradient(x::Vector -> hmcNormal(0, 1)(x[1]), [1])[1] ≈ gradient(ddN, 1)

# MvNormal
μ = [1, 1]
Σ = [1 0; 0 1]
ddMN = dMvNormal(μ, Σ)
@test pdf(ddMN, [2, 1]) ≈ realpart(pdf(ddMN, Dual[2, 1]))
@test ForwardDiff.gradient(x::Vector -> hmcMvNormal(μ, Σ)(x), [2, 1]) ≈ gradient(ddMN, [2, 1])
rand(ddMN)
# Bernoulli
ddB = dBernoulli(0.3)
@test pdf(ddB, 1) ≈ realpart(pdf(ddB, Dual(1)))
@test ForwardDiff.gradient(x::Vector -> hmcBernoulli(0.3)(x[1]), [1])[1] ≈ gradient(ddB, 1)
