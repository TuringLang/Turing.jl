using Turing, DualNumbers, Distributions, Base.Test

# InverseGamma
dd = dInverseGamma(2, 3)
@test pdf(dd, 1) ≈ realpart(pdf(dd, Dual(1, 0)))

# Normal
dd = dNormal(0, 1)
@test pdf(dd, 1) ≈ realpart(pdf(dd, Dual(1, 0)))
