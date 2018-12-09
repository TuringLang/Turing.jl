using Test
using Turing: BinomialLogit
using Distributions: Binomial, logpdf
using StatsFuns: logistic

n = 10
logitp = randn()
d1 = BinomialLogit(n, logitp)
d2 = Binomial(n, logistic(logitp))
k = 3
@test logpdf(d1, k) ≈ logpdf(d2, k)
