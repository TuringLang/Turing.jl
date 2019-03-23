using Test
using Turing: BinomialLogit
using Distributions: Binomial, logpdf
using StatsFuns: logistic

ns = 10
logitp = randn()
d1 = BinomialLogit(ns, logitp)
d2 = Binomial(ns, logistic(logitp))
k = 3
@test logpdf(d1, k) ≈ logpdf(d2, k)
