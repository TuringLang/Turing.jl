using Distributions
using Turing
using Base.Test

include("naive.bayes.data.jl")
include("naive.bayes.model.jl")

sim1 = sample(nbmodel(K, V, M, N, z, w, alpha, β), HMC(250, 0.1, 3))
describe(sim1)
