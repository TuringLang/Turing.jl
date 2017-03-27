using Distributions
using Turing
using Base.Test

include("naive.bayes.data.jl")
include("naive.bayes.jl")

sim1 = sample(nbmodel, nbdata, HMC(300, 0.1, 3))
describe(sim1)
