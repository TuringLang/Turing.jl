using Turing
using Distributions
using Base.Test

include("data.jl")
include("model.jl")

chain = @sample(gaussmodel(gaussdata), PG(20, 300))

sim1 = TuringChains(chain)
describe(sim1)
