using Turing
using Distributions
using Base.Test

include("data.jl")
include("model.jl")

chain = @sample(gaussmodel(gaussdata), SMC(300))
