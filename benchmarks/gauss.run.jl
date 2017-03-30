using Turing
using Distributions
using Base.Test

include("gauss.data.jl")
include("gauss.model.jl")

sim1 = @sample(gaussmodel(gaussdata), PG(20, 2000))
describe(sim1)

sim2 = @sample(gaussmodel(gaussdata), HMC(2000, 0.25, 5))
describe(sim2)

sim3 = @sample(gaussmodel(gaussdata), HMC(2000, 0.25, 5))
describe(sim3)

println("Numerical test for Gibbs")
println("  1. s ≈ 49/24 ? $(abs(mean(sim3[:s]) - 49/24) <= 0.15)")
println("  2. s ≈ 7/6 ? $(abs(mean(sim3[:m]) 7/6) <= 0.15)") 
