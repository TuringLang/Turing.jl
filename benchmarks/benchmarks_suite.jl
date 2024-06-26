using Turing, BenchmarkTools
using LinearAlgebra

const BenchmarkSuite = BenchmarkTools.BenchmarkGroup()

#
# Add models to benchmarks
#

include("models/hlr.jl")
include("models/lr.jl")
include("models/sv_nuts.jl")

# constrained
BenchmarkSuite["constrained"] = BenchmarkGroup(["constrained"])

data = [0, 1, 0, 1, 1, 1, 1, 1, 1, 1]

@model function constrained_test(obs)
    p ~ Beta(2, 2)
    for i in 1:length(obs)
        obs[i] ~ Bernoulli(p)
    end
    return p
end

BenchmarkSuite["constrained"]["constrained"] = @benchmarkable sample(
    $(constrained_test(data)), $(HMC(0.01, 2)), 2000
)

## gdemo

BenchmarkSuite["gdemo"] = BenchmarkGroup(["gdemo"])

@model function gdemo(x, y)
    s² ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s²))
    x ~ Normal(m, sqrt(s²))
    y ~ Normal(m, sqrt(s²))
    return s², m
end

BenchmarkSuite["gdemo"]["hmc"] = @benchmarkable sample(
    $(gdemo(1.5, 2.0)), $(HMC(0.01, 2)), 2000
)

## MvNormal

BenchmarkSuite["mnormal"] = BenchmarkGroup(["mnormal"])

# Define the target distribution and its gradient

@model function target(dim)
    Θ = Vector{Real}(undef, dim)
    return θ ~ MvNormal(zeros(dim), I)
end

# Sampling parameter settings
dim = 10
n_samples = 100_000
n_adapts = 2_000

BenchmarkSuite["mnormal"]["hmc"] = @benchmarkable sample(
    $(target(dim)), $(HMC(0.1, 5)), $n_samples
)

## MvNormal: ForwardDiff vs ReverseDiff

@model function mdemo(d, N)
    Θ = Vector(undef, N)
    for n in 1:N
        Θ[n] ~ d
    end
end

dim2 = 250
A = rand(Wishart(dim2, Matrix{Float64}(I, dim2, dim2)));
d = MvNormal(zeros(dim2), A)

# ForwardDiff
BenchmarkSuite["mnormal"]["forwarddiff"] = @benchmarkable sample(
    $(mdemo(d, 1)), $(HMC(0.1, 5; adtype=AutoForwardDiff(; chunksize=0))), 5000
)

# ReverseDiff
BenchmarkSuite["mnormal"]["reversediff"] = @benchmarkable sample(
    $(mdemo(d, 1)), $(HMC(0.1, 5; adtype=AutoReverseDiff(; compile=false))), 5000
)
