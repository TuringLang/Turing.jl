using Turing, BenchmarkTools, BenchmarkHelper


## Dummny benchmarks

BenchmarkSuite["dummy"] = BenchmarkGroup(["dummy"])

data = [0, 1, 0, 1, 1, 1, 1, 1, 1, 1]


@model constrained_test(obs) = begin
    p ~ Beta(2,2)
    for i = 1:length(obs)
        obs[i] ~ Bernoulli(p)
    end
    p
end


BenchmarkSuite["dummy"]["dummy"] = @benchmarkable sample(constrained_test($data), HMC(0.01, 2), 2000)


## gdemo

BenchmarkSuite["gdemo"] = BenchmarkGroup(["gdemo"])

@model gdemo(x, y) = begin
    s² ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s²))
    x ~ Normal(m, sqrt(s²))
    y ~ Normal(m, sqrt(s²))
    return s², m
end

BenchmarkSuite["gdemo"]["hmc"] = @benchmarkable sample(gdemo(1.5, 2.0), HMC(0.01, 2), 2000)


##
## MvNormal
##

BenchmarkSuite["mnormal"] = BenchmarkGroup(["mnormal"])

# Define the target distribution and its gradient
const D = 10

@model target(dim) = begin
   Θ = Vector{Real}(undef, dim)
   θ ~ MvNormal(zeros(D), ones(dim))
end

# Sampling parameter settings
n_samples = 100_000
n_adapts = 2_000

BenchmarkSuite["mnormal"]["hmc"] = @benchmarkable sample(target($D), HMC(0.1, 5), $n_samples)

## MvNormal: ForwardDiff vs BackwardDiff (Tracker)

using LinearAlgebra

@model mdemo(d, N) = begin
    Θ = Vector(undef, N)
   for n=1:N
      Θ[n] ~ d
   end
end

dim2 = 250
A    = rand(Wishart(dim2, Matrix{Float64}(I, dim2, dim2)));
d    = MvNormal(zeros(dim2), A)

# ForwardDiff
Turing.setadbackend(:forwarddiff)
BenchmarkSuite["mnormal"]["forwarddiff"] = @benchmarkable sample(mdemo($d, 1), HMC(0.1, 5), 5000)


# BackwardDiff
Turing.setadbackend(:reversediff)
BenchmarkSuite["mnormal"]["reversediff"] = @benchmarkable sample(mdemo($d, 1), HMC(0.1, 5), 5000)
