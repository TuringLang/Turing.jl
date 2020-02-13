using Turing, BenchmarkTools, BenchmarkHelper

# TODO: only definite `suite` once for all benchmarks
suite = BenchmarkGroup()

## Dummny benchmarks

suite["dummy"] = BenchmarkGroup(["dummy"])

data = [0, 1, 0, 1, 1, 1, 1, 1, 1, 1]

# log_report("Dummy Benchmark started!")

@model constrained_test(obs) = begin
    p ~ Beta(2,2)
    for i = 1:length(obs)
        obs[i] ~ Bernoulli(p)
    end
    p
end

# log_report("Dummy model constructed!")

#BENCHMARK_RESULT = @benchmark_expr "HMC" sample(constrained_test(data),
#                                                HMC(1.5, 3),
#                                                1000)

suite["dummy"]["dummy"] = @benchmarkable sample(constrained_test($data), HMC(0.01, 2), 2000)

# log_report("Dummy benchmark finished!")

## gdemo

suite["gdemo"] = BenchmarkGroup(["gdemo"])

@model gdemo(x, y) = begin
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    x ~ Normal(m, sqrt(s))
    y ~ Normal(m, sqrt(s))
    return s, m
end

# BENCHMARK_RESULT = @benchmark_expr "NUTS" sample(gdemo(1.5, 2.0),
#                                                 Turing.NUTS(1000, 0.65),
#                                                 6000)

suite["gdemo"]["hmc"] = @benchmarkable sample(gdemo(1.5, 2.0), HMC(0.01, 2), 2000)

# Execute `run` from a benchmarking manager, e.g. Nanosoldier. 
# run(suite)

##
## MvNormal
##

suite["mnormal"] = BenchmarkGroup(["mnormal"])

# Define the target distribution and its gradient
const D = 10

@model target(dim) = begin
   Θ = Vector{Real}(undef, dim)
   θ ~ MvNormal(zeros(D), ones(dim))
end

# Sampling parameter settings
n_samples = 100_000
n_adapts = 2_000

# Sampling
#BENCHMARK_RESULT = @benchmark_expr("NUTS(Leapfrog(...))",
#                                   sample(target(D), HMC(0.1, 5), n_samples));

suite["mnormal"]["hmc"] = @benchmarkable sample(target($D), HMC(0.1, 5), $n_samples)

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
Turing.setadbackend(:forward_diff)
# @benchmark chain = sample(mdemo(d, 1), HMC(0.1, 5), 5000)
suite["mnormal"]["forward_diff"] = @benchmarkable sample(mdemo($d, 1), HMC(0.1, 5), 5000)


#BackwardDiff
Turing.setadbackend(:reverse_diff)
# @benchmark chain = sample(mdemo(d, 1), HMC(0.1, 5), 5000)
suite["mnormal"]["reverse_diff"] = @benchmarkable sample(mdemo($d, 1), HMC(0.1, 5), 5000)
