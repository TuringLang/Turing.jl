using Turing
using LinearAlgebra

include("lr_helper.jl")

if !haskey(BenchmarkSuite, "nuts")
    BenchmarkSuite["nuts"] = BenchmarkGroup(["nuts"])
end

X, Y = readlrdata()

@model function lr_nuts(x, y, σ)

    N,D = size(x)

    α ~ Normal(0, σ)
    β ~ MvNormal(zeros(D), σ^2 * I)

    for n = 1:N
        y[n] ~ BinomialLogit(1, dot(x[n,:], β) + α)
    end
end

# Sampling parameter settings
n_samples = 1_000
n_adapts = 1_000

# Sampling
BenchmarkSuite["nuts"]["lr"] = @benchmarkable sample(lr_nuts(X, Y, 100),
                                                     NUTS(0.65), n_samples)
