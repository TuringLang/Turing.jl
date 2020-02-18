using Turing, TuringBenchmarks.TuringTools
using LinearAlgebra

include("lr_helper.jl")

X, Y = readlrdata()

@model lr_nuts(x, y, σ) = begin

    N,D = size(x)

    α ~ Normal(0, σ)
    β ~ MvNormal(zeros(D), ones(D)*σ)

    for n = 1:N
        y[n] ~ BinomialLogit(1, dot(x[n,:], β) + α)
    end
end

# Sampling parameter settings
n_adapts = 1_000

# Sampling
@tbenchmark chain = sample(lr_nuts(x, y, 100), NUTS(0.65), n_samples);
