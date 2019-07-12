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
n_samples = 10_000
n_adapts = 1_000

# Sampling
bench_res = @tbenchmark_expr("NUTS(Leapfrog(...))",
                             sample(lr_nuts(x, y, 100),
                             NUTS(n_samples, n_adapts, 0.65)));

LOG_DATA = build_log_data("[NUTS] LogisticRegression-Benchmark", bench_res...)
print_log(LOG_DATA)
