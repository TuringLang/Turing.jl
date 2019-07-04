using Turing, TuringBenchmarks.TuringTools
using StatsFuns: logistic

include("lr_helper.jl")

x, y = readlrdata()

@assert all(yi -> 0 <= yi <= 1, y)

@model lr_nuts(x, y, σ²) = begin

    N,D = size(x)

    w0 ~ Normal(0, sqrt(σ²))
    w ~ MvNormal(zeros(D), ones(D)*sqrt(σ²))

    v = logistic.(x*w .+ w0)

    for n = 1:N
        y[n] ~ Bernoulli(v[n])
    end
end

# Sampling parameter settings
n_samples = 10_000
n_adapts = 1_000

# Sampling
bench_res = @tbenchmark_expr("NUTS(Leapfrog(...))",
                             sample(lr_nuts(x, y, 100), NUTS(n_samples, n_adapts, 0.65)));

LOG_DATA = build_log_data("[NUTS] LogisticRegression-Benchmark", bench_res...)
print_log(LOG_DATA)
