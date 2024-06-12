using Turing
using DelimitedFiles

if !haskey(BenchmarkSuite, "nuts")
    BenchmarkSuite["nuts"] = BenchmarkGroup(["nuts"])
end

fname = joinpath(dirname(@__FILE__), "sv_nuts.data")
y, header = readdlm(fname, ','; header=true)

# Stochastic volatility (SV)
@model function sv_nuts(y, dy, ::Type{T}=Vector{Float64}) where {T}
    N = size(y, 1)

    τ ~ Exponential(1 / 100)
    ν ~ Exponential(1 / 100)
    s = T(undef, N)

    s[1] ~ Exponential(1 / 100)
    for n in 2:N
        s[n] ~ Normal(log(s[n - 1]), τ)
        s[n] = exp(s[n])
        dy = log(y[n] / y[n - 1]) / s[n]
        dy ~ TDist(ν)
    end
end

# Sampling parameter settings
n_samples = 10_000

# Sampling
BenchmarkSuite["nuts"]["sv"] = @benchmarkable sample(sv_nuts(y, NaN), NUTS(0.65), n_samples)
