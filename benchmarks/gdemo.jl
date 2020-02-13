using Turing, BenchmarkTools, BenchmarkHelper

# TODO: only definite `suite` once for all benchmarks
suite = BenchmarkGroup()

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

suite["gdemo"] = @benchmarkable sample(gdemo(1.5, 2.0), HMC(0.01, 2), 2000)

# Execute `run` from a benchmarking manager, e.g. Nanosoldier. 
# run(suite)
