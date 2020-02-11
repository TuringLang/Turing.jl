using Turing, BenchmarkHelper

@model gdemo(x, y) = begin
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    x ~ Normal(m, sqrt(s))
    y ~ Normal(m, sqrt(s))
    return s, m
end

data = (1.5, 2.0)

BENCHMARK_RESULT = @benchmark_expr "NUTS" sample(gdemo(1.5, 2.0), Turing.NUTS(2000000, 0.65))
