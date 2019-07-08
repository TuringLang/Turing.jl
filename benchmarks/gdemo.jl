using Turing, TuringBenchmarks.TuringTools

@model gdemo(x, y) = begin
    s ~ InverseGamma(2,3)
    m ~ Normal(0,sqrt(s))
    x ~ Normal(m, sqrt(s))
    y ~ Normal(m, sqrt(s))
    return s, m
end

data = (1.5, 2.0)

# sample(gdemo(1.5, 2.0), Turing.NUTS(2000000, 0.65));
bench_res = @tbenchmark(Turing.NUTS(2000000, 0.65), gdemo, data...)

LOG_DATA = build_log_data("GDemo-Benchmark", bench_res...)
print_log(LOG_DATA)
