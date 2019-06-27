using Turing, TuringBenchmarks

@model gdemo(x, y) = begin
    s ~ InverseGamma(2,3)
    m ~ Normal(0,sqrt(s))
    x ~ Normal(m, sqrt(s))
    y ~ Normal(m, sqrt(s))
    return s, m
end

data = (1.5, 2.0)

bench_res = @tbenchmark(Turing.NUTS(2000000, 0.65), gdemo, data...)

logd = build_logd("GDemo-Benchmark", bench_res...)

print_log(logd)
send_log(logd)
