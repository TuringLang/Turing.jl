include("gauss.data.jl")
include("gauss.model.jl")

bench_res = tbenchmark("PG(20, 2000)", "gaussmodel", "gaussdata")
logd = build_logd("Gaussian Model", bench_res...)
print_log(logd)

bench_res = tbenchmark("HMC(2000, 0.25, 5)", "gaussmodel", "gaussdata")
logd = build_logd("Gaussian Model", bench_res...)
print_log(logd)

bench_res = tbenchmark("Gibbs(200, HMC(10, 0.25, 5, :mu), PG(20, 10, :lam))", "gaussmodel", "gaussdata")
logd = build_logd("Gaussian Model", bench_res...)
print_log(logd)
