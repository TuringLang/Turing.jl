using Distributions
using Turing
using Stan

include(Pkg.dir("Turing")*"/benchmarks/benchmarkhelper.jl")

include(Pkg.dir("Turing")*"/benchmarks/naive.bayes.data.jl")
include(Pkg.dir("Turing")*"/benchmarks/naive.bayes.model.jl")

bench_res = tbenchmark("HMCDA(1000, 0.95, 0.3)", "nbmodel", "K, V, M, N, z, w, alpha, β")
bench_res[4].names = ["phi[1]", "phi[2]", "phi[3]", "phi[4]"]
logd = build_logd("Naive Bayes", bench_res...)

include("naive.bayes-stan.run.jl")
logd["stan"] = stan_d
logd["time_stan"] = nb_time

print_log(logd)
