using Distributions
using Turing
using Stan

include(Pkg.dir("Turing")*"/benchmarks/benchmarkhelper.jl")

include(Pkg.dir("Turing")*"/benchmarks/naive.bayes-stan.data.jl")
include(Pkg.dir("Turing")*"/benchmarks/naive.bayes.model.jl")

bench_res = tbenchmark("eNUTS(1000, 0.35)", "nbmodel", "data=nbstandata[1]")
bench_res[4].names = ["phi[1]", "phi[2]", "phi[3]", "phi[4]"]
logd = build_logd("Naive Bayes", bench_res...)

include(Pkg.dir("Turing")*"/benchmarks/naive.bayes-stan.run.jl")
logd["stan"] = stan_d
logd["time_stan"] = nb_time

print_log(logd)
