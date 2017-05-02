using Turing, Stan, Distributions

include("benchmarkhelper.jl")
include("lda-stan.data.jl")
include("lda.model.jl")

bench_res = tbenchmark("HMCDA(1000, 0.65, 1.5)", "ldamodel", "data=ldastandata[1]")
bench_res[4].names = ["phi[1]", "phi[2]"]
logd = build_logd("LDA", bench_res...)

include("lda-stan.run.jl")
logd["stan"] = lda_stan_d
logd["time_stan"] = lda_time

print_log(logd)
