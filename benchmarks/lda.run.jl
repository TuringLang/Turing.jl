include("lda.data.jl")
include("lda.model.jl")

bench_res = tbenchmark("HMCDA(25, 0.65, 1.5)", "ldamodel", "K, V, M, N, w, doc, alpha, β")
bench_res[4].names = ["phi[1]", "phi[2]"]
logd = build_logd("LDA", bench_res...)

include("lda-stan.run.jl")
logd["stan"] = lda_stan_d
logd["time_stan"] = lda_time

print_log(logd)
