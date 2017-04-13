include("lda.data.jl")
include("lda.model.jl")

bench_res = tbenchmark("Gibbs(5, PG(5, 1, :z), HMC(1, 0.1, 1, :theta, :phi))", "ldamodel", "K, V, M, N, w, doc, alpha, β")
bench_res[4].names = ["phi[1]", "phi[2]"]
logd = build_logd("LDA", bench_res...)

logd["stan"] = lda_stan_d
logd["time_stan"] = lda_time

print_log(logd)
