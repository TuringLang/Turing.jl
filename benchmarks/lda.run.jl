include("lda.data.jl")
include("lda.model.jl")

# bench_res = tbenchmark("HMC(250, 0.1, 3)", "ldamodel", "K, V, M, N, z, w, alpha, β")
# bench_res[4].names = ["phi[1]", "phi[2]"]
# logd = build_logd("Naive Bayes", bench_res...)
#
# logd["stan"] = lda_stan_d
# logd["time_stan"] = lda_time
#
# print_log(logd)
