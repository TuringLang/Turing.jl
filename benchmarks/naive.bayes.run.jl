include("naive.bayes.data.jl")
include("naive.bayes.model.jl")

bench_res = tbenchmark("HMC(250, 0.1, 3)", "nbmodel", "K, V, M, N, z, w, alpha, β")
bench_res[4].names = ["phi[1]", "phi[2]", "phi[3]", "phi[4]"]
logd = build_logd("Naive Bayes", bench_res...)

# logd["stan"] = Dict("s" => mean(s_stan), "m" => mean(m_stan))
# logd["time_stan"] = sg_time

print_log(logd)
