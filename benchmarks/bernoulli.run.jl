include("bernoulli.data.jl")
include("bernoulli.model.jl")

bench_res = tbenchmark("HMC(1000, 0.25, 5)", "bermodel", "berdata")
logd = build_logd("Bernoulli Model", bench_res...)
logd["stan"] = Dict("theta" => mean(theta_stan))
logd["time_stan"] = ber_time

print_log(logd)
