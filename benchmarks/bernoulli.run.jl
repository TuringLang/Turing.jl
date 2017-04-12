include("bernoulli.data.jl")
include("bernoulli.model.jl")

logd = build_logd("Bernoulli Model", tbenchmark("HMC(1000, 0.25, 5)", "bermodel", "berdata")...)
logd["stan"] = Dict("theta" => mean(theta_stan))

print_log(logd)
