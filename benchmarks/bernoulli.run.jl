include("bernoulli.data.jl")
include("bernoulli.model.jl")

alg_str = "HMC(1000, 0.25, 5)"
alg = eval(parse(alg_str))
ber_sim, time, mem, _, _  = @timed sample(bermodel(berdata), alg)

logd = build_logd("Bernoulli Model", alg_str, time, mem, ber_sim)
logd["stan"] = Dict("theta" => mean(theta_stan))

print_log(logd)
