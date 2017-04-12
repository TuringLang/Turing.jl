include("simplegauss.data.jl")
include("simplegauss.model.jl")

logd = build_logd("Simple Gaussian Model", tbenchmark("Gibbs(1000, PG(100, 3, :s), HMC(2, 0.1, 3, :m))", "simplegaussmodel", "simplegaussdata")...)
logd["analytic"] = Dict("s" => 49/24, "m" => 7/6)
logd["stan"] = Dict("s" => mean(s_stan), "m" => mean(m_stan))

print_log(logd)
