include("simplegauss.data.jl")
include("simplegauss.model.jl")

bench_res = tbenchmark("Gibbs(1000, PG(100, 3, :s), HMC(2, 0.1, 3, :m))", "simplegaussmodel", "simplegaussdata")
logd = build_logd("Simple Gaussian Model", bench_res...)
logd["analytic"] = Dict("s" => 49/24, "m" => 7/6)

include("simplegauss-stan.run.jl")

logd["stan"] = Dict("s" => mean(s_stan), "m" => mean(m_stan))
logd["time_stan"] = sg_time

print_log(logd)
