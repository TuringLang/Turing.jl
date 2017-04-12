include("simplegauss.data.jl")
include("simplegauss.model.jl")

alg_str = "Gibbs(1000, PG(100, 3, :s), HMC(2, 0.1, 3, :m))"
alg = eval(parse(alg_str))
simple_gauss_sim, time, mem, _, _ = @timed sample(simplegaussmodel(simplegaussdata), alg)

logd = build_logd("Simple Gaussian Model", alg_str, time, mem, simple_gauss_sim)
logd["analytic"] = Dict("s" => 49/24, "m" => 7/6)
logd["stan"] = Dict("s" => mean(s_stan), "m" => mean(m_stan))

print_log(logd)
