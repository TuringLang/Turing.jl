using Mamba, Stan

include("simplegauss-stan.data.jl")
include("simplegauss-stan.model.jl")

simplegaussstan = Stanmodel(name="simplegauss", model=simplegaussstanmodel, nchains=1);

simple_gauss_stan_sim = stan(simplegaussstan, simplegaussstandata, CmdStanDir=CMDSTAN_HOME)
describe(simple_gauss_stan_sim)

s_stan = simple_gauss_stan_sim[1:1000, ["s"], :].value[:]
m_stan = simple_gauss_stan_sim[1:1000, ["m"], :].value[:]
s_turing = simple_gauss_sim[:s]
m_turing = simple_gauss_sim[:m]

println("The difference between the means from Stan and Turing is:")
println("  E(s_stan) - E(s_turing) = $(mean(s_stan) - mean(s_turing))")
println("  E(m_stan) - E(m_turing) = $(mean(m_stan) - mean(m_turing))")
