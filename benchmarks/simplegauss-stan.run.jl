include("simplegauss-stan.data.jl")
include("simplegauss-stan.model.jl")

simplegaussstan = Stanmodel(name="simplegauss", model=simplegaussstanmodel, nchains=1);

simple_gauss_stan_sim = stan(simplegaussstan, simplegaussstandata, CmdStanDir=CMDSTAN_HOME, summary=false);

s_stan = simple_gauss_stan_sim[1:1000, ["s"], :].value[:]
m_stan = simple_gauss_stan_sim[1:1000, ["m"], :].value[:]
