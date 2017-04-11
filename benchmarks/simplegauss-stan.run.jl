using Mamba, Stan

include("simplegauss-stan.data.jl")
include("simplegauss-stan.model.jl")

simplegaussstan = Stanmodel(name="simplegauss", model=simplegaussmodel, nchains=1);

sim2 = stan(simplegaussstan, simplegaussstandata, CmdStanDir=CMDSTAN_HOME)
describe(sim2)

sim2
