using Mamba, Stan

include("bernoulli-stan.data.jl")
include("bernoulli-stan.model.jl")

berstan = Stanmodel(name="bernoulli", model=berstanmodel, nchains=1);

sim2 = stan(berstan, berstandata, CmdStanDir=CMDSTAN_HOME)
describe(sim2)
