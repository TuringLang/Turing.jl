using Mamba, Stan

include("simple-normal-mixture-stan.model.jl")
include("simple-normal-mixture-stan.data.jl")

simplenormalmixturestan = Stanmodel(name="normalmixture", model=simplenormalmixturemodel, nchains=1);

sim1 = stan(simplenormalmixturestan, simplenormalmixturedata, CmdStanDir=CMDSTAN_HOME)
describe(sim1)
