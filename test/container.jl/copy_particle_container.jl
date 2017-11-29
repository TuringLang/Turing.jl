using Turing: ParticleContainer, Trace, copy
using Base.Test

pc = ParticleContainer{Trace}(x -> x * x)
newpc = copy(pc)

@test newpc.logE        == pc.logE
@test newpc.logWs       == pc.logWs
@test newpc.conditional == pc.conditional
@test newpc.n_consume   == pc.n_consume
