module Utilities

using ..Turing
using Distributions
using Bijectors
using MCMCChain: AbstractChains, Chains
import Distributions: sample

export  resample,
        randcat,
        resample_multinomial,
        resample_residual,
        resample_stratified,
        resample_systematic,
        vectorize,
        reconstruct,
        reconstruct!,
        Sample, 
        Chain,
        init,
        vectorize

@static if isdefined(Turing, :CmdStan)
    include("stan-interface.jl")
end
include("helper.jl")
include("robustinit.jl")
include("util.jl")         # utility functions
include("io.jl")           # I/O
include("resample.jl")
include("distributions.jl")

end # module
