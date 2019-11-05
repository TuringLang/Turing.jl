module Utilities

using ..Turing: AbstractSampler, Sampler
using Distributions, Bijectors
using StatsFuns, SpecialFunctions
using MCMCChains: AbstractChains, Chains, setinfo
import Distributions: sample

export  vectorize,
        reconstruct,
        reconstruct!,
        Sample,
        Chain,
        init,
        vectorize,
        set_resume!,
        namedtuple,
        ntmerge

include("helper.jl")
include("robustinit.jl")

end # module
