module Utilities

using DynamicPPL: AbstractSampler, Sampler
using Distributions, Bijectors
using StatsFuns, SpecialFunctions
using MCMCChains: Chains, setinfo
import Distributions: sample

export  vectorize,
        reconstruct,
        reconstruct!,
        Sample,
        Chain,
        init,
        vectorize,
        set_resume!,
        FlattenIterator

include("robustinit.jl")
include("helper.jl")

end # module
