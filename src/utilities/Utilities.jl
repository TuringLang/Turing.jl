module Utilities

using DynamicPPL: AbstractSampler, Sampler
using DynamicPPL: init, inittrans, reconstruct, reconstruct!, vectorize
using Distributions, Bijectors
using StatsFuns, SpecialFunctions
import Distributions: sample

export  vectorize,
        reconstruct,
        reconstruct!,
        Sample,
        Chain,
        init,
        set_resume!,
        FlattenIterator

include("helper.jl")

end # module
