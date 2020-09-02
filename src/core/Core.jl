module Core

using DistributionsAD, Bijectors
using Libtask, ForwardDiff, Random
using Distributions, LinearAlgebra
using ..Utilities, Reexport
using Tracker: Tracker
using ..Turing: Turing
using DynamicPPL: Model, AbstractSampler, Sampler, SampleFromPrior
using LinearAlgebra: copytri!
using Bijectors: PDMatDistribution
import Bijectors: link, invlink
using StatsFuns: logsumexp, softmax
@reexport using DynamicPPL
using Requires

include("container.jl")
include("ad.jl")
function __init__()
    @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
        include("compat/zygote.jl")
        export ZygoteAD
    end
    @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
        include("compat/reversediff.jl")
        export ReverseDiffAD, getrdcache, setrdcache, emptyrdcache
    end
end

export  @model,
        @varname,
        generate_observe,
        translate_tilde!,
        get_vars,
        get_data,
        get_default_values,
        ParticleContainer,
        Particle,
        Trace,
        fork,
        forkr,
        current_trace,
        getweights,
        getweight,
        effectiveSampleSize,
        sweep!,
        ResampleWithESSThreshold,
        ADBackend,
        setadbackend,
        setadsafe,
        ForwardDiffAD,
        TrackerAD,
        value,
        gradient_logp,
        CHUNKSIZE,
        ADBACKEND,
        setchunksize,
        verifygrad,
        @logprob_str,
        @prob_str

end # module
