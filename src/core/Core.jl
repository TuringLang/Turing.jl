module Core

using DistributionsAD, Bijectors
using MacroTools, Libtask, ForwardDiff, Random
using Distributions, LinearAlgebra
using ..Utilities, Reexport
using Tracker: Tracker
using ..Turing: Turing
using DynamicPPL: Model, runmodel!,
    AbstractSampler, Sampler, SampleFromPrior
using LinearAlgebra: copytri!
using Bijectors: PDMatDistribution
import Bijectors: link, invlink
using StatsFuns: logsumexp, softmax
@reexport using DynamicPPL
using Requires

include("container.jl")
include("ad.jl")
@init @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
    include("compat/zygote.jl")
    export ZygoteAD
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
        effectiveSampleSize,
        increase_logweight,
        inrease_logevidence,
        resample!,
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
        gradient_logp_forward,
        gradient_logp_reverse,
        @varinfo,
        @logpdf,
        @sampler,
        @logprob_str,
        @prob_str

end # module
