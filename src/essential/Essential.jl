module Essential

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
using AdvancedVI
using StatsFuns: logsumexp, softmax
@reexport using DynamicPPL
using ADTypes: ADTypes, AutoTracker, AutoReverseDiff, AutoZygote

import AdvancedPS
import LogDensityProblems
import LogDensityProblemsAD

include("container.jl")
include("ad.jl")

Base.@deprecate_binding ForwardDiffAD AutoForwardDiff
Base.@deprecate_binding TrackerAD AutoTracker
Base.@deprecate_binding ReverseDiffAD AutoReverseDiff
Base.@deprecate_binding ZygoteAD AutoZygote

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
        AutoForwardDiff,
        AutoTracker,
        AutoZygote,
        AutoReverseDiff,
        value,
        CHUNKSIZE,
        ADBACKEND,
        setchunksize,
        setrdcache,
        getrdcache,
        verifygrad,
        @logprob_str,
        @prob_str

end # module
