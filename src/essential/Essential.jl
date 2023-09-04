module Essential

using DistributionsAD, Bijectors
using Libtask, ForwardDiff, Random
using Distributions, LinearAlgebra
using Reexport
using ..Turing: Turing
using DynamicPPL: Model, AbstractSampler, Sampler, SampleFromPrior
using LinearAlgebra: copytri!
using Bijectors: PDMatDistribution
using AdvancedVI
using StatsFuns: logsumexp, softmax
@reexport using DynamicPPL

import AdvancedPS
import LogDensityProblems
import LogDensityProblemsAD

include("container.jl")
include("ad.jl")

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
        ZygoteAD,
        ReverseDiffAD,
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
