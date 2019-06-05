module Core

using MacroTools, Libtask, ForwardDiff, Random
using Distributions, LinearAlgebra
using ..Utilities, Reexport
using Tracker: Tracker
using ..Turing: Turing, Model, runmodel!,
    AbstractSampler, Sampler, SampleFromPrior

include("RandomVariables.jl")
@reexport using .RandomVariables

include("compiler.jl")
include("container.jl")
include("ad.jl")

export  @model,
        @VarName,
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
        weights,
        effectiveSampleSize,
        increase_logweight,
        inrease_logevidence,
        resample!,
        getsample, 
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
        gradient_logp_reverse

end # module
