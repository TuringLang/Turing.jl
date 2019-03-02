module Core

using MacroTools, Libtask, ForwardDiff
using ..Utilities, Reexport
using Flux.Tracker: Tracker
using ..Turing: Turing, Model, Sampler, runmodel!

include("VarReplay.jl")
@reexport using .VarReplay

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
        FluxTrackerAD,
        value,
        gradient_logp,
        CHUNKSIZE, 
        ADBACKEND,
        setchunksize,
        verifygrad,
        gradient_logp_forward,
        gradient_logp_reverse

end # module
