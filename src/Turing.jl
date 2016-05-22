
module Turing

# Code associated with running probabilistic programs as tasks
include("trace/trace.jl")

using Distributions
using Turing.Traces

# Turing essentials - modelling macros and inference algorithms
export @model, @assume, @observe, @predict, InferenceAlgorithm, IS, SMC, PG, sample

# Turing-safe data structures and associated functions
export TArray, tzeros, localcopy

# Debugging helpers
export dprintln

# Inference code
include("distributions/distributions.jl")
include("core/util.jl")
include("core/compiler.jl")
include("core/intrinsic.jl")
include("core/conditional.jl")
include("core/container.jl")
include("core/io.jl")
include("samplers/sampler.jl")

## global data structures
const TURING = Dict{Symbol, Any}()
global sampler = nothing
global debug_level = 0


# debugging print function: The first argument controls the verbosity of message,
#  e.g. larger v leads to more verbose debugging messages.
dprintln(v, args...) = v > Turing.debug_level ? println(args...) : nothing

end
