
module Turing

# Code associated with running probabilistic programs as tasks
include("trace/trace.jl")

using Distributions
using ForwardDiff: Dual
using Turing.Traces

# Turing essentials - modelling macros and inference algorithms
export @model, @assume, @observe, @predict, InferenceAlgorithm, HMC, IS, SMC, PG, sample

# Turing-safe data structures and associated functions
export TArray, tzeros, localcopy, IArray

# Debugging helpers
export dprintln

## global data structures
const TURING = Dict{Symbol, Any}()
global sampler = nothing
global debug_level = 0

# debugging print function: The first argument controls the verbosity of message,
#  e.g. larger v leads to more verbose debugging messages.
dprintln(v, args...) = v < Turing.debug_level ? println(args...) : nothing

# Inference code
include("distributions/distributions.jl")
include("distributions/ddistributions.jl")
include("core/util.jl")
include("core/compiler.jl")
include("core/intrinsic.jl")
include("core/conditional.jl")
include("core/container.jl")
include("core/io.jl")
include("core/IArray.jl")
include("samplers/sampler.jl")
include("distributions/bnp.jl")

end
