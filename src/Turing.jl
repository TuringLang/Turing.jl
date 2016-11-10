module Turing

# Code associated with running probabilistic programs as tasks
include("trace/trace.jl")

using ForwardDiff: Dual, npartials
using Turing.Traces

# TODO: move this to seperate AD module
Base.convert(::Type{Float64}, d::Dual{0,Float64}) = d.value

###########
# Warning #################################
# The following overloadings is temporary #
# before StatsFuns.jl accepts our PR.     #

using StatsFuns
StatsFuns.betapdf(α::Real, β::Real, x::Real) = x^(α - 1) * (1 - x)^(β - 1) / beta(α, β)
StatsFuns.betalogpdf(α::Real, β::Real, x::Real) = (α - 1) * log(x) + (β - 1) * log(1 - x) - log(beta(α, β))
StatsFuns.gammapdf(k::Real, θ::Real, x::Real) = 1 / (gamma(k) * θ^k) * x^(k - 1) * exp(-x / θ)
StatsFuns.gammalogpdf(k::Real, θ::Real, x::Real) = -log(gamma(k)) - k * log(θ) + (k - 1) * log(x) - x / θ

###########################################

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
include("core/util.jl")
include("core/compiler.jl")
include("core/intrinsic.jl")
include("core/container.jl")
include("core/io.jl")
include("samplers/sampler.jl")

end
