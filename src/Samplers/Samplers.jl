module Samplers

import ..Turing
using ..Turing: DEFAULT_ADAPT_CONF_TYPE, STAN_DEFAULT_ADAPT_CONF, Model
using Requires
using Parameters: @unpack
using ..Utilities

export  InferenceAlgorithm,
        Hamiltonian,
        AbstractGibbs,
        GibbsComponent,
        StaticHamiltonian,
        AdaptiveHamiltonian,
        AbstractSampler,
        Sampler,
        HamiltonianRobustInit,
        SampleFromPrior,
        AnySampler,
        MH, 
        Gibbs,      # classic sampling
        HMC, 
        SGLD, 
        SGHMC, 
        HMCDA, 
        NUTS,       # Hamiltonian-like sampling
        DynamicNUTS,
        IS, 
        SMC, 
        CSMC, 
        PG, 
        PIMH, 
        PMMH, 
        IPMCMC,  # particle-based sampling
        getspace

#######################
# Sampler abstraction #
#######################
abstract type InferenceAlgorithm end
abstract type Hamiltonian{AD} <: InferenceAlgorithm end
abstract type AbstractGibbs <: InferenceAlgorithm end
abstract type StaticHamiltonian{AD} <: Hamiltonian{AD} end
abstract type AdaptiveHamiltonian{AD} <: Hamiltonian{AD} end
abstract type AbstractSampler end

"""
    Sampler{T}

Generic interface for implementing inference algorithms.
An implementation of an algorithm should include the following:

1. A type specifying the algorithm and its parameters, derived from InferenceAlgorithm
2. A method of `sample` function that produces results of inference, which is where actual inference happens.

Turing translates models to chunks that call the modelling functions at specified points. The dispatch is based on the value of a `sampler` variable. To include a new inference algorithm implements the requirements mentioned above in a separate file,
then include that file at the end of this one.
"""
mutable struct Sampler{T<:InferenceAlgorithm} <: AbstractSampler
    alg   ::  T
    info  ::  Dict{Symbol, Any}         # sampler infomation
end
Sampler(alg, model::Model) = Sampler(alg)
function Sampler(alg::A, info::Dict) where {A <: InferenceAlgorithm}
    Sampler{A}(alg, info)
end

# Helper functions
include("util.jl")

# Concrete algorithm implementations.
include("hmcda.jl")
include("nuts.jl")
include("sghmc.jl")
include("sgld.jl")
include("hmc.jl")
if isdefined(Turing, :DynamicHMC)
    include("dynamichmc.jl")
    getspace(::Type{<:DynamicNUTS{space}}) where space = space
end
include("mh.jl")
include("is.jl")
include("smc.jl")
include("pgibbs.jl")
include("pmmh.jl")
include("ipmcmc.jl")
include("gibbs.jl")

getspace(alg::InferenceAlgorithm) = getspace(typeof(alg))
for A in (:IPMCMC, :IS, :MH, :PMMH, :SMC, :Gibbs, :PG)
    @eval getspace(::Type{<:$A{space}}) where space = space
end
for A in (:HMC, :SGHMC, :SGLD, :HMCDA, :NUTS)
    @eval getspace(::Type{<:$A{<:Any, space}}) where space = space
end
getspace(spl::Sampler) = getspace(typeof(spl))
getspace(::Type{<:Sampler{A}}) where A = getspace(A)

"""
Robust initialization method for model parameters in Hamiltonian samplers.
"""
struct HamiltonianRobustInit <: AbstractSampler end
struct SampleFromPrior <: AbstractSampler end

# This can be removed when all `spl=nothing` is replaced with
#   `spl=SampleFromPrior`
const AnySampler = Union{Nothing, AbstractSampler}

end # module
