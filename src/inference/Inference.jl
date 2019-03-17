module Inference

using ..Core, ..Core.VarReplay, ..Utilities
using Distributions, Libtask, Bijectors
using ProgressMeter, LinearAlgebra, Setfield
using ..Turing: PROGRESS, CACHERESET, AbstractSampler, SampleFromPrior
using ..Turing: Model, runmodel!, get_pvars, get_dvars,
    Sampler, SampleFromPrior, SampleFromUniform
using ..Turing: in_pvars, in_dvars, Turing
using StatsFuns: logsumexp
using Parameters: @unpack

import Distributions: sample
import ..Core: getchunksize, getADtype
import ..Turing: getspace
import ..Utilities: Sample

export  InferenceAlgorithm,
        Hamiltonian,
        AbstractGibbs,
        GibbsComponent,
        StaticHamiltonian,
        AdaptiveHamiltonian,
        SampleFromUniform,
        SampleFromPrior,
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
        getspace,
        assume,
        observe,
        step,
        WelfordVar,
        WelfordCovar,
        NaiveCovar,
        get_var,
        get_covar,
        add_sample!,
        reset!

#######################
# Sampler abstraction #
#######################
abstract type AbstractAdapter end
abstract type InferenceAlgorithm end
abstract type AbstractGibbs{space} <: InferenceAlgorithm end
abstract type Hamiltonian{AD, space} <: InferenceAlgorithm end
abstract type StaticHamiltonian{AD, space} <: Hamiltonian{AD, space} end
abstract type AdaptiveHamiltonian{AD, space} <: Hamiltonian{AD, space} end

getchunksize(::T) where {T <: Hamiltonian} = getchunksize(T)
getchunksize(::Type{<:Hamiltonian{AD}}) where AD = getchunksize(AD)
getADtype(alg::Hamiltonian) = getADtype(typeof(alg))
getADtype(::Type{<:Hamiltonian{AD}}) where {AD} = AD

# mutable struct HMCState{T<:Real}
#     epsilon  :: T
#     std     :: Vector{T}
#     lf_num   :: Integer
#     eval_num :: Integer
# end
#
#  struct Sampler{TH<:Hamiltonian,TA<:AbstractAdapter} <: AbstractSampler
#    alg   :: TH
#    state :: HMCState
#    adapt :: TA
#  end

# Helper functions
include("adapt/adapt.jl")
include("support/hmc_core.jl")
include("support/util.jl")

# Concrete algorithm implementations.
include("hmcda.jl")
include("nuts.jl")
include("sghmc.jl")
include("sgld.jl")
include("hmc.jl")
include("mh.jl")
include("is.jl")
include("smc.jl")
include("pgibbs.jl")
include("pmmh.jl")
include("ipmcmc.jl")
include("gibbs.jl")

# utility funcs for querying sampler information
require_gradient(spl::Sampler) = false
require_particles(spl::Sampler) = false

include("sample.jl")
include("fallbacks.jl")

getspace(alg::InferenceAlgorithm) = getspace(typeof(alg))
for A in (:IPMCMC, :IS, :MH, :PMMH, :SMC, :Gibbs, :PG)
    @eval getspace(::Type{<:$A{space}}) where space = space
end
getspace(::Type{<:Hamiltonian{<:Any, space}}) where space = space

end # module
