module Inference

using ..Core, ..Core.RandomVariables, ..Utilities
using Distributions, Libtask, Bijectors
using ProgressMeter, LinearAlgebra
using ..Turing: PROGRESS, CACHERESET, AbstractRunner
using ..Turing: Model, runmodel!, get_pvars, get_dvars,
    Sampler, Selector, SampleFromDistribution
using ..Turing: in_pvars, in_dvars, Turing
using StatsFuns: logsumexp

import Distributions: sample, logpdf
import ..Core: getchunksize, getADtype
import ..Utilities: Sample, save, resume

export  InferenceAlgorithm,
        Hamiltonian,
        AbstractGibbs,
        GibbsComponent,
        StaticHamiltonian,
        AdaptiveHamiltonian,
        SampleFromUniform,
        SampleFromPrior,
        ComputeLogJointDensity,
        ComputeLogDensity,
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
        reset!,
        logjoint,
        logpdf

###########
# Runners #
###########
struct SampleFromPrior <: SampleFromDistribution end
_rand(::SampleFromPrior, dist::Distribution) = rand(dist)
_rand(::SampleFromPrior, dist::Distribution, n::Int) = rand(dist, n)

struct SampleFromUniform <: SampleFromDistribution end
_rand(::SampleFromUniform, dist::Distribution) = init(dist)
_rand(::SampleFromUniform, dist::Distribution, n::Int) = init(dist, n)

struct ComputeLogJointDensity <: AbstractRunner end

@inline Sampler(alg::ComputeLogJointDensity) = Sampler(alg, Selector())
@inline Sampler(alg::ComputeLogJointDensity, s::Selector) = Sampler(alg, Dict{Symbol,Any}(), s)

struct ComputeLogDensity <: AbstractRunner end

@inline Sampler(alg::ComputeLogDensity) = Sampler(alg, Selector())
@inline Sampler(alg::ComputeLogDensity, s::Selector) = Sampler(alg, Dict{Symbol,Any}(), s)

struct ParticleFiltering <: AbstractRunner end

@inline Sampler(alg::ParticleFiltering) = Sampler(alg, Selector())
@inline Sampler(alg::ParticleFiltering, s::Selector) = Sampler(alg, Dict{Symbol,Any}(), s)

#####################
# Utility Functions #
#####################

@inline function logjoint(model::Model, vi::AbstractVarInfo; selector::Selector=Selector())
    runmodel!(model, vi, Sampler(ComputeLogJointDensity(), s))
    return vi.logp
end

@inline function logpdf(model::Model, vi::AbstractVarInfo; selector::Selector=Selector())
    runmodel!(model, vi, Sampler(ComputeLogDensity(), s))
    return vi.logp
end

#######################
# Sampler abstraction #
#######################
abstract type AbstractAdapter end
abstract type InferenceAlgorithm end
abstract type Hamiltonian{AD} <: InferenceAlgorithm end
abstract type StaticHamiltonian{AD} <: Hamiltonian{AD} end
abstract type AdaptiveHamiltonian{AD} <: Hamiltonian{AD} end

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
#  struct Sampler{TH<:Hamiltonian,TA<:AbstractAdapter} <: AbstractRunner
#    alg   :: TH
#    state :: HMCState
#    adapt :: TA
#  end

# Helper functions
include("adapt/adapt.jl")
include("support/hmc_core.jl")

include("runners.jl")

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

## Fallback functions

# utility funcs for querying sampler information
require_gradient(spl::Sampler) = false
require_particles(spl::Sampler) = false

# ############################# #
# Assume and observe functions. #
# ############################# #

# Those functions have to be implemented for each new sampler.

function assume(spl::Sampler, dist::Distribution, vn::VarName, vi::VarInfo)
    @error "[assume]: Unmanaged inference algorithm: $(typeof(spl))"
end

function assume(spl::Sampler, dists::Vector{<:Distribution}, vn::VarName, var, vi::VarInfo)
    @error "[assume]: Unmanaged inference algorithm: $(typeof(spl))"
end

function observe(spl::Sampler, dist::Distribution, value, vi::VarInfo)
    @error "[observe]: Unmanaged inference algorithm: $(typeof(spl))"
end

function observe(spl::Sampler, dists::Vector{<:Distribution}, values, vi::VarInfo)
    @error "[observe]: Unmanaged inference algorithm: $(typeof(spl))"
end

##############
# Utilities  #
##############

# VarInfo to Sample
function Sample(vi::UntypedVarInfo)
    value = Dict{Symbol, Any}() # value is named here because of Sample has a field called value
    for vn in keys(vi)
        value[RandomVariables.sym_idx(vn)] = vi[vn]
    end
    # NOTE: do we need to check if lp is 0?
    value[:lp] = getlogp(vi)
    return Sample(0.0, value)
end

# VarInfo, combined with spl.info, to Sample
function Sample(vi::AbstractVarInfo, spl::Sampler)
    s = Sample(vi)
    if haskey(spl.info, :wum)
        s.value[:epsilon] = getss(spl.info[:wum])
    end
    if haskey(spl.info, :lf_num)
        s.value[:lf_num] = spl.info[:lf_num]
    end
    if haskey(spl.info, :eval_num)
        s.value[:eval_num] = spl.info[:eval_num]
    end
    return s
end

end # module
