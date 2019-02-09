module Inference

using ..Core, ..Core.VarReplay, ..Utilities
using Distributions, Libtask, Bijectors
using ProgressMeter, LinearAlgebra
using ..Turing: PROGRESS, CACHERESET, AbstractSampler
using ..Turing: Sampler, Model, runmodel!, get_pvars, get_dvars
using ..Turing: in_pvars, in_dvars
using StatsFuns: logsumexp

import Distributions: sample
import ..Core: getchunksize, getADtype
import ..Utilities: Sample

export  InferenceAlgorithm,
        Hamiltonian,
        AbstractGibbs,
        GibbsComponent,
        StaticHamiltonian,
        AdaptiveHamiltonian,
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
abstract type Hamiltonian{AD} <: InferenceAlgorithm end
abstract type StaticHamiltonian{AD} <: Hamiltonian{AD} end
abstract type AdaptiveHamiltonian{AD} <: Hamiltonian{AD} end

getchunksize(::T) where {T <: Hamiltonian} = getchunksize(T)
getchunksize(::Type{<:Hamiltonian{AD}}) where AD = getchunksize(AD)
getADtype(alg::Hamiltonian) = getADtype(typeof(alg))
getADtype(::Type{<:Hamiltonian{AD}}) where {AD} = AD

# Sampler stats
abstract type AbstractSamplerStats end
struct NullStats <: AbstractSamplerStats end
struct MHStats <: AbstractSamplerStats
    is_accept       ::  Bool
end
struct HMCStats <: AbstractSamplerStats
    accept_ratio    ::  Float64
    is_accept       ::  Bool
    lf_step_size    ::  Float64
    n_lf_steps      ::  Integer
end
struct SGLDStats <: AbstractSamplerStats
    is_accept       ::  Bool
    lf_step_size    ::  Float64
end


"""
Robust initialization method for model parameters in Hamiltonian samplers.
"""
struct HamiltonianRobustInit <: AbstractSampler end
struct SampleFromPrior <: AbstractSampler end

# This can be removed when all `spl=nothing` is replaced with
#   `spl=SampleFromPrior`
const AnySampler = Union{Nothing, AbstractSampler}

# Helper functions
include("adapt/adapt.jl")
include("support/hmc_core.jl")

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

assume(spl::Sampler, dist::Distribution) =
error("Turing.assume: unmanaged inference algorithm: $(typeof(spl))")

observe(spl::Sampler, weight::Float64) =
error("Turing.observe: unmanaged inference algorithm: $(typeof(spl))")

## Default definitions for assume, observe, when sampler = nothing.
##  Note: `A<:Union{Nothing, SampleFromPrior, HamiltonianRobustInit}` can be
##   simplified into `A<:Union{SampleFromPrior, HamiltonianRobustInit}` when
##   all `spl=nothing` is replaced with `spl=SampleFromPrior`.
function assume(spl::A,
    dist::Distribution,
    vn::VarName,
    vi::VarInfo) where {A<:Union{Nothing, SampleFromPrior, HamiltonianRobustInit}}

    if haskey(vi, vn)
        r = vi[vn]
    else
        r = isa(spl, HamiltonianRobustInit) ? init(dist) : rand(dist)
        push!(vi, vn, r, dist, 0)
    end
    # NOTE: The importance weight is not correctly computed here because
    #       r is genereated from some uniform distribution which is different from the prior
    # acclogp!(vi, logpdf_with_trans(dist, r, istrans(vi, vn)))

    r, logpdf_with_trans(dist, r, istrans(vi, vn))

end

function assume(spl::A,
    dists::Vector{T},
    vn::VarName,
    var::Any,
    vi::VarInfo) where {T<:Distribution, A<:Union{Nothing, SampleFromPrior, HamiltonianRobustInit}}

    @assert length(dists) == 1 "Turing.assume only support vectorizing i.i.d distribution"
    dist = dists[1]
    n = size(var)[end]

    vns = map(i -> copybyindex(vn, "[$i]"), 1:n)

    if haskey(vi, vns[1])
        rs = vi[vns]
    else
        rs = isa(spl, HamiltonianRobustInit) ? init(dist, n) : rand(dist, n)

        if isa(dist, UnivariateDistribution) || isa(dist, MatrixDistribution)
            for i = 1:n
                push!(vi, vns[i], rs[i], dist, 0)
            end
            @assert size(var) == size(rs) "Turing.assume: variable and random number dimension unmatched"
            var = rs
        elseif isa(dist, MultivariateDistribution)
            for i = 1:n
                push!(vi, vns[i], rs[:,i], dist, 0)
            end
            if isa(var, Vector)
                @assert length(var) == size(rs)[2] "Turing.assume: variable and random number dimension unmatched"
                for i = 1:n
                    var[i] = rs[:,i]
                end
            elseif isa(var, Matrix)
                @assert size(var) == size(rs) "Turing.assume: variable and random number dimension unmatched"
                var = rs
            else
                @error("Turing.assume: unsupported variable container"); error()
            end
        end
    end

    # acclogp!(vi, sum(logpdf_with_trans(dist, rs, istrans(vi, vns[1]))))

    var, sum(logpdf_with_trans(dist, rs, istrans(vi, vns[1])))

end

function observe(spl::A,
    dist::Distribution,
    value::Any,
    vi::VarInfo) where {A<:Union{Nothing, SampleFromPrior, HamiltonianRobustInit}}

    vi.num_produce += one(vi.num_produce)
    @debug "dist = $dist"
    @debug "value = $value"

    # acclogp!(vi, logpdf(dist, value))
    logpdf(dist, value)

end

function observe(spl::A,
    dists::Vector{T},
    value::Any,
    vi::VarInfo) where {T<:Distribution, A<:Union{Nothing, SampleFromPrior, HamiltonianRobustInit}}

    @assert length(dists) == 1 "Turing.observe only support vectorizing i.i.d distribution"
    dist = dists[1]
    @assert isa(dist, UnivariateDistribution) || isa(dist, MultivariateDistribution) "Turing.observe: vectorizing matrix distribution is not supported"
    if isa(dist, UnivariateDistribution)  # only univariate distributions support broadcast operation (logpdf.) by Distributions.jl
        # acclogp!(vi, sum(logpdf.(Ref(dist), value)))
        sum(logpdf.(Ref(dist), value))
    else
        # acclogp!(vi, sum(logpdf(dist, value)))
        sum(logpdf(dist, value))
    end

end


##############
# Utilities  #
##############

# VarInfo to Sample
@inline function Sample(vi::VarInfo)
    value = Dict{Symbol, Any}() # value is named here because of Sample has a field called value
    for vn in keys(vi)
        value[sym(vn)] = vi[vn]
    end
    # NOTE: do we need to check if lp is 0?
    value[:_lp] = getlogp(vi)
    if ~isempty(vi.pred)
        for sym in keys(vi.pred)
        # if ~haskey(sample.value, sym)
            value[sym] = vi.pred[sym]
        # end
        end
        # TODO: check why 1. 2. cause errors
        # TODO: which one is faster?
        # 1. Using empty!
        # empty!(vi.pred)
        # 2. Reassign an enmtpy dict
        # vi.pred = Dict{Symbol,Any}()
        # 3. Do nothing?
    end
    return Sample(0.0, value)
end

# VarInfo, combined with spl.info, to Sample
@inline function Sample(vi::VarInfo, spl::Sampler; elapsed=nothing)
    s = Sample(vi)
    if !(elapsed == nothing)
        s.value[:_elapsed] = elapsed
    end
    if haskey(spl.info, :wum)
        s.value[:_epsilon] = getss(spl.info[:wum])
    end
    if haskey(spl.info, :lf_num)
        s.value[:_lf_num] = spl.info[:lf_num]
    end
    if haskey(spl.info, :eval_num)
        s.value[:_eval_num] = spl.info[:eval_num]
    end
    return s
end

# VarInfo, combined with SamplingStats, to Sample
@inline function Sample(vi::VarInfo, stats::T; elapsed=nothing) where {T<:AbstractSamplerStats}
    s = Sample(vi)
    if !(elapsed == nothing)
        s.value[:_elapsed] = elapsed
    end
    for fn in fieldnames(T)
        s.value[Symbol("_$fn")] = getfield(stats, fn)
    end
    return s
end

end # module
