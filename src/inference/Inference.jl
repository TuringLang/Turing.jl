module Inference

using ..Core, ..Core.RandomVariables, ..Utilities
using ..Core.RandomVariables: Metadata, _tail
using Distributions, Libtask, Bijectors
using ProgressMeter, LinearAlgebra
using ..Turing: PROGRESS, CACHERESET, AbstractSampler
using ..Turing: Model, runmodel!, get_pvars, get_dvars,
    Sampler, SampleFromPrior, SampleFromUniform,
    Selector
using ..Turing: in_pvars, in_dvars, Turing
using StatsFuns: logsumexp
using Random: GLOBAL_RNG, AbstractRNG
import AdvancedHMC; const AHMC = AdvancedHMC

import ..Turing: getspace
import Distributions: sample
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

"""
    mh_accept(H::T, H_new::T, log_proposal_ratio::T) where {T<:Real}

Peform MH accept criteria with log acceptance ratio. Returns a `Bool` for acceptance.

Note: This function is only used in PMMH.
"""
function mh_accept(H::T, H_new::T, log_proposal_ratio::T) where {T<:Real}
    return log(rand()) + H_new < H + log_proposal_ratio, min(0, -(H_new - H))
end

# Concrete algorithm implementations.
include("sghmc.jl")
include("hmc.jl")
include("mh.jl")
include("is.jl")
include("AdvancedSMC.jl")
include("gibbs.jl")

getspace(::IS) = ()
getspace(::Type{<:IS}) = ()
for alg in (:SMC, :PG, :PMMH, :IPMCMC, :MH)
    @eval getspace(::$alg{space}) where {space} = space
    @eval getspace(::Type{<:$alg{space}}) where {space} = space
end
for alg in (:HMC, :HMCDA, :NUTS, :SGLD, :SGHMC)
    @eval getspace(::$alg{<:Any, space}) where {space} = space
    @eval getspace(::Type{<:$alg{<:Any, space}}) where {space} = space
end

@inline floatof(::Type{T}) where {T <: Real} = typeof(one(T)/one(T))
@inline floatof(::Type) = Real

@inline Turing.Core.get_matching_type(spl::Turing.Sampler, vi::Turing.RandomVariables.VarInfo, ::Type{T}) where {T <: AbstractFloat} = floatof(eltype(vi, spl))
@inline Turing.Core.get_matching_type(spl::Turing.Sampler{<:Hamiltonian}, vi::Turing.RandomVariables.VarInfo, ::Type{TV}) where {T, N, TV <: Array{T, N}} = Array{Turing.Core.get_matching_type(spl, vi, T), N}
@inline Turing.Core.get_matching_type(spl::Turing.Sampler{<:Union{PG, SMC}}, vi::Turing.RandomVariables.VarInfo, ::Type{TV}) where {T, N, TV <: Array{T, N}} = TArray{T, N}

## Fallback functions

# utility funcs for querying sampler information
require_gradient(spl::Sampler) = false
require_particles(spl::Sampler) = false

assume(spl::Sampler, dist::Distribution) =
error("Turing.assume: unmanaged inference algorithm: $(typeof(spl))")

observe(spl::Sampler, weight::Float64) =
error("Turing.observe: unmanaged inference algorithm: $(typeof(spl))")

## Default definitions for assume, observe, when sampler = nothing.
function assume(spl::A,
    dist::Distribution,
    vn::VarName,
    vi::VarInfo) where {A<:Union{SampleFromPrior, SampleFromUniform}}

    if haskey(vi, vn)
        r = vi[vn]
    else
        r = isa(spl, SampleFromUniform) ? init(dist) : rand(dist)
        push!(vi, vn, r, dist, spl)
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
    vi::VarInfo) where {T<:Distribution, A<:Union{SampleFromPrior, SampleFromUniform}}

    @assert length(dists) == 1 "Turing.assume only support vectorizing i.i.d distribution"
    dist = dists[1]
    n = size(var)[end]

    vns = map(i -> VarName(vn, "[$i]"), 1:n)

    if haskey(vi, vns[1])
        rs = vi[vns]
    else
        rs = isa(spl, SampleFromUniform) ? init(dist, n) : rand(dist, n)

        if isa(dist, UnivariateDistribution) || isa(dist, MatrixDistribution)
            for i = 1:n
                push!(vi, vns[i], rs[i], dist, spl)
            end
            @assert size(var) == size(rs) "Turing.assume: variable and random number dimension unmatched"
            var = rs
        elseif isa(dist, MultivariateDistribution)
            for i = 1:n
                push!(vi, vns[i], rs[:,i], dist, spl)
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


observe(::Nothing,
        dist::T,
        value::Any,
        vi::VarInfo) where T = observe(SampleFromPrior(), dist, value, vi)

function observe(spl::A,
    dist::Distribution,
    value::Any,
    vi::VarInfo) where {A<:Union{SampleFromPrior, SampleFromUniform}}

    vi.num_produce += one(vi.num_produce)
    Turing.DEBUG && @debug "dist = $dist"
    Turing.DEBUG && @debug "value = $value"

    # acclogp!(vi, logpdf(dist, value))
    logpdf(dist, value)
end

function observe(spl::A,
    dists::Vector{T},
    value::Any,
    vi::VarInfo) where {T<:Distribution, A<:Union{SampleFromPrior, SampleFromUniform}}

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
Sample(vi::VarInfo) = Sample(0.0, todict(vi))
function todict(vi::VarInfo)
    value = todict(vi.metadata, vi)
    value[:lp] = getlogp(vi)
    return value
end
function todict(md::Metadata, vi::VarInfo)
    value = Dict{Symbol, Any}() # value is named here because of Sample has a field called value
    for vn in keys(md.idcs)
        value[Symbol(vn)] = vi[vn]
    end
    return value
end
function todict(metadata::NamedTuple{names}, vi::VarInfo) where {names}
    length(names) === 0 && return Dict{Symbol, Any}()
    f = names[1]
    mdf = getfield(metadata, f)
    return merge(todict(mdf, vi), todict(_tail(metadata), vi))
end

# VarInfo, combined with spl.info, to Sample
function Sample(vi::AbstractVarInfo, spl::Sampler)
    s = Sample(vi)
    if haskey(spl.info, :eval_num)
        s.value[:eval_num] = spl.info[:eval_num]
    end
    return s
end

getspace(spl::Sampler) = getspace(typeof(spl))
getspace(::Type{<:Sampler{Talg}}) where {Talg} = getspace(Talg)

end # module
