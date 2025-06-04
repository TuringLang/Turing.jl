module Inference

using ..Essential
using DynamicPPL:
    Metadata,
    VarInfo,
    # TODO(mhauru) all_varnames_grouped_by_symbol isn't exported by DPPL, because it is only
    # implemented for NTVarInfo. It is used by mh.jl. Either refactor mh.jl to not use it
    # or implement it for other VarInfo types and export it from DPPL.
    all_varnames_grouped_by_symbol,
    syms,
    islinked,
    setindex!!,
    push!!,
    setlogp!!,
    getlogp,
    VarName,
    getsym,
    getdist,
    Model,
    Sampler,
    SampleFromPrior,
    SampleFromUniform,
    DefaultContext,
    PriorContext,
    LikelihoodContext,
    set_flag!,
    unset_flag!
using Distributions, Libtask, Bijectors
using DistributionsAD: VectorOfMultivariate
using LinearAlgebra
using ..Turing: PROGRESS, Turing
using StatsFuns: logsumexp
using Random: AbstractRNG
using DynamicPPL
using AbstractMCMC: AbstractModel, AbstractSampler
using DocStringExtensions: FIELDS, TYPEDEF, TYPEDFIELDS
using DataStructures: OrderedSet
using Accessors: Accessors

import ADTypes
import AbstractMCMC
import AdvancedHMC
const AHMC = AdvancedHMC
import AdvancedMH
const AMH = AdvancedMH
import AdvancedPS
import Accessors
import EllipticalSliceSampling
import LogDensityProblems
import Random
import MCMCChains
import StatsBase: predict

export InferenceAlgorithm,
    Hamiltonian,
    StaticHamiltonian,
    AdaptiveHamiltonian,
    SampleFromUniform,
    SampleFromPrior,
    MH,
    ESS,
    Emcee,
    Gibbs,      # classic sampling
    HMC,
    SGLD,
    PolynomialStepsize,
    SGHMC,
    HMCDA,
    NUTS,       # Hamiltonian-like sampling
    IS,
    SMC,
    CSMC,
    PG,
    RepeatSampler,
    Prior,
    assume,
    observe,
    predict,
    externalsampler

###############################################
# Abstract inferface for inference algorithms #
###############################################

include("algorithm.jl")

####################
# Sampler wrappers #
####################

include("repeat_sampler.jl")
include("external_sampler.jl")

# TODO: make a nicer `set_namedtuple!` and move these functions to DynamicPPL.
function DynamicPPL.unflatten(vi::DynamicPPL.NTVarInfo, θ::NamedTuple)
    set_namedtuple!(deepcopy(vi), θ)
    return vi
end
function DynamicPPL.unflatten(vi::SimpleVarInfo, θ::NamedTuple)
    return SimpleVarInfo(θ, vi.logp, vi.transformation)
end

"""
    mh_accept(logp_current::Real, logp_proposal::Real, log_proposal_ratio::Real)

Decide if a proposal ``x'`` with log probability ``\\log p(x') = logp_proposal`` and
log proposal ratio ``\\log k(x', x) - \\log k(x, x') = log_proposal_ratio`` in a
Metropolis-Hastings algorithm with Markov kernel ``k(x_t, x_{t+1})`` and current state
``x`` with log probability ``\\log p(x) = logp_current`` is accepted by evaluating the
Metropolis-Hastings acceptance criterion
```math
\\log U \\leq \\log p(x') - \\log p(x) + \\log k(x', x) - \\log k(x, x')
```
for a uniform random number ``U \\in [0, 1)``.
"""
function mh_accept(logp_current::Real, logp_proposal::Real, log_proposal_ratio::Real)
    # replacing log(rand()) with -randexp() yields test errors
    return log(rand()) + logp_current ≤ logp_proposal + log_proposal_ratio
end

######################
# Default Transition #
######################
# Default
getstats(t) = nothing

abstract type AbstractTransition end

struct Transition{T,F<:AbstractFloat,S<:Union{NamedTuple,Nothing}} <: AbstractTransition
    θ::T
    lp::F # TODO: merge `lp` with `stat`
    stat::S
end

Transition(θ, lp) = Transition(θ, lp, nothing)
function Transition(model::DynamicPPL.Model, vi::AbstractVarInfo, t)
    θ = getparams(model, vi)
    lp = getlogp(vi)
    return Transition(θ, lp, getstats(t))
end

function metadata(t::Transition)
    stat = t.stat
    if stat === nothing
        return (lp=t.lp,)
    else
        return merge((lp=t.lp,), stat)
    end
end

DynamicPPL.getlogp(t::Transition) = t.lp

# Metadata of VarInfo object
metadata(vi::AbstractVarInfo) = (lp=getlogp(vi),)

##########################
# Chain making utilities #
##########################

"""
    getparams(model, t)

Return a named tuple of parameters.
"""
getparams(model, t) = t.θ
function getparams(model::DynamicPPL.Model, vi::DynamicPPL.VarInfo)
    # NOTE: In the past, `invlink(vi, model)` + `values_as(vi, OrderedDict)` was used.
    # Unfortunately, using `invlink` can cause issues in scenarios where the constraints
    # of the parameters change depending on the realizations. Hence we have to use
    # `values_as_in_model`, which re-runs the model and extracts the parameters
    # as they are seen in the model, i.e. in the constrained space. Moreover,
    # this means that the code below will work both of linked and invlinked `vi`.
    # Ref: https://github.com/TuringLang/Turing.jl/issues/2195
    # NOTE: We need to `deepcopy` here to avoid modifying the original `vi`.
    vals = DynamicPPL.values_as_in_model(model, true, deepcopy(vi))

    # Obtain an iterator over the flattened parameter names and values.
    iters = map(DynamicPPL.varname_and_value_leaves, keys(vals), values(vals))

    # Materialize the iterators and concatenate.
    return mapreduce(collect, vcat, iters)
end
function getparams(::DynamicPPL.Model, ::DynamicPPL.VarInfo{NamedTuple{(),Tuple{}}})
    return float(Real)[]
end

function _params_to_array(model::DynamicPPL.Model, ts::Vector)
    names_set = OrderedSet{VarName}()
    # Extract the parameter names and values from each transition.
    dicts = map(ts) do t
        nms_and_vs = getparams(model, t)
        nms = map(first, nms_and_vs)
        vs = map(last, nms_and_vs)
        for nm in nms
            push!(names_set, nm)
        end
        # Convert the names and values to a single dictionary.
        return OrderedDict(zip(nms, vs))
    end
    names = collect(names_set)
    vals = [
        get(dicts[i], key, missing) for i in eachindex(dicts), (j, key) in enumerate(names)
    ]

    return names, vals
end

function get_transition_extras(ts::AbstractVector{<:VarInfo})
    valmat = reshape([getlogp(t) for t in ts], :, 1)
    return [:lp], valmat
end

function get_transition_extras(ts::AbstractVector)
    # Extract all metadata.
    extra_data = map(metadata, ts)
    return names_values(extra_data)
end

function names_values(extra_data::AbstractVector{<:NamedTuple{names}}) where {names}
    values = [getfield(data, name) for data in extra_data, name in names]
    return collect(names), values
end

function names_values(xs::AbstractVector{<:NamedTuple})
    # Obtain all parameter names.
    names_set = Set{Symbol}()
    for x in xs
        for k in keys(x)
            push!(names_set, k)
        end
    end
    names_unique = collect(names_set)

    # Extract all values as matrix.
    values = [haskey(x, name) ? x[name] : missing for x in xs, name in names_unique]

    return names_unique, values
end

getlogevidence(transitions, sampler, state) = missing

# Default MCMCChains.Chains constructor.
# This is type piracy (at least for SampleFromPrior).
function AbstractMCMC.bundle_samples(
    ts::Vector{<:Union{AbstractTransition,AbstractVarInfo}},
    model::AbstractModel,
    spl::Union{Sampler{<:InferenceAlgorithm},SampleFromPrior,RepeatSampler},
    state,
    chain_type::Type{MCMCChains.Chains};
    save_state=false,
    stats=missing,
    sort_chain=false,
    include_varname_to_symbol=true,
    discard_initial=0,
    thinning=1,
    kwargs...,
)
    # Convert transitions to array format.
    # Also retrieve the variable names.
    varnames, vals = _params_to_array(model, ts)
    varnames_symbol = map(Symbol, varnames)

    # Get the values of the extra parameters in each transition.
    extra_params, extra_values = get_transition_extras(ts)

    # Extract names & construct param array.
    nms = [varnames_symbol; extra_params]
    parray = hcat(vals, extra_values)

    # Get the average or final log evidence, if it exists.
    le = getlogevidence(ts, spl, state)

    # Set up the info tuple.
    info = NamedTuple()

    if include_varname_to_symbol
        info = merge(info, (varname_to_symbol=OrderedDict(zip(varnames, varnames_symbol)),))
    end

    if save_state
        info = merge(info, (model=model, sampler=spl, samplerstate=state))
    end

    # Merge in the timing info, if available
    if !ismissing(stats)
        info = merge(info, (start_time=stats.start, stop_time=stats.stop))
    end

    # Conretize the array before giving it to MCMCChains.
    parray = MCMCChains.concretize(parray)

    # Chain construction.
    chain = MCMCChains.Chains(
        parray,
        nms,
        (internals=extra_params,);
        evidence=le,
        info=info,
        start=discard_initial + 1,
        thin=thinning,
    )

    return sort_chain ? sort(chain) : chain
end

# This is type piracy (for SampleFromPrior).
function AbstractMCMC.bundle_samples(
    ts::Vector{<:Union{AbstractTransition,AbstractVarInfo}},
    model::AbstractModel,
    spl::Union{Sampler{<:InferenceAlgorithm},SampleFromPrior,RepeatSampler},
    state,
    chain_type::Type{Vector{NamedTuple}};
    kwargs...,
)
    return map(ts) do t
        # Construct a dictionary of pairs `vn => value`.
        params = OrderedDict(getparams(model, t))
        # Group the variable names by their symbol.
        sym_to_vns = group_varnames_by_symbol(keys(params))
        # Convert the values to a vector.
        vals = map(values(sym_to_vns)) do vns
            map(Base.Fix1(getindex, params), vns)
        end
        return merge(NamedTuple(zip(keys(sym_to_vns), vals)), metadata(t))
    end
end

"""
    group_varnames_by_symbol(vns)

Group the varnames by their symbol.

# Arguments
- `vns`: Iterable of `VarName`.

# Returns
- `OrderedDict{Symbol, Vector{VarName}}`: A dictionary mapping symbol to a vector of varnames.
"""
function group_varnames_by_symbol(vns)
    d = OrderedDict{Symbol,Vector{VarName}}()
    for vn in vns
        sym = DynamicPPL.getsym(vn)
        if !haskey(d, sym)
            d[sym] = VarName[]
        end
        push!(d[sym], vn)
    end
    return d
end

function save(c::MCMCChains.Chains, spl::Sampler, model, vi, samples)
    nt = NamedTuple{(:sampler, :model, :vi, :samples)}((spl, model, deepcopy(vi), samples))
    return setinfo(c, merge(nt, c.info))
end

#######################################
# Concrete algorithm implementations. #
#######################################

include("ess.jl")
include("hmc.jl")
include("mh.jl")
include("is.jl")
include("particle_mcmc.jl")
include("gibbs.jl")
include("sghmc.jl")
include("emcee.jl")
include("prior.jl")

####################################
# Generic sample() method dispatch #
####################################

include("sample.jl")

################
# Typing tools #
################

function DynamicPPL.get_matching_type(
    spl::Sampler{<:Union{PG,SMC}}, vi, ::Type{TV}
) where {T,N,TV<:Array{T,N}}
    return Array{T,N}
end

##############
# Utilities  #
##############

"""

    transitions_from_chain(
        [rng::AbstractRNG,]
        model::Model,
        chain::MCMCChains.Chains;
        sampler = DynamicPPL.SampleFromPrior()
    )

Execute `model` conditioned on each sample in `chain`, and return resulting transitions.

The returned transitions are represented in a `Vector{<:Turing.Inference.Transition}`.

# Details

In a bit more detail, the process is as follows:
1. For every `sample` in `chain`
   1. For every `variable` in `sample`
      1. Set `variable` in `model` to its value in `sample`
   2. Execute `model` with variables fixed as above, sampling variables NOT present
      in `chain` using `SampleFromPrior`
   3. Return sampled variables and log-joint

# Example
```julia-repl
julia> using Turing

julia> @model function demo()
           m ~ Normal(0, 1)
           x ~ Normal(m, 1)
       end;

julia> m = demo();

julia> chain = Chains(randn(2, 1, 1), ["m"]); # 2 samples of `m`

julia> transitions = Turing.Inference.transitions_from_chain(m, chain);

julia> [Turing.Inference.getlogp(t) for t in transitions] # extract the logjoints
2-element Array{Float64,1}:
 -3.6294991938628374
 -2.5697948166987845

julia> [first(t.θ.x) for t in transitions] # extract samples for `x`
2-element Array{Array{Float64,1},1}:
 [-2.0844148956440796]
 [-1.704630494695469]
```
"""
function transitions_from_chain(
    model::DynamicPPL.Model, chain::MCMCChains.Chains; kwargs...
)
    return transitions_from_chain(Random.default_rng(), model, chain; kwargs...)
end

function transitions_from_chain(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    chain::MCMCChains.Chains;
    sampler=DynamicPPL.SampleFromPrior(),
)
    vi = Turing.VarInfo(model)

    iters = Iterators.product(1:size(chain, 1), 1:size(chain, 3))
    transitions = map(iters) do (sample_idx, chain_idx)
        # Set variables present in `chain` and mark those NOT present in chain to be resampled.
        DynamicPPL.setval_and_resample!(vi, chain, sample_idx, chain_idx)
        model(rng, vi, sampler)

        # Convert `VarInfo` into `NamedTuple` and save.
        Transition(model, vi)
    end

    return transitions
end

end # module
