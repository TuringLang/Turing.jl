module Inference

using DynamicPPL:
    DynamicPPL,
    @model,
    Metadata,
    VarInfo,
    LogDensityFunction,
    SimpleVarInfo,
    AbstractVarInfo,
    # TODO(mhauru) all_varnames_grouped_by_symbol isn't exported by DPPL, because it is only
    # implemented for NTVarInfo. It is used by mh.jl. Either refactor mh.jl to not use it
    # or implement it for other VarInfo types and export it from DPPL.
    all_varnames_grouped_by_symbol,
    syms,
    setindex!!,
    push!!,
    setlogp!!,
    getlogjoint,
    getlogjoint_internal,
    VarName,
    getsym,
    getdist,
    Model,
    DefaultContext
using Distributions, Libtask, Bijectors
using DistributionsAD: VectorOfMultivariate
using LinearAlgebra
using ..Turing: PROGRESS, Turing
using StatsFuns: logsumexp
using Random: AbstractRNG
using AbstractMCMC: AbstractModel, AbstractSampler
using DocStringExtensions: FIELDS, TYPEDEF, TYPEDFIELDS
using DataStructures: OrderedSet, OrderedDict
using Accessors: Accessors

import ADTypes
import AbstractMCMC
import AbstractPPL
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

export Hamiltonian,
    StaticHamiltonian,
    AdaptiveHamiltonian,
    MH,
    ESS,
    Emcee,
    Gibbs,      # classic sampling
    GibbsConditional,  # conditional sampling
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
    predict,
    externalsampler,
    init_strategy,
    loadstate

#########################################
# Generic AbstractMCMC methods dispatch #
#########################################

const DEFAULT_CHAIN_TYPE = MCMCChains.Chains
include("abstractmcmc.jl")

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
getstats(::Any) = NamedTuple()
getstats(nt::NamedTuple) = nt

struct Transition{T,F<:AbstractFloat,N<:NamedTuple}
    θ::T
    logprior::F
    loglikelihood::F
    stat::N

    """
        Transition(model::Model, vi::AbstractVarInfo, stats; reevaluate=true)

    Construct a new `Turing.Inference.Transition` object using the outputs of a
    sampler step.

    Here, `vi` represents a VarInfo _for which the appropriate parameters have
    already been set_. However, the accumulators (e.g. logp) may in general
    have junk contents. The role of this method is to re-evaluate `model` and
    thus set the accumulators to the correct values.

    `stats` is any object on which `Turing.Inference.getstats` can be called to
    return a NamedTuple of statistics. This could be, for example, the transition
    returned by an (unwrapped) external sampler. Or alternatively, it could
    simply be a NamedTuple itself (for which `getstats` acts as the identity).

    By default, the model is re-evaluated in order to obtain values of:
      - the values of the parameters as per user parameterisation (`vals_as_in_model`)
      - the various components of the log joint probability (`logprior`, `loglikelihood`)
    that are guaranteed to be correct.

    If you **know** for a fact that the VarInfo `vi` already contains this information,
    then you can set `reevaluate=false` to skip the re-evaluation step.

    !!! warning
        Note that in general this is unsafe and may lead to wrong results.

    If `reevaluate` is set to `false`, it is the caller's responsibility to ensure that
    the `VarInfo` passed in has `ValuesAsInModelAccumulator`, `LogPriorAccumulator`,
    and `LogLikelihoodAccumulator` set up with the correct values. Note that the
    `ValuesAsInModelAccumulator` must also have `include_colon_eq == true`, i.e. it
    must be set up to track `x := y` statements.
    """
    function Transition(
        model::DynamicPPL.Model, vi::AbstractVarInfo, stats; reevaluate=true
    )
        # Avoid mutating vi as it may be used later e.g. when constructing
        # sampler states.
        vi = deepcopy(vi)
        if reevaluate
            vi = DynamicPPL.setaccs!!(
                vi,
                (
                    DynamicPPL.ValuesAsInModelAccumulator(true),
                    DynamicPPL.LogPriorAccumulator(),
                    DynamicPPL.LogLikelihoodAccumulator(),
                ),
            )
            _, vi = DynamicPPL.evaluate!!(model, vi)
        end

        # Extract all the information we need
        vals_as_in_model = DynamicPPL.getacc(vi, Val(:ValuesAsInModel)).values
        logprior = DynamicPPL.getlogprior(vi)
        loglikelihood = DynamicPPL.getloglikelihood(vi)

        # Get additional statistics
        stats = getstats(stats)
        return new{typeof(vals_as_in_model),typeof(logprior),typeof(stats)}(
            vals_as_in_model, logprior, loglikelihood, stats
        )
    end

    function Transition(
        model::DynamicPPL.Model,
        untyped_vi::DynamicPPL.VarInfo{<:DynamicPPL.Metadata},
        stats;
        reevaluate=true,
    )
        # Re-evaluating the model is unconscionably slow for untyped VarInfo. It's
        # much faster to convert it to a typed varinfo first, hence this method.
        # https://github.com/TuringLang/Turing.jl/issues/2604
        return Transition(
            model, DynamicPPL.typed_varinfo(untyped_vi), stats; reevaluate=reevaluate
        )
    end
end

function getstats_with_lp(t::Transition)
    return merge(
        t.stat,
        (
            lp=t.logprior + t.loglikelihood,
            logprior=t.logprior,
            loglikelihood=t.loglikelihood,
        ),
    )
end
function getstats_with_lp(vi::AbstractVarInfo)
    return (
        lp=DynamicPPL.getlogjoint(vi),
        logprior=DynamicPPL.getlogprior(vi),
        loglikelihood=DynamicPPL.getloglikelihood(vi),
    )
end

##########################
# Chain making utilities #
##########################

# TODO(penelopeysm): Separate Turing.Inference.getparams (should only be
# defined for AbstractVarInfo and Turing.Inference.Transition; returns varname
# => value maps) from AbstractMCMC.getparams (defined for any sampler transition,
# returns vector).
"""
    Turing.Inference.getparams(model::DynamicPPL.Model, t::Any)

Return a vector of parameter values from the given sampler transition `t` (i.e.,
the first return value of AbstractMCMC.step). By default, returns the `t.θ` field.

!!! note
    This method only needs to be implemented for external samplers. It will be
removed in future releases and replaced with `AbstractMCMC.getparams`.
"""
getparams(::DynamicPPL.Model, t) = t.θ
"""
    Turing.Inference.getparams(model::DynamicPPL.Model, t::AbstractVarInfo)

Return a key-value map of parameters from the varinfo.
"""
function getparams(model::DynamicPPL.Model, vi::DynamicPPL.VarInfo)
    t = Transition(model, vi, nothing)
    return getparams(model, t)
end
function _params_to_array(model::DynamicPPL.Model, ts::Vector)
    names_set = OrderedSet{VarName}()
    # Extract the parameter names and values from each transition.
    dicts = map(ts) do t
        # In general getparams returns a dict of VarName => values. We need to also
        # split it up into constituent elements using
        # `AbstractPPL.varname_and_value_leaves` because otherwise MCMCChains.jl
        # won't understand it.
        vals = getparams(model, t)
        nms_and_vs = if isempty(vals)
            Tuple{VarName,Any}[]
        else
            iters = map(AbstractPPL.varname_and_value_leaves, keys(vals), values(vals))
            mapreduce(collect, vcat, iters)
        end
        nms = map(first, nms_and_vs)
        vs = map(last, nms_and_vs)
        for nm in nms
            push!(names_set, nm)
        end
        # Convert the names and values to a single dictionary.
        return OrderedDict(zip(nms, vs))
    end
    names = collect(names_set)
    vals = [get(dicts[i], key, missing) for i in eachindex(dicts), key in names]

    return names, vals
end

function get_transition_extras(ts::AbstractVector)
    # Extract stats + log probabilities from each transition or VarInfo
    extra_data = map(getstats_with_lp, ts)
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
function AbstractMCMC.bundle_samples(
    ts::Vector{<:Transition},
    model::DynamicPPL.Model,
    spl::AbstractSampler,
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

function AbstractMCMC.bundle_samples(
    ts::Vector{<:Transition},
    model::DynamicPPL.Model,
    spl::AbstractSampler,
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
        return merge(NamedTuple(zip(keys(sym_to_vns), vals)), getstats_with_lp(t))
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

function save(c::MCMCChains.Chains, spl::AbstractSampler, model, vi, samples)
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
include("gibbs_conditional.jl")
include("sghmc.jl")
include("emcee.jl")
include("prior.jl")

################
# Typing tools #
################

function DynamicPPL.get_matching_type(
    spl::Union{PG,SMC}, vi, ::Type{TV}
) where {T,N,TV<:Array{T,N}}
    return Array{T,N}
end

end # module
