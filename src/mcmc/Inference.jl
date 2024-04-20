module Inference

using ..Essential
using DynamicPPL: Metadata, VarInfo, TypedVarInfo,
    islinked, invlink!, link!,
    setindex!!, push!!,
    setlogp!!, getlogp,
    VarName, getsym, vectorize,
    _getvns, getdist,
    Model, Sampler, SampleFromPrior, SampleFromUniform,
    DefaultContext, PriorContext,
    LikelihoodContext, set_flag!, unset_flag!,
    getspace, inspace
using Distributions, Libtask, Bijectors
using DistributionsAD: VectorOfMultivariate
using LinearAlgebra
using ..Turing: PROGRESS, Turing
using StatsFuns: logsumexp
using Random: AbstractRNG
using DynamicPPL
using AbstractMCMC: AbstractModel, AbstractSampler
using DocStringExtensions: TYPEDEF, TYPEDFIELDS
using DataStructures: OrderedSet
using Setfield: Setfield

import ADTypes
import AbstractMCMC
import AdvancedHMC; const AHMC = AdvancedHMC
import AdvancedMH; const AMH = AdvancedMH
import AdvancedPS
import BangBang
import EllipticalSliceSampling
import LogDensityProblems
import LogDensityProblemsAD
import Random
import MCMCChains
import StatsBase: predict

export  InferenceAlgorithm,
        Hamiltonian,
        GibbsComponent,
        StaticHamiltonian,
        AdaptiveHamiltonian,
        SampleFromUniform,
        SampleFromPrior,
        MH,
        ESS,
        Emcee,
        Gibbs,      # classic sampling
        GibbsConditional,
        HMC,
        SGLD,
        PolynomialStepsize,
        SGHMC,
        HMCDA,
        NUTS,       # Hamiltonian-like sampling
        DynamicNUTS,
        IS,
        SMC,
        CSMC,
        PG,
        Prior,
        assume,
        dot_assume,
        observe,
        dot_observe,
        predict,
        isgibbscomponent,
        externalsampler

#######################
# Sampler abstraction #
#######################
abstract type AbstractAdapter end
abstract type InferenceAlgorithm end
abstract type ParticleInference <: InferenceAlgorithm end
abstract type Hamiltonian <: InferenceAlgorithm end
abstract type StaticHamiltonian <: Hamiltonian end
abstract type AdaptiveHamiltonian <: Hamiltonian end

"""
    ExternalSampler{S<:AbstractSampler,Unconstrained}

Represents a sampler that is not an implementation of `InferenceAlgorithm`.

The `Unconstrained` type-parameter is to indicate whether the sampler requires unconstrained space.

# Fields
$(TYPEDFIELDS)
"""
struct ExternalSampler{S<:AbstractSampler,AD<:ADTypes.AbstractADType,Unconstrained} <: InferenceAlgorithm
    "the sampler to wrap"
    sampler::S
    "the automatic differentiation (AD) backend to use"
    adtype::AD
end

"""
    ExternalSampler(sampler, adtype)
    ExternalSampler{unconstrained}(sampler, adtype)

Wrap a sampler so it can be used as an inference algorithm.

# Arguments
- `sampler::AbstractSampler`: The sampler to wrap.
- `adtype::ADTypes.AbstractADType`: The automatic differentiation (AD) backend to use.
- `unconstrained::Bool=true`: Whether the sampler requires unconstrained space.
"""
function ExternalSampler{unconstrained}(sampler, adtype) where {unconstrained}
    return ExternalSampler{typeof(sampler),typeof(adtype),unconstrained}(sampler, adtype)
end
ExternalSampler(sampler, adtype) = ExternalSampler{true}(sampler, adtype)

"""
    requires_unconstrained_space(sampler::ExternalSampler)

Return `true` if the sampler requires unconstrained space, and `false` otherwise.
"""
requires_unconstrained_space(::ExternalSampler{<:Any,<:Any,Unconstrained}) where {Unconstrained} = Unconstrained

"""
    externalsampler(sampler::AbstractSampler; adtype=AutoForwardDiff(), unconstrained=true)

Wrap a sampler so it can be used as an inference algorithm.

# Arguments
- `sampler::AbstractSampler`: The sampler to wrap.

# Keyword Arguments
- `adtype::ADTypes.AbstractADType=ADTypes.AutoForwardDiff()`: The automatic differentiation (AD) backend to use.
- `unconstrained::Bool=true`: Whether the sampler requires unconstrained space.
"""
function externalsampler(sampler::AbstractSampler; adtype=Turing.DEFAULT_ADTYPE, unconstrained::Bool=true)
    return ExternalSampler{unconstrained}(sampler, adtype)
end


getADType(spl::Sampler) = getADType(spl.alg)
getADType(::SampleFromPrior) = Turing.DEFAULT_ADTYPE

getADType(ctx::DynamicPPL.SamplingContext) = getADType(ctx.sampler)
getADType(ctx::DynamicPPL.AbstractContext) = getADType(DynamicPPL.NodeTrait(ctx), ctx)
getADType(::DynamicPPL.IsLeaf, ctx::DynamicPPL.AbstractContext) = Turing.DEFAULT_ADTYPE
getADType(::DynamicPPL.IsParent, ctx::DynamicPPL.AbstractContext) = getADType(DynamicPPL.childcontext(ctx))

getADType(alg::Hamiltonian) = alg.adtype

function LogDensityProblemsAD.ADgradient(ℓ::DynamicPPL.LogDensityFunction)
    return LogDensityProblemsAD.ADgradient(getADType(ℓ.context), ℓ)
end

function LogDensityProblems.logdensity(
    f::Turing.LogDensityFunction{<:AbstractVarInfo,<:Model,<:DynamicPPL.DefaultContext},
    x::NamedTuple
)
    return DynamicPPL.logjoint(f.model, DynamicPPL.unflatten(f.varinfo, x))
end

# TODO: make a nicer `set_namedtuple!` and move these functions to DynamicPPL.
function DynamicPPL.unflatten(vi::TypedVarInfo, θ::NamedTuple)
    set_namedtuple!(deepcopy(vi), θ)
    return vi
end
DynamicPPL.unflatten(vi::SimpleVarInfo, θ::NamedTuple) = SimpleVarInfo(θ, vi.logp, vi.transformation)

# Algorithm for sampling from the prior
struct Prior <: InferenceAlgorithm end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler::DynamicPPL.Sampler{<:Prior},
    state=nothing;
    kwargs...,
)
    vi = last(DynamicPPL.evaluate!!(
        model,
        VarInfo(),
        SamplingContext(
            rng, DynamicPPL.SampleFromPrior(), DynamicPPL.PriorContext()
        )
    ))
    return vi, nothing
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
# Extended in contrib/inference/abstractmcmc.jl
getstats(t) = nothing

struct Transition{T, F<:AbstractFloat, S<:Union{NamedTuple, Nothing}}
    θ     :: T
    lp    :: F # TODO: merge `lp` with `stat`
    stat  :: S
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
        return (lp = t.lp,)
    else
        return merge((lp = t.lp,), stat)
    end
end

DynamicPPL.getlogp(t::Transition) = t.lp

# Metadata of VarInfo object
metadata(vi::AbstractVarInfo) = (lp = getlogp(vi),)

#########################################
# Default definitions for the interface #
#########################################

function AbstractMCMC.sample(
    model::AbstractModel,
    alg::InferenceAlgorithm,
    N::Integer;
    kwargs...
)
    return AbstractMCMC.sample(Random.default_rng(), model, alg, N; kwargs...)
end

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::AbstractModel,
    alg::InferenceAlgorithm,
    N::Integer;
    kwargs...
)
    return AbstractMCMC.sample(rng, model, Sampler(alg, model), N; kwargs...)
end

function AbstractMCMC.sample(
    model::AbstractModel,
    alg::InferenceAlgorithm,
    ensemble::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer,
    n_chains::Integer;
    kwargs...
)
    return AbstractMCMC.sample(Random.default_rng(), model, alg, ensemble, N, n_chains;
                               kwargs...)
end

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::AbstractModel,
    alg::InferenceAlgorithm,
    ensemble::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer,
    n_chains::Integer;
    kwargs...
)
    return AbstractMCMC.sample(rng, model, Sampler(alg, model), ensemble, N, n_chains;
                               kwargs...)
end

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::AbstractModel,
    sampler::Sampler{<:InferenceAlgorithm},
    ensemble::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer,
    n_chains::Integer;
    chain_type=MCMCChains.Chains,
    progress=PROGRESS[],
    kwargs...
)
    return AbstractMCMC.mcmcsample(rng, model, sampler, ensemble, N, n_chains;
                                   chain_type=chain_type, progress=progress, kwargs...)
end

##########################
# Chain making utilities #
##########################

DynamicPPL.default_chain_type(sampler::Prior) = MCMCChains.Chains
DynamicPPL.default_chain_type(sampler::Sampler{<:InferenceAlgorithm}) = MCMCChains.Chains

"""
    getparams(model, t)

Return a named tuple of parameters.
"""
getparams(model, t) = t.θ
function getparams(model::DynamicPPL.Model, vi::DynamicPPL.VarInfo)
    # Want the end-user to receive parameters in constrained space, so we `link`.
    vi = DynamicPPL.invlink(vi, model)

    # Extract parameter values in a simple form from the `VarInfo`.
    vals = DynamicPPL.values_as(vi, OrderedDict)

    # Obtain an iterator over the flattened parameter names and values.
    iters = map(DynamicPPL.varname_and_value_leaves, keys(vals), values(vals))

    # Materialize the iterators and concatenate.
    return mapreduce(collect, vcat, iters)
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
    vals = [get(dicts[i], key, missing) for i in eachindex(dicts),
        (j, key) in enumerate(names)]


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

function names_values(extra_data::AbstractVector{<:NamedTuple{names}}) where names
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
    values = [
        haskey(x, name) ? x[name] : missing
        for x in xs, name in names_unique
    ]

    return names_unique, values
end

getlogevidence(transitions, sampler, state) = missing

# Default MCMCChains.Chains constructor.
# This is type piracy (at least for SampleFromPrior).
function AbstractMCMC.bundle_samples(
    ts::Vector,
    model::AbstractModel,
    spl::Union{Sampler{<:InferenceAlgorithm},SampleFromPrior},
    state,
    chain_type::Type{MCMCChains.Chains};
    save_state = false,
    stats = missing,
    sort_chain = false,
    include_varname_to_symbol = true,
    discard_initial = 0,
    thinning = 1,
    kwargs...
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
        info = merge(info, (varname_to_symbol = OrderedDict(zip(varnames, varnames_symbol)),))
    end

    if save_state
        info = merge(info, (model = model, sampler = spl, samplerstate = state))
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
        (internals = extra_params,);
        evidence=le,
        info=info,
        start=discard_initial + 1,
        thin=thinning,
    )

    return sort_chain ? sort(chain) : chain
end

# This is type piracy (for SampleFromPrior).
function AbstractMCMC.bundle_samples(
    ts::Vector,
    model::AbstractModel,
    spl::Union{Sampler{<:InferenceAlgorithm},SampleFromPrior},
    state,
    chain_type::Type{Vector{NamedTuple}};
    kwargs...
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
include("gibbs_conditional.jl")
include("gibbs.jl")
include("sghmc.jl")
include("emcee.jl")
include("abstractmcmc.jl")

################
# Typing tools #
################

for alg in (:SMC, :PG, :MH, :IS, :ESS, :Gibbs, :Emcee)
    @eval DynamicPPL.getspace(::$alg{space}) where {space} = space
end
for alg in (:HMC, :HMCDA, :NUTS, :SGLD, :SGHMC)
    @eval DynamicPPL.getspace(::$alg{<:Any, space}) where {space} = space
end

function DynamicPPL.get_matching_type(
    spl::Sampler{<:Union{PG, SMC}},
    vi,
    ::Type{TV},
) where {T, N, TV <: Array{T, N}}
    return Array{T, N}
end

##############
# Utilities  #
##############

DynamicPPL.getspace(spl::Sampler) = getspace(spl.alg)
DynamicPPL.inspace(vn::VarName, spl::Sampler) = inspace(vn, getspace(spl.alg))

"""

    predict([rng::AbstractRNG,] model::Model, chain::MCMCChains.Chains; include_all=false)

Execute `model` conditioned on each sample in `chain`, and return the resulting `Chains`.

If `include_all` is `false`, the returned `Chains` will contain only those variables
sampled/not present in `chain`.

# Details
Internally calls `Turing.Inference.transitions_from_chain` to obtained the samples
and then converts these into a `Chains` object using `AbstractMCMC.bundle_samples`.

# Example
```jldoctest
julia> using Turing; Turing.setprogress!(false);
[ Info: [Turing]: progress logging is disabled globally

julia> @model function linear_reg(x, y, σ = 0.1)
           β ~ Normal(0, 1)

           for i ∈ eachindex(y)
               y[i] ~ Normal(β * x[i], σ)
           end
       end;

julia> σ = 0.1; f(x) = 2 * x + 0.1 * randn();

julia> Δ = 0.1; xs_train = 0:Δ:10; ys_train = f.(xs_train);

julia> xs_test = [10 + Δ, 10 + 2 * Δ]; ys_test = f.(xs_test);

julia> m_train = linear_reg(xs_train, ys_train, σ);

julia> chain_lin_reg = sample(m_train, NUTS(100, 0.65), 200);
┌ Info: Found initial step size
└   ϵ = 0.003125

julia> m_test = linear_reg(xs_test, Vector{Union{Missing, Float64}}(undef, length(ys_test)), σ);

julia> predictions = predict(m_test, chain_lin_reg)
Object of type Chains, with data of type 100×2×1 Array{Float64,3}

Iterations        = 1:100
Thinning interval = 1
Chains            = 1
Samples per chain = 100
parameters        = y[1], y[2]

2-element Array{ChainDataFrame,1}

Summary Statistics
  parameters     mean     std  naive_se     mcse       ess   r_hat
  ──────────  ───────  ──────  ────────  ───────  ────────  ──────
        y[1]  20.1974  0.1007    0.0101  missing  101.0711  0.9922
        y[2]  20.3867  0.1062    0.0106  missing  101.4889  0.9903

Quantiles
  parameters     2.5%    25.0%    50.0%    75.0%    97.5%
  ──────────  ───────  ───────  ───────  ───────  ───────
        y[1]  20.0342  20.1188  20.2135  20.2588  20.4188
        y[2]  20.1870  20.3178  20.3839  20.4466  20.5895


julia> ys_pred = vec(mean(Array(group(predictions, :y)); dims = 1));

julia> sum(abs2, ys_test - ys_pred) ≤ 0.1
true
```
"""
function predict(model::Model, chain::MCMCChains.Chains; kwargs...)
    return predict(Random.default_rng(), model, chain; kwargs...)
end
function predict(rng::AbstractRNG, model::Model, chain::MCMCChains.Chains; include_all = false)
    # Don't need all the diagnostics
    chain_parameters = MCMCChains.get_sections(chain, :parameters)

    spl = DynamicPPL.SampleFromPrior()

    # Sample transitions using `spl` conditioned on values in `chain`
    transitions = transitions_from_chain(rng, model, chain_parameters; sampler = spl)

    # Let the Turing internals handle everything else for you
    chain_result = reduce(
        MCMCChains.chainscat, [
            AbstractMCMC.bundle_samples(
                transitions[:, chain_idx],
                model,
                spl,
                nothing,
                MCMCChains.Chains
            ) for chain_idx = 1:size(transitions, 2)
        ]
    )

    parameter_names = if include_all
        names(chain_result, :parameters)
    else
        filter(k -> ∉(k, names(chain_parameters, :parameters)), names(chain_result, :parameters))
    end

    return chain_result[parameter_names]
end

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
    model::Turing.Model,
    chain::MCMCChains.Chains;
    kwargs...
)
    return transitions_from_chain(Random.default_rng(), model, chain; kwargs...)
end

function transitions_from_chain(
    rng::Random.AbstractRNG,
    model::Turing.Model,
    chain::MCMCChains.Chains;
    sampler = DynamicPPL.SampleFromPrior()
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
