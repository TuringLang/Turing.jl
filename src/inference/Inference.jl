module Inference

using ..Core
using ..Core: logZ
using ..Utilities
using DynamicPPL: Metadata, _tail, VarInfo, TypedVarInfo, 
    islinked, invlink!, getlogp, tonamedtuple, VarName, getsym, vectorize, 
    settrans!, _getvns, getdist, CACHERESET, AbstractSampler,
    Model, Sampler, SampleFromPrior, SampleFromUniform,
    Selector, AbstractSamplerState, DefaultContext, PriorContext,
    LikelihoodContext, MiniBatchContext, set_flag!, unset_flag!, NamedDist, NoDist,
    getspace, inspace
using Distributions, Libtask, Bijectors
using DistributionsAD: VectorOfMultivariate
using LinearAlgebra
using ..Turing: PROGRESS, Turing
using StatsFuns: logsumexp
using Random: GLOBAL_RNG, AbstractRNG, randexp
using DynamicPPL
using AbstractMCMC: AbstractModel, AbstractSampler
using Bijectors: _debug
using DocStringExtensions: TYPEDEF, TYPEDFIELDS

import AbstractMCMC
import AdvancedHMC; const AHMC = AdvancedHMC
import AdvancedMH; const AMH = AdvancedMH
import ..Core: getchunksize, getADbackend
import DynamicPPL: get_matching_type,
    VarName, _getranges, _getindex, getval, _getvns
import EllipticalSliceSampling
import Random
import MCMCChains

export  InferenceAlgorithm,
        Hamiltonian,
        GibbsComponent,
        StaticHamiltonian,
        AdaptiveHamiltonian,
        SampleFromUniform,
        SampleFromPrior,
        MH,
        ESS,
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
        Prior,
        assume,
        dot_assume,
        observe,
        dot_observe,
        resume

#######################
# Sampler abstraction #
#######################
abstract type AbstractAdapter end
abstract type InferenceAlgorithm end
abstract type ParticleInference <: InferenceAlgorithm end
abstract type Hamiltonian{AD} <: InferenceAlgorithm end
abstract type StaticHamiltonian{AD} <: Hamiltonian{AD} end
abstract type AdaptiveHamiltonian{AD} <: Hamiltonian{AD} end

getchunksize(::Type{<:Hamiltonian{AD}}) where AD = getchunksize(AD)
getADbackend(::Hamiltonian{AD}) where AD = AD()

# Algorithm for sampling from the prior
struct Prior <: InferenceAlgorithm end

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

struct Transition{T, F<:AbstractFloat}
    θ  :: T
    lp :: F
end

function Transition(spl::Sampler, nt::NamedTuple=NamedTuple())
    theta = merge(tonamedtuple(spl.state.vi), nt)
    lp = getlogp(spl.state.vi)
    return Transition{typeof(theta), typeof(lp)}(theta, lp)
end

function additional_parameters(::Type{<:Transition})
    return [:lp]
end

DynamicPPL.getlogp(t::Transition) = t.lp

##########################################
# Internal variable names for MCMCChains #
##########################################

const TURING_INTERNAL_VARS = (internals = [
    "elapsed",
    "eval_num",
    "lf_eps",
    "lp",
    "weight",
    "le",
    "acceptance_rate",
    "hamiltonian_energy",
    "hamiltonian_energy_error",
    "max_hamiltonian_energy_error",
    "is_accept",
    "log_density",
    "n_steps",
    "numerical_error",
    "step_size",
    "nom_step_size",
    "tree_depth",
    "is_adapt",
],)

#########################################
# Default definitions for the interface #
#########################################

function AbstractMCMC.sample(
    model::AbstractModel,
    alg::InferenceAlgorithm,
    N::Integer;
    kwargs...
)
    return AbstractMCMC.sample(Random.GLOBAL_RNG, model, alg, N; kwargs...)
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
    rng::AbstractRNG,
    model::AbstractModel,
    sampler::Sampler{<:InferenceAlgorithm},
    N::Integer;
    chain_type=MCMCChains.Chains,
    resume_from=nothing,
    progress=PROGRESS[],
    kwargs...
)
    if resume_from === nothing
        return AbstractMCMC.mcmcsample(rng, model, sampler, N;
                                       chain_type=chain_type, progress=progress, kwargs...)
    else
        return resume(resume_from, N; chain_type=chain_type, progress=progress, kwargs...)
    end
end

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::AbstractModel,
    alg::Prior,
    N::Integer;
    chain_type=MCMCChains.Chains,
    resume_from=nothing,
    progress=PROGRESS[],
    kwargs...
)
    if resume_from === nothing
        return AbstractMCMC.mcmcsample(rng, model, SampleFromPrior(), N;
                                       chain_type=chain_type, progress=progress, kwargs...)
    else
        return resume(resume_from, N; chain_type=chain_type, progress=progress, kwargs...)
    end
end

function AbstractMCMC.sample(
    model::AbstractModel,
    alg::InferenceAlgorithm,
    parallel::AbstractMCMC.AbstractMCMCParallel,
    N::Integer,
    n_chains::Integer;
    kwargs...
)
    return AbstractMCMC.sample(Random.GLOBAL_RNG, model, alg, parallel, N, n_chains;
                               kwargs...)
end

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::AbstractModel,
    alg::InferenceAlgorithm,
    parallel::AbstractMCMC.AbstractMCMCParallel,
    N::Integer,
    n_chains::Integer;
    kwargs...
)
    return AbstractMCMC.sample(rng, model, Sampler(alg, model), parallel, N, n_chains;
                               kwargs...)
end

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::AbstractModel,
    sampler::Sampler{<:InferenceAlgorithm},
    parallel::AbstractMCMC.AbstractMCMCParallel,
    N::Integer,
    n_chains::Integer;
    chain_type=MCMCChains.Chains,
    progress=PROGRESS[],
    kwargs...
)
    return AbstractMCMC.mcmcsample(rng, model, sampler, parallel, N, n_chains;
                                   chain_type=chain_type, progress=progress, kwargs...)
end

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::AbstractModel,
    alg::Prior,
    parallel::AbstractMCMC.AbstractMCMCParallel,
    N::Integer,
    n_chains::Integer;
    chain_type=MCMCChains.Chains,
    progress=PROGRESS[],
    kwargs...
)
    return AbstractMCMC.sample(rng, model, SampleFromPrior(), parallel, N, n_chains;
                               chain_type=chain_type, progress=progress, kwargs...)
end

function AbstractMCMC.sample_init!(
    ::AbstractRNG,
    model::AbstractModel,
    spl::Sampler{<:InferenceAlgorithm},
    N::Integer;
    kwargs...
)
    # Resume the sampler.
    set_resume!(spl; kwargs...)

    # Set the parameters to a starting value.
    initialize_parameters!(spl; kwargs...)
end

function initialize_parameters!(
    spl::Sampler;
    init_theta::Union{Nothing,Vector}=nothing,
    verbose::Bool=false,
    kwargs...
)
    islinked(spl.state.vi, spl) && invlink!(spl.state.vi, spl)
    # Get `init_theta`
    if init_theta !== nothing
        verbose && @info "Using passed-in initial variable values" init_theta
        # Convert individual numbers to length 1 vector; `ismissing(v)` is needed as `size(missing)` is undefined`
        init_theta = [ismissing(v) || size(v) == () ? [v] : v for v in init_theta]
        # Flatten `init_theta`
        init_theta_flat = foldl(vcat, map(vec, init_theta))
        # Create a mask to inidicate which values are not missing
        theta_mask = map(x -> !ismissing(x), init_theta_flat)
        # Get all values
        theta = spl.state.vi[spl]
        @assert length(theta) == length(init_theta_flat) "Provided initial value doesn't match the dimension of the model"
        # Update those which are provided (i.e. not missing)
        theta[theta_mask] .= init_theta_flat[theta_mask]
        # Update in `vi`
        spl.state.vi[spl] = theta
    end
end


##########################
# Chain making utilities #
##########################

"""
    getparams(t)

Return a named tuple of parameters.
"""
getparams(t) = t.θ
getparams(t::VarInfo) = tonamedtuple(TypedVarInfo(t))

function _params_to_array(ts)
    names_set = Set{String}()
    # Extract the parameter names and values from each transition.
    dicts = map(ts) do t
        nms, vs = flatten_namedtuple(getparams(t))
        for nm in nms
            push!(names_set, nm)
        end
        # Convert the names and values to a single dictionary.
        return Dict(nms[j] => vs[j] for j in 1:length(vs))
    end
    names = collect(names_set)
    vals = [get(dicts[i], key, missing) for i in eachindex(dicts),
        (j, key) in enumerate(names)]

    return names, vals
end

function flatten_namedtuple(nt::NamedTuple)
    names_vals = mapreduce(vcat, keys(nt)) do k
        v = nt[k]
        if length(v) == 1
            return [(string(k), v)]
        else
            return mapreduce(vcat, zip(v[1], v[2])) do (vnval, vn)
                return collect(FlattenIterator(vn, vnval))
            end
        end
    end
    return [vn[1] for vn in names_vals], [vn[2] for vn in names_vals]
end

function get_transition_extras(ts::AbstractVector{<:VarInfo})
    valmat = reshape([getlogp(t) for t in ts], :, 1)
    return ["lp"], valmat
end

function get_transition_extras(ts::AbstractVector)
    # Get the extra field names from the sampler state type.
    # This handles things like :lp or :weight.
    extra_params = additional_parameters(eltype(ts))

    # Get the values of the extra parameters.
    local extra_names
    all_vals = []

    # Iterate through each transition.
    for t in ts
        extra_names = String[]
        vals = []

        # Iterate through each of the additional field names
        # in the struct.
        for p in extra_params
            # Check whether the field contains a NamedTuple,
            # in which case we need to iterate through each
            # key/value pair.
            prop = getproperty(t, p)
            if prop isa NamedTuple
                for (k, v) in pairs(prop)
                    push!(extra_names, string(k))
                    push!(vals, v)
                end
            else
                push!(extra_names, string(p))
                push!(vals, prop)
            end
        end
        push!(all_vals, vals)
    end

    # Convert the vector-of-vectors to a matrix.
    valmat = [all_vals[i][j] for i in 1:length(ts), j in 1:length(all_vals[1])]

    return extra_names, valmat
end

getlogevidence(sampler) = missing
function getlogevidence(sampler::Sampler)
    if isdefined(sampler.state, :average_logevidence)
        return sampler.state.average_logevidence
    elseif isdefined(sampler.state, :final_logevidence)
        return sampler.state.final_logevidence
    else
        return missing
    end
end

# Default MCMCChains.Chains constructor.
# This is type piracy (at least for SampleFromPrior).
function AbstractMCMC.bundle_samples(
    rng::AbstractRNG,
    model::AbstractModel,
    spl::Union{Sampler{<:InferenceAlgorithm},SampleFromPrior},
    N::Integer,
    ts::Vector,
    chain_type::Type{MCMCChains.Chains};
    save_state = false,
    kwargs...
)
    # Convert transitions to array format.
    # Also retrieve the variable names.
    nms, vals = _params_to_array(ts)

    # Get the values of the extra parameters in each transition.
    extra_params, extra_values = get_transition_extras(ts)

    # Extract names & construct param array.
    nms = [nms; extra_params]
    parray = hcat(vals, extra_values)

    # Get the average or final log evidence, if it exists.
    le = getlogevidence(spl)

    # Set up the info tuple.
    if save_state
        info = (range = rng, model = model, spl = spl)
    else
        info = NamedTuple()
    end

    # Conretize the array before giving it to MCMCChains.
    parray = MCMCChains.concretize(parray)

    # Chain construction.
    return MCMCChains.Chains(
        parray,
        string.(nms),
        deepcopy(TURING_INTERNAL_VARS);
        evidence=le,
        info=info,
        sorted=true
    )
end

# This is type piracy (for SampleFromPrior).
function AbstractMCMC.bundle_samples(
    rng::AbstractRNG,
    model::AbstractModel,
    spl::Union{Sampler{<:InferenceAlgorithm},SampleFromPrior},
    N::Integer,
    ts::Vector,
    chain_type::Type{Vector{NamedTuple}};
    discard_adapt::Bool=true,
    save_state=false,
    kwargs...
)
    nts = Vector{NamedTuple}(undef, N)

    for (i, t) in enumerate(ts)
        params = getparams(t)

        k = collect(keys(params))
        vs = []
        for v in values(params)
            push!(vs, v[1])
        end

        push!(k, :lp)

        nts[i] = NamedTuple{tuple(k...)}(tuple(vs..., getlogp(t)))
    end

    return map(identity, nts)
end

function save(c::MCMCChains.Chains, spl::Sampler, model, vi, samples)
    nt = NamedTuple{(:spl, :model, :vi, :samples)}((spl, model, deepcopy(vi), samples))
    return setinfo(c, merge(nt, c.info))
end

function resume(
    c::MCMCChains.Chains,
    n_iter::Int;
    chain_type=MCMCChains.Chains,
    progress=PROGRESS[],
    kwargs...
)
    @assert !isempty(c.info) "[Turing] cannot resume from a chain without state info"

    # Sample a new chain.
    newchain = AbstractMCMC.mcmcsample(
        c.info[:range],
        c.info[:model],
        c.info[:spl],
        n_iter;
        resume_from=c,
        reuse_spl_n=n_iter,
        chain_type=MCMCChains.Chains,
        progress=progress,
        kwargs...
    )

    # Stick the new samples at the end of the old chain.
    return vcat(c, newchain)
end

function set_resume!(
    s::Sampler;
    resume_from::Union{MCMCChains.Chains, Nothing}=nothing,
    kwargs...
)
    # If we're resuming, grab the sampler info.
    if resume_from !== nothing
        s = resume_from.info[:spl]
    end
end

#########################
# Default sampler state #
#########################

"""
A blank `AbstractSamplerState` that contains only `VarInfo` information.
"""
mutable struct SamplerState{VIType<:VarInfo} <: AbstractSamplerState
    vi :: VIType
end

#######################################
# Concrete algorithm implementations. #
#######################################

include("ess.jl")
include("hmc.jl")
include("mh.jl")
include("is.jl")
include("AdvancedSMC.jl")
include("gibbs.jl")
include("../contrib/inference/sghmc.jl")

################
# Typing tools #
################

for alg in (:SMC, :PG, :MH, :IS, :ESS, :Gibbs)
    @eval DynamicPPL.getspace(::$alg{space}) where {space} = space
end
for alg in (:HMC, :HMCDA, :NUTS, :SGLD, :SGHMC)
    @eval DynamicPPL.getspace(::$alg{<:Any, space}) where {space} = space
end

floatof(::Type{T}) where {T <: Real} = typeof(one(T)/one(T))
floatof(::Type) = Real # fallback if type inference failed

function get_matching_type(
    spl::AbstractSampler, 
    vi,
    ::Type{T},
) where {T}
    return T
end
function get_matching_type(
    spl::AbstractSampler, 
    vi, 
    ::Type{<:Union{Missing, AbstractFloat}},
)
    return Union{Missing, floatof(eltype(vi, spl))}
end
function get_matching_type(
    spl::AbstractSampler,
    vi,
    ::Type{<:AbstractFloat},
)
    return floatof(eltype(vi, spl))
end
function get_matching_type(
    spl::AbstractSampler,
    vi,
    ::Type{TV},
) where {T, N, TV <: Array{T, N}}
    return Array{get_matching_type(spl, vi, T), N}
end
function get_matching_type(
    spl::Sampler{<:Union{PG, SMC}}, 
    vi,
    ::Type{TV},
) where {T, N, TV <: Array{T, N}}
    return TArray{T, N}
end

##############
# Utilities  #
##############

DynamicPPL.getspace(spl::Sampler) = getspace(spl.alg)
DynamicPPL.inspace(vn::VarName, spl::Sampler) = inspace(vn, getspace(spl.alg))

end # module
