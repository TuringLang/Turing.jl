module Inference

using ..Core, ..Utilities
using DynamicPPL: Metadata, _tail, VarInfo, TypedVarInfo, 
    islinked, invlink!, getlogp, tonamedtuple, VarName, getsym, vectorize, 
    settrans!, _getvns, getdist, split_var_str, CACHERESET, AbstractSampler,
    Model, runmodel!, Sampler, SampleFromPrior, SampleFromUniform,
    Selector, AbstractSamplerState, DefaultContext, PriorContext,
    LikelihoodContext, MiniBatchContext, set_flag!, unset_flag!
using Distributions, Libtask, Bijectors
using ProgressMeter, LinearAlgebra
using ..Turing: PROGRESS, NamedDist, NoDist, Turing
using StatsFuns: logsumexp
using Random: GLOBAL_RNG, AbstractRNG, randexp
using AbstractMCMC, DynamicPPL

import MCMCChains: Chains
import AdvancedHMC; const AHMC = AdvancedHMC
import ..Core: getchunksize, getADtype
import AbstractMCMC: AbstractTransition, sample, step!, sample_init!,
    transitions_init, sample_end!, AbstractSampler, transition_type,
    callback, init_callback, AbstractCallback, psample
import DynamicPPL: tilde, dot_tilde, getspace, get_matching_type

export  InferenceAlgorithm,
        Hamiltonian,
        AbstractGibbs,
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
        reset!,
        step!,
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
getADtype(::Hamiltonian{AD}) where AD = AD

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

struct Transition{T, F<:AbstractFloat} <: AbstractTransition
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
    rng::AbstractRNG,
    model::AbstractModel,
    alg::InferenceAlgorithm,
    N::Integer;
    kwargs...
)
    return sample(rng, model, Sampler(alg, model), N; progress=PROGRESS[], kwargs...)
end

function AbstractMCMC.sample(
    model::AbstractModel,
    alg::InferenceAlgorithm,
    N::Integer;
    resume_from=nothing,
    kwargs...
)
    if resume_from === nothing
        return sample(model, Sampler(alg, model), N; progress=PROGRESS[], kwargs...)
    else
        return resume(resume_from, N)
    end
end


function AbstractMCMC.psample(
    model::AbstractModel,
    alg::InferenceAlgorithm,
    N::Integer,
    n_chains::Integer;
    kwargs...
)
    return psample(GLOBAL_RNG, model, alg, N, n_chains; progress=false, kwargs...)
end

function AbstractMCMC.psample(
    rng::AbstractRNG,
    model::AbstractModel,
    alg::InferenceAlgorithm,
    N::Integer,
    n_chains::Integer;
    kwargs...
)
    return psample(rng, model, Sampler(alg, model), N, n_chains; progress=false, kwargs...)
end

function AbstractMCMC.sample_init!(
    ::AbstractRNG,
    model::Model,
    spl::Sampler,
    N::Integer;
    kwargs...
)
    # Resume the sampler.
    set_resume!(spl; kwargs...)

    # Set the parameters to a starting value.
    initialize_parameters!(spl; kwargs...)
end

function AbstractMCMC.sample_end!(
    ::AbstractRNG,
    ::Model,
    ::AbstractSampler,
    ::Integer,
    ::Vector{<:AbstractTransition};
    kwargs...
)
    # Silence the default API function.
end

function initialize_parameters!(
    spl::AbstractSampler;
    init_theta::Union{Nothing,Vector}=nothing,
    verbose::Bool=false,
    kwargs...
)
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

function _params_to_array(ts::Vector{<:AbstractTransition}, spl::Sampler)
    names = Set{String}()
    dicts = Vector{Dict{String, Any}}()

    # Extract the parameter names and values from each transition.
    for t in ts
        nms, vs = flatten_namedtuple(t.θ)
        push!(names, nms...)

        # Convert the names and values to a single dictionary.
        d = Dict{String, Any}()
        for (k, v) in zip(nms, vs)
            d[k] = v
        end
        push!(dicts, d)
    end

    # Convert the set to an ordered vector so the parameter ordering
    # is deterministic.
    ordered_names = collect(names)
    vals = Matrix{Union{Real, Missing}}(undef, length(ts), length(ordered_names))

    # Place each element of all dicts into the returned value matrix.
    for i in eachindex(dicts)
        for (j, key) in enumerate(ordered_names)
            vals[i,j] = get(dicts[i], key, missing)
        end
    end

    return ordered_names, vals
end

function flatten_namedtuple(nt::NamedTuple{pnames}) where {pnames}
    vals = Vector{Real}()
    names = Vector{AbstractString}()

    for k in pnames
        v = nt[k]
        if length(v) == 1
            flatten(names, vals, string(k), v)
        else
            for (vnval, vn) in zip(v[1], v[2])
                flatten(names, vals, vn, vnval)
            end
        end
    end

    return names, vals
end

function flatten(names, value :: AbstractArray, k :: String, v)
    if isa(v, Number)
        name = k
        push!(value, v)
        push!(names, name)
    elseif isa(v, Array)
        for i = eachindex(v)
            if isa(v[i], Number)
                name = string(ind2sub(size(v), i))
                name = replace(name, "(" => "[");
                name = replace(name, ",)" => "]");
                name = replace(name, ")" => "]");
                name = k * name
                isa(v[i], Nothing) && println(v, i, v[i])
                push!(value, v[i])
                push!(names, name)
            elseif isa(v[i], AbstractArray)
                name = k * string(ind2sub(size(v), i))
                flatten(names, value, name, v[i])
            else
                error("Unknown var type: typeof($v[i])=$(typeof(v[i]))")
            end
        end
    else
        error("Unknown var type: typeof($v)=$(typeof(v))")
    end
    return
end

ind2sub(v, i) = Tuple(CartesianIndices(v)[i])

function get_transition_extras(ts::Vector{<:AbstractTransition})
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

# Default Chains constructor.
function AbstractMCMC.bundle_samples(
    rng::AbstractRNG,
    model::AbstractModel,
    spl::Sampler,
    N::Integer,
    ts::Vector{<:AbstractTransition};
    discard_adapt::Bool=true,
    save_state=true,
    kwargs...
)
    # Check if we have adaptation samples.
    if discard_adapt && :n_adapts in fieldnames(typeof(spl.alg))
        ts = ts[(spl.alg.n_adapts+1):end]
    end

    # Convert transitions to array format.
    # Also retrieve the variable names.
    nms, vals = _params_to_array(ts, spl)

    # Get the values of the extra parameters in each Transition struct.
    extra_params, extra_values = get_transition_extras(ts)

    # Extract names & construct param array.
    nms = string.(vcat(nms..., string.(extra_params)...))
    parray = hcat(vals, extra_values)

    # If the state field has average_logevidence or final_logevidence, grab that.
    le = missing
    if :average_logevidence in fieldnames(typeof(spl.state))
        le = getproperty(spl.state, :average_logevidence)
    elseif :final_logevidence in fieldnames(typeof(spl.state))
        le = getproperty(spl.state, :final_logevidence)
    end

    # Check whether to invlink! the varinfo
    if islinked(spl.state.vi, spl)
        invlink!(spl.state.vi, spl)
    end

    # Set up the info tuple.
    if save_state
        info = (range = rng, model = model, spl = spl, vi = spl.state.vi)
    else
        info = NamedTuple()
    end

    # Chain construction.
    return Chains(
        parray,
        string.(nms),
        deepcopy(TURING_INTERNAL_VARS);
        evidence=le,
        info=info,
        sorted=true
    )
end

function save(c::Chains, spl::AbstractSampler, model, vi, samples)
    nt = NamedTuple{(:spl, :model, :vi, :samples)}((spl, model, deepcopy(vi), samples))
    return setinfo(c, merge(nt, c.info))
end

function resume(c::Chains, n_iter::Int; kwargs...)
    @assert !isempty(c.info) "[Turing] cannot resume from a chain without state info"

    # Sample a new chain.
    newchain = sample(
        c.info[:range],
        c.info[:model],
        c.info[:spl],
        n_iter;
        resume_from=c,
        reuse_spl_n=n_iter,
        kwargs...
    )

    # Stick the new samples at the end of the old chain.
    return vcat(c, newchain)
end

function set_resume!(
    s::Sampler;
    resume_from::Union{Chains, Nothing}=nothing,
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
include("../contrib/inference/AdvancedSMCExtensions.jl")

################
# Typing tools #
################

for alg in (:SMC, :PG, :PMMH, :IPMCMC, :MH, :IS, :ESS, :Gibbs)
    @eval getspace(::$alg{space}) where {space} = space
end
for alg in (:HMC, :HMCDA, :NUTS, :SGLD, :SGHMC)
    @eval getspace(::$alg{<:Any, space}) where {space} = space
end

floatof(::Type{T}) where {T <: Real} = typeof(one(T)/one(T))
floatof(::Type) = Real # fallback if type inference failed

function get_matching_type(
    spl::AbstractSampler, 
    vi::VarInfo, 
    ::Type{T},
) where {T}
    return T
end
function get_matching_type(
    spl::AbstractSampler, 
    vi::VarInfo, 
    ::Type{<:AbstractFloat},
)
    return floatof(eltype(vi, spl))
end
function get_matching_type(
    spl::Sampler{<:Hamiltonian}, 
    vi::VarInfo, 
    ::Type{<:Union{Missing, AbstractFloat}},
)
    return Union{Missing, floatof(eltype(vi, spl))}
end
function get_matching_type(
    spl::Sampler{<:Hamiltonian}, 
    vi::VarInfo, 
    ::Type{<:AbstractFloat},
)
    return floatof(eltype(vi, spl))
end
function get_matching_type(
    spl::Sampler{<:Hamiltonian}, 
    vi::VarInfo, 
    ::Type{TV},
) where {T, N, TV <: Array{T, N}}
    return Array{get_matching_type(spl, vi, T), N}
end
function get_matching_type(
    spl::Sampler{<:Union{PG, SMC}}, 
    vi::VarInfo, 
    ::Type{TV},
) where {T, N, TV <: Array{T, N}}
    return TArray{T, N}
end

## Fallback functions

alg_str(spl::Sampler) = string(nameof(typeof(spl.alg)))
transition_type(spl::Sampler) = typeof(Transition(spl))

# utility funcs for querying sampler information
require_gradient(spl::Sampler) = false
require_particles(spl::Sampler) = false

_getindex(x, inds::Tuple) = _getindex(x[first(inds)...], Base.tail(inds))
_getindex(x, inds::Tuple{}) = x

# assume
function tilde(ctx::DefaultContext, sampler, right, vn::VarName, _, vi)
    return _tilde(sampler, right, vn, vi)
end
function tilde(ctx::PriorContext, sampler, right, vn::VarName, inds, vi)
    if ctx.vars !== nothing
        vi[vn] = vectorize(right, _getindex(getfield(ctx.vars, getsym(vn)), inds))
        settrans!(vi, false, vn)
    end
    return _tilde(sampler, right, vn, vi)
end
function tilde(ctx::LikelihoodContext, sampler, right, vn::VarName, inds, vi)
    if ctx.vars !== nothing
        vi[vn] = vectorize(right, _getindex(getfield(ctx.vars, getsym(vn)), inds))
        settrans!(vi, false, vn)
    end
    return _tilde(sampler, NoDist(right), vn, vi)
end
function tilde(ctx::MiniBatchContext, sampler, right, left::VarName, inds, vi)
    return tilde(ctx.ctx, sampler, right, left, inds, vi)
end

function _tilde(sampler, right, vn::VarName, vi)
    return Turing.assume(sampler, right, vn, vi)
end
function _tilde(sampler, right::NamedDist, vn::VarName, vi)
    name = right.name
    if name isa String
        sym_str, inds = split_var_str(name, String)
        sym = Symbol(sym_str)
        vn = VarName{sym}(inds)
    elseif name isa Symbol
        vn = VarName{name}("")
    elseif name isa VarName
        vn = name
    else
        throw("Unsupported variable name. Please use either a string, symbol or VarName.")
    end
    return _tilde(sampler, right.dist, vn, vi)
end

# observe
function tilde(ctx::DefaultContext, sampler, right, left, vi)
    return _tilde(sampler, right, left, vi)
end
function tilde(ctx::PriorContext, sampler, right, left, vi)
    return 0
end
function tilde(ctx::LikelihoodContext, sampler, right, left, vi)
    return _tilde(sampler, right, left, vi)
end
function tilde(ctx::MiniBatchContext, sampler, right, left, vi)
    return ctx.loglike_scalar * tilde(ctx.ctx, sampler, right, left, vi)
end

_tilde(sampler, right, left, vi) = Turing.observe(sampler, right, left, vi)

function assume(spl::Sampler, dist)
    error("Turing.assume: unmanaged inference algorithm: $(typeof(spl))")
end

function observe(spl::Sampler, weight)
    error("Turing.observe: unmanaged inference algorithm: $(typeof(spl))")
end

function assume(
    spl::Union{SampleFromPrior, SampleFromUniform},
    dist::Distribution,
    vn::VarName,
    vi::VarInfo,
)
    if haskey(vi, vn)
        if is_flagged(vi, vn, "del")
            unset_flag!(vi, vn, "del")
            r = spl isa SampleFromUniform ? init(dist) : rand(dist)
            vi[vn] = vectorize(dist, r)
            setorder!(vi, vn, vi.num_produce)
        else
        r = vi[vn]
        end
    else
        r = isa(spl, SampleFromUniform) ? init(dist) : rand(dist)
        push!(vi, vn, r, dist, spl)
    end
    # NOTE: The importance weight is not correctly computed here because
    #       r is genereated from some uniform distribution which is different from the prior
    # acclogp!(vi, logpdf_with_trans(dist, r, istrans(vi, vn)))

    return r, logpdf_with_trans(dist, r, istrans(vi, vn))
end

function observe(
    spl::Union{SampleFromPrior, SampleFromUniform},
    dist::Distribution,
    value,
    vi::VarInfo,
)
    vi.num_produce += one(vi.num_produce)
    return logpdf(dist, value)
end

# .~ functions

# assume
function dot_tilde(ctx::DefaultContext, sampler, right, left, vn::VarName, _, vi)
    vns, dist = get_vns_and_dist(right, left, vn)
    return _dot_tilde(sampler, dist, left, vns, vi)
end
function dot_tilde(
    ctx::LikelihoodContext,
    sampler,
    right,
    left,
    vn::VarName,
    inds,
    vi,
)
    if ctx.vars !== nothing
        var = _getindex(getfield(ctx.vars, getsym(vn)), inds)
        vns, dist = get_vns_and_dist(right, var, vn)
        set_val!(vi, vns, dist, var)
        settrans!.(Ref(vi), false, vns)
    else
        vns, dist = get_vns_and_dist(right, left, vn)
    end
    return _dot_tilde(sampler, NoDist(dist), left, vns, vi)
end
function dot_tilde(ctx::MiniBatchContext, sampler, right, left, vn::VarName, inds, vi)
    return dot_tilde(ctx.ctx, sampler, right, left, vn, inds, vi)
end
function dot_tilde(
    ctx::PriorContext,
    sampler,
    right,
    left,
    vn::VarName,
    inds,
    vi,
)
    if ctx.vars !== nothing
        var = _getindex(getfield(ctx.vars, getsym(vn)), inds)
        vns, dist = get_vns_and_dist(right, var, vn)
        set_val!(vi, vns, dist, var)
        settrans!.(Ref(vi), false, vns)
    else
        vns, dist = get_vns_and_dist(right, left, vn)
    end
    return _dot_tilde(sampler, dist, left, vns, vi)
end

function get_vns_and_dist(dist::NamedDist, var, vn::VarName)
    name = dist.name
    if name isa String
        sym_str, inds = split_var_str(name, String)
        sym = Symbol(sym_str)
        vn = VarName{sym}(inds)
    elseif name isa Symbol
        vn = VarName{name}("")
    elseif name isa VarName
        vn = name
    else
        throw("Unsupported variable name. Please use either a string, symbol or VarName.")
    end
    return get_vns_and_dist(dist.dist, var, vn)
end
function get_vns_and_dist(dist::MultivariateDistribution, var::AbstractMatrix, vn::VarName)
    getvn = i -> VarName(vn, vn.indexing * "[Colon(),$i]")
    return getvn.(1:size(var, 2)), dist
end
function get_vns_and_dist(
    dist::Union{Distribution, AbstractArray{<:Distribution}}, 
    var::AbstractArray, 
    vn::VarName
)
    getvn = ind -> VarName(vn, vn.indexing * "[" * join(Tuple(ind), ",") * "]")
    return getvn.(CartesianIndices(var)), dist
end

function _dot_tilde(sampler, right, left, vns::AbstractArray{<:VarName}, vi)
    return dot_assume(sampler, right, vns, left, vi)
end

# Ambiguity error when not sure to use Distributions convention or Julia broadcasting semantics
function _dot_tilde(
    sampler::AbstractSampler,
    right::Union{MultivariateDistribution, AbstractVector{<:MultivariateDistribution}},
    left::AbstractMatrix{>:AbstractVector},
    vn::AbstractVector{<:VarName},
    vi::VarInfo,
)
    throw(ambiguity_error_msg())
end

function dot_assume(
    spl::Union{SampleFromPrior, SampleFromUniform},
    dist::MultivariateDistribution,
    vns::AbstractVector{<:VarName},
    var::AbstractMatrix,
    vi::VarInfo,
)
    @assert dim(dist) == size(var, 1)
    r = get_and_set_val!(vi, vns, dist, spl)
    lp = sum(logpdf_with_trans(dist, r, istrans(vi, vns[1])))
    var .= r
    return var, lp
end
function dot_assume(
    spl::Union{SampleFromPrior, SampleFromUniform},
    dists::Union{Distribution, AbstractArray{<:Distribution}},
    vns::AbstractArray{<:VarName},
    var::AbstractArray,
    vi::VarInfo,
)
    r = get_and_set_val!(vi, vns, dists, spl)
    lp = sum(logpdf_with_trans.(dists, r, istrans(vi, vns[1])))
    var .= r
    return var, lp
end
function dot_assume(
    spl::Sampler,
    ::Any,
    ::AbstractArray{<:VarName},
    ::Any,
    ::VarInfo
)
    error("[Turing] $(alg_str(spl)) doesn't support vectorizing assume statement")
end

function get_and_set_val!(
    vi::VarInfo,
    vns::AbstractVector{<:VarName},
    dist::MultivariateDistribution,
    spl::AbstractSampler,
)
    n = length(vns)
    if haskey(vi, vns[1])
        if is_flagged(vi, vns[1], "del")
            unset_flag!(vi, vns[1], "del")
            r = spl isa SampleFromUniform ? init(dist, n) : rand(dist, n)
            for i in 1:n
                vn = vns[i]
                vi[vn] = vectorize(dist, r[:, i])
                setorder!(vi, vn, vi.num_produce)
            end
        else
        r = vi[vns]
        end
    else
        r = spl isa SampleFromUniform ? init(dist, n) : rand(dist, n)
        for i in 1:n
            push!(vi, vns[i], r[:,i], dist, spl)
        end
    end
    return r
end
function get_and_set_val!(
    vi::VarInfo,
    vns::AbstractArray{<:VarName},
    dists::Union{Distribution, AbstractArray{<:Distribution}},
    spl::AbstractSampler,
)
    if haskey(vi, vns[1])
        if is_flagged(vi, vns[1], "del")
            unset_flag!(vi, vns[1], "del")
            f = (vn, dist) -> spl isa SampleFromUniform ? init(dist) : rand(dist)
            r = f.(vns, dists)
            for i in eachindex(vns)
                vn = vns[i]
                dist = dists isa AbstractArray ? dists[i] : dists
                vi[vn] = vectorize(dist, r[i])
                setorder!(vi, vn, vi.num_produce)
            end
        else
        r = reshape(vi[vec(vns)], size(vns))
        end
    else
        f = (vn, dist) -> spl isa SampleFromUniform ? init(dist) : rand(dist)
        r = f.(vns, dists)
        push!.(Ref(vi), vns, r, dists, Ref(spl))
    end
    return r
end

function set_val!(
    vi::VarInfo,
    vns::AbstractVector{<:VarName},
    dist::MultivariateDistribution,
    val::AbstractMatrix,
)
    @assert size(val, 2) == length(vns)
    foreach(enumerate(vns)) do (i, vn)
        vi[vn] = val[:,i]
    end
    return val
end
function set_val!(
    vi::VarInfo,
    vns::AbstractArray{<:VarName},
    dists::Union{Distribution, AbstractArray{<:Distribution}},
    val::AbstractArray,
)
    @assert size(val) == size(vns)
    foreach(CartesianIndices(val)) do ind
        dist = dists isa AbstractArray ? dists[ind] : dists
        vi[vns[ind]] = vectorize(dist, val[ind])
    end
    return val
end

# observe
function dot_tilde(ctx::DefaultContext, sampler, right, left, vi)
    return _dot_tilde(sampler, right, left, vi)
end
function dot_tilde(ctx::PriorContext, sampler, right, left, vi)
    return 0
end
function dot_tilde(ctx::LikelihoodContext, sampler, right, left, vi)
    return _dot_tilde(sampler, right, left, vi)
end
function dot_tilde(ctx::MiniBatchContext, sampler, right, left, vi)
    return ctx.loglike_scalar * dot_tilde(ctx.ctx, sampler, right, left, left, vi)
end

function _dot_tilde(sampler, right, left::AbstractArray, vi)
    return dot_observe(sampler, right, left, vi)
end
# Ambiguity error when not sure to use Distributions convention or Julia broadcasting semantics
function _dot_tilde(
    sampler::AbstractSampler,
    right::Union{MultivariateDistribution, AbstractVector{<:MultivariateDistribution}},
    left::AbstractMatrix{>:AbstractVector},
    vi::VarInfo,
)
    throw(ambiguity_error_msg())
end

function dot_observe(
    spl::Union{SampleFromPrior, SampleFromUniform},
    dist::MultivariateDistribution,
    value::AbstractMatrix,
    vi::VarInfo,
)
    vi.num_produce += one(vi.num_produce)
    Turing.DEBUG && @debug "dist = $dist"
    Turing.DEBUG && @debug "value = $value"
    return sum(logpdf(dist, value))
end
function dot_observe(
    spl::Union{SampleFromPrior, SampleFromUniform},
    dists::Union{Distribution, AbstractArray{<:Distribution}},
    value::AbstractArray,
    vi::VarInfo,
)
    vi.num_produce += one(vi.num_produce)
    Turing.DEBUG && @debug "dists = $dists"
    Turing.DEBUG && @debug "value = $value"
    return sum(logpdf.(dists, value))
end
function dot_observe(
    spl::Sampler,
    ::Any,
    ::Any,
    ::VarInfo,
)
    error("[Turing] $(alg_str(spl)) doesn't support vectorizing observe statement")
end

##############
# Utilities  #
##############

getspace(spl::Sampler) = getspace(spl.alg)
function ambiguity_error_msg()
    return "Ambiguous `lhs .~ rhs` or `@. lhs ~ rhs` syntax. The broadcasting can either be 
    column-wise following the convention of Distributions.jl or element-wise following 
    Julia's general broadcasting semantics. Please make sure that the element type of `lhs` 
    is not a supertype of the support type of `AbstractVector` to eliminate ambiguity."
end

end # module
