module Inference

using ..Core, ..Core.RandomVariables, ..Utilities
using ..Core.RandomVariables: Metadata, _tail, TypedVarInfo, 
    islinked, invlink!, getlogp, tonamedtuple
using Distributions, Libtask, Bijectors
using ProgressMeter, LinearAlgebra
using ..Turing: PROGRESS, CACHERESET, AbstractSampler
using ..Turing: Model, runmodel!, get_pvars, get_dvars,
    Sampler, SampleFromPrior, SampleFromUniform,
    Selector, AbstractSamplerState
using ..Turing: in_pvars, in_dvars, Turing
using StatsFuns: logsumexp
using Random: GLOBAL_RNG, AbstractRNG
using ..Turing.Interface

import MCMCChains: Chains
import AdvancedHMC; const AHMC = AdvancedHMC
import ..Turing: getspace
import Distributions: sample
import ..Core: getchunksize, getADtype
import ..Interface: AbstractTransition, sample, step!, sample_init!,
    transitions_init, sample_end!, AbstractSampler, transition_type,
    callback, init_callback, AbstractCallback, psample

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
        reset!,
        step!

#######################
# Sampler abstraction #
#######################
abstract type AbstractAdapter end
abstract type InferenceAlgorithm end
abstract type ParticleInference <: InferenceAlgorithm end
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

Interface.transition_type(::Sampler{alg}) where alg = transition_type(alg)

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

function Interface.sample(
    rng::AbstractRNG,
    model::ModelType,
    alg::AlgType,
    N::Integer;
    kwargs...
) where {
    ModelType<:Sampleable,
    SamplerType<:AbstractSampler,
    AlgType<:InferenceAlgorithm
}
    return sample(rng, model, Sampler(alg, model), N; progress=PROGRESS[], kwargs...)
end

function Interface.sample(
    model::ModelType,
    alg::AlgType,
    N::Integer;
    resume_from=nothing,
    kwargs...
) where {
    ModelType<:Sampleable,
    SamplerType<:AbstractSampler,
    AlgType<:InferenceAlgorithm
}
    if resume_from === nothing
        return sample(model, Sampler(alg, model), N; progress=PROGRESS[], kwargs...)
    else
        return resume(resume_from, N)
    end
end


function Interface.psample(
    model::ModelType,
    alg::AlgType,
    N::Integer,
    n_chains::Integer;
    kwargs...
) where {
    ModelType<:Sampleable,
    SamplerType<:AbstractSampler,
    AlgType<:InferenceAlgorithm
}
    return psample(GLOBAL_RNG, model, alg, N, n_chains; progress=false, kwargs...)
end

function Interface.psample(
    rng::AbstractRNG,
    model::ModelType,
    alg::AlgType,
    N::Integer,
    n_chains::Integer;
    kwargs...
) where {
    ModelType<:Sampleable,
    SamplerType<:AbstractSampler,
    AlgType<:InferenceAlgorithm
}
    return psample(rng, model, Sampler(alg, model), N, n_chains; progress=false, kwargs...)
end

function Interface.sample_init!(
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

function Interface.sample_end!(
    ::AbstractRNG,
    ::Model,
    spl::AbstractSampler,
    ::Integer,
    ::Vector{TransitionType};
    kwargs...
) where {TransitionType<:AbstractTransition}
    # Silence the default API function.
end

function initialize_parameters!(
    spl::AbstractSampler;
    init_theta::Union{Nothing,Array{<:Any,1}}=nothing,
    verbose::Bool=false,
    kwargs...
)
    # Get `init_theta`
    if init_theta != nothing
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

function _params_to_array(ts::Vector{T}, spl::Sampler) where {T<:AbstractTransition}
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

function get_transition_extras(ts::Vector{T}) where T<:AbstractTransition
    # Get the extra field names from the sampler state type.
    # This handles things like :lp or :weight.
    extra_params = additional_parameters(T)

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
function Chains(
    rng::AbstractRNG,
    model::ModelType,
    spl::Sampler,
    N::Integer,
    ts::Vector{T};
    discard_adapt::Bool=true,
    save_state=false,
    kwargs...
) where {ModelType<:Sampleable, T<:AbstractTransition}
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
    info = if save_state
        (range = rng,
        model = model,
        spl = spl)
    else
        NamedTuple()
    end

    # Chain construction.
    return Chains(
        parray,
        string.(nms),
        deepcopy(TURING_INTERNAL_VARS);
        evidence=le,
        info=info
    )
end

function save(c::Chains, spl::AbstractSampler, model, vi, samples)
    nt = NamedTuple{(:spl, :model, :vi, :samples)}((spl, model, deepcopy(vi), samples))
    return setinfo(c, merge(nt, c.info))
end

function resume(c::Chains, n_iter::Int; kwargs...)
    @assert !isempty(c.info) "[Turing] cannot resume from a chain without state info"
    return sample(
        c.info[:range],
        c.info[:model],
        c.info[:spl],
        n_iter;    # this is actually not used
        resume_from=c,
        reuse_spl_n=n_iter,
        kwargs...
    )
end

function set_resume!(
    s::Sampler;
    resume_from::Union{Chains, Nothing}=nothing,
    kwargs...
)
    # If we're resuming, grab the sampler info.
    if resume_from != nothing
        s = resume_from.info[:spl]
    end
end

function split_var_str(var_str)
    ind = findfirst(c -> c == '[', var_str)
    inds = Vector{String}[]
    if ind == nothing
        return var_str, inds
    end
    sym = var_str[1:ind-1]
    ind = length(sym)
    while ind < length(var_str)
        ind += 1
        @assert var_str[ind] == '['
        push!(inds, String[])
        while var_str[ind] != ']'
            ind += 1
            if var_str[ind] == '['
                ind2 = findnext(c -> c == ']', var_str, ind)
                push!(inds[end], strip(var_str[ind:ind2]))
                ind = ind2+1
            else
                ind2 = findnext(c -> c == ',' || c == ']', var_str, ind)
                push!(inds[end], strip(var_str[ind:ind2-1]))
                ind = ind2
            end
        end
    end
    return sym, inds
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

for alg in (:SMC, :PG, :PMMH, :IPMCMC, :MH, :IS)
    @eval getspace(::$alg{space}) where {space} = space
    @eval getspace(::Type{<:$alg{space}}) where {space} = space
end
for alg in (:HMC, :HMCDA, :NUTS, :SGLD, :SGHMC)
    @eval getspace(::$alg{<:Any, space}) where {space} = space
    @eval getspace(::Type{<:$alg{<:Any, space}}) where {space} = space
end
getspace(::Gibbs) = Tuple{}()
getspace(::Type{<:Gibbs}) = Tuple{}()

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
    @assert isa(dist, UnivariateDistribution) || 
        isa(dist, MultivariateDistribution) "Turing.observe: vectorizing matrix distribution is not supported"
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

getspace(spl::Sampler) = getspace(typeof(spl))
getspace(::Type{<:Sampler{Talg}}) where {Talg} = getspace(Talg)


end # module
