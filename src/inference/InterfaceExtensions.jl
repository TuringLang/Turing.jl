module InterfaceExtensions

using ..Turing
using ..Interface
import MCMCChains: Chains

import ..Interface: AbstractTransition, sample, step!, sample_init!,
    transitions_init, sample_end!, AbstractSampler, transition_type
import ..Turing: Model, Sampler
import ..RandomVariables: islinked, invlink!, getlogp, tonamedtuple
import ..Inference: InferenceAlgorithm, ParticleInference

export sample,
       sample_init!,
       sample_end!,
       set_resume!,
       resume,
       save,
       transition_type,
       transitions_init,
       transition,
       Transition,
       ParticleTransition,
       Chains

# Internal variable names for MCMCChains.
const TURING_INTERNAL_VARS =
    Dict(:internals => ["elapsed", "eval_num", "lf_eps", "lp", "weight", "le",
                        "acceptance_rate", "hamiltonian_energy", "is_accept", 
                        "log_density", "n_steps", "numerical_error", "step_size",
                        "tree_depth"])

####################
# Transition Types #
####################

# Used by all non-particle samplers.
struct Transition{T} <: AbstractTransition
    θ  :: T
    lp :: Float64
end

function transition(spl::Sampler)
    theta = tonamedtuple(spl.state.vi)
    lp = getlogp(spl.state.vi)
    return Transition{typeof(theta)}(theta, lp)
end

function transition(spl::Sampler, nt::NamedTuple)
    theta = merge(tonamedtuple(spl.state.vi), nt)
    lp = getlogp(spl.state.vi)
    return Transition{typeof(theta)}(theta, lp)
end

function additional_parameters(::Type{Transition})
    return [:lp]
end

# used by PG and SMC
struct ParticleTransition{T} <: AbstractTransition
    θ::T
    lp::Float64
    le::Float64
    weight::Float64
end

transition_type(::Sampler{<:ParticleInference}) = ParticleTransition

function additional_parameters(::Type{<:ParticleTransition})
    return [:lp,:le, :weight]
end

"""
    transition(vi::AbstractVarInfo, spl::Sampler{<:Union{SMC, PG}}, weight::Float64)

Returns a TransitionType for the particle samplers.
"""
function transition(
        theta::T,
        lp::Float64,
        le::Float64,
        weight::Float64
) where {T}
    return ParticleTransition{T}(theta, lp, le, weight)
end


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
    return sample(rng, model, Sampler(alg, model), N; kwargs...)
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
        return sample(model, Sampler(alg, model), N; kwargs...)
    else
        return resume(resume_from, N)
    end
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

##########################
# Chain making utilities #
##########################

function _params_to_array(ts::Vector{T}, spl::Sampler) where {T<:Union{ParticleTransition,Transition}}
    local names
    vals  = Vector{Vector{Float64}}()
    for t in ts
        names, vs = flatten_namedtuple(t.θ)
        push!(vals, vs)
    end
    
    return names, vals
end

function flatten_namedtuple(nt::NamedTuple{pnames}) where {pnames}
    vals  = Vector{Float64}()
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
                push!(value, Float64(v[i]))
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

    # Get the extra field names from the sampler state type.
    # This handles things like :lp or :weight.
    extra_params = additional_parameters(T)

    # Get the values of the extra parameters.
    extra_values = vcat(map(t -> [getproperty(t, p) for p in extra_params], ts))

    # Extract names & construct param array.
    nms = string.(vcat(nms..., string.(extra_params)...))
    parray = vcat([hcat(vals[i]..., extra_values[i]...) for i in 1:length(ts)]...)

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
        convert(Array{Real}, parray),
        string.(nms),
        TURING_INTERNAL_VARS;
        evidence=le,
        info=info
    )
end

Interface.transition_type(::Sampler{alg}) where alg = transition_type(alg)

#########
# Chain #
#########

# ind2sub is deprecated in Julia 1.0
ind2sub(v, i) = Tuple(CartesianIndices(v)[i])

function save(c::Chains, spl::AbstractSampler, model, vi, samples)
    nt = NamedTuple{(:spl, :model, :vi, :samples)}((spl, model, deepcopy(vi), samples))
    return setinfo(c, merge(nt, c.info))
end

function resume(c::Chains, n_iter::Int)
    @assert !isempty(c.info) "[Turing] cannot resume from a chain without state info"
    return sample(
        c.info[:range],
        c.info[:model],
        c.info[:spl],
        n_iter;    # this is actually not used
        resume_from=c,
        reuse_spl_n=n_iter
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

end # module