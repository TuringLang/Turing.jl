module InterfaceExtensions

using ..Turing
using ..Interface
import MCMCChains: Chains

import ..Interface: AbstractTransition, sample, step!, sample_init!,
    transitions_init, sample_end!, AbstractSampler, transition_type,
    callback, init_callback, AbstractCallback
import ..Turing: Model, Sampler, PROGRESS
import ..RandomVariables: islinked, invlink!, getlogp, tonamedtuple
import ..Inference: InferenceAlgorithm, ParticleInference, AHMC, Hamiltonian,
                    StaticHamiltonian, AdaptiveHamiltonian
import ProgressMeter

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
       Chains,
       initialize_parameters!

# Internal variable names for MCMCChains.
const TURING_INTERNAL_VARS =
    (internals = ["elapsed", "eval_num", "lf_eps", "lp", "weight", "le",
                  "acceptance_rate", "hamiltonian_energy", "is_accept", 
                  "log_density", "n_steps", "numerical_error", "step_size",
                  "tree_depth"],)

####################
# Transition Types #
####################

######################
# Default Transition #
######################

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

#######################
# Particle Transition #
#######################
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

function transition(
    theta::T,
    lp::Float64,
    le::Float64,
    weight::Float64
) where {T}
    return ParticleTransition{T}(theta, lp, le, weight)
end

##########################
# Hamiltonian Transition #
##########################

function transition(spl::Sampler{<:Hamiltonian}, s::SPL) where SPL<:AHMC.Sample
    theta = tonamedtuple(spl.state.vi)
    lp = getlogp(spl.state.vi)
    return HamiltonianTransition{typeof(theta), typeof(s.stat)}(theta, lp, s.stat)
end

struct HamiltonianTransition{T, NT<:NamedTuple} <: AbstractTransition
    θ    :: T
    lp   :: Float64
    stat :: NT
end

transition_type(::Sampler{<:Union{StaticHamiltonian, AdaptiveHamiltonian}}) = 
    HamiltonianTransition

function additional_parameters(::Type{<:HamiltonianTransition})
    return [:lp,:stat]
end

#######################################################
# Special callback functionality for the HMC samplers #
#######################################################

mutable struct HMCCallback{
    ProgType<:ProgressMeter.AbstractProgress
} <: AbstractCallback
    p :: ProgType
end


function callback(
    rng::AbstractRNG,
    model::ModelType,
    spl::SamplerType,
    N::Integer,
    iteration::Integer,
    t::HamiltonianTransition,
    cb::HMCCallback;
    kwargs...
) where {
    ModelType<:Sampleable,
    SamplerType<:AbstractSampler
}
    ProgressMeter.next!(cb.p, t.stat, iteration, spl.state.h.metric)
end

function init_callback(
    rng::AbstractRNG,
    model::Model,
    s::Sampler{<:Union{StaticHamiltonian, AdaptiveHamiltonian}},
    N::Integer;
    dt::Real=0.25,
    kwargs...
)
    return HMCCallback(ProgressMeter.Progress(N, dt=dt, desc="Sampling ", barlen=31))
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
    local names
    vals  = Vector{Vector{Float64}}()
    for t in ts
        names, vs = flatten_namedtuple(t.θ)
        push!(vals, vs)
    end
    
    return string.(names), vals
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

function get_transition_extras(ts::Vector{T}) where T<:AbstractTransition
    # Get the extra field names from the sampler state type.
    # This handles things like :lp or :weight.
    extra_params = additional_parameters(T)

    # Get the values of the extra parameters.
    local extra_names
    all_vals = []
    for t in ts
        extra_names = String[]
        vals = []
        for p in extra_params
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

    return extra_names, vcat(all_vals)
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
        deepcopy(TURING_INTERNAL_VARS);
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