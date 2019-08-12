
# Internal variables for MCMCChains.
const TURING_INTERNAL_VARS =
    Dict(:internals => ["elapsed", "eval_num", "lf_eps", "lp", "weight", "le",
                        "acceptance_rate", "hamiltonian_energy", "is_accept", 
                        "log_density", "n_steps", "numerical_error", "step_size",
                        "tree_depth"])

#########################################
# Default definitions for the interface #
#########################################
function sample(
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

function sample(
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

function sample_init!(
    ::AbstractRNG,
    model::Model,
    spl::Sampler,
    N::Integer;
    kwargs...
)
    # Resume the sampler.
    set_resume!(spl; kwargs...)
end

function sample_end!(
    ::AbstractRNG,
    ::Model,
    spl::AbstractSampler,
    ::Integer,
    ::Vector{TransitionType};
    kwargs...
) where {TransitionType<:AbstractTransition}
    # Silence the default API function.
end

"""
    transitions_init(
        ::AbstractRNG,
        model::Model,
        spl::Sampler,
        N::Integer;
        kwargs...
    )

Create a vector of `Transition` structs of length `N`.
"""
function transitions_init(
    ::AbstractRNG,
    model::Model,
    spl::Sampler,
    N::Integer;
    kwargs...
)
    ttype = transition_type(spl)
    return Vector{ttype}(undef, N)
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

function flatten(names, value :: Array{Float64}, k :: String, v)
    if isa(v, Number)
        name = k
        push!(value, v)
        push!(names, name)
    elseif isa(v, Array)
        for i = eachindex(v)
            if isa(v[i], Number)
                name = string(Turing.Utilities.ind2sub(size(v), i))
                name = replace(name, "(" => "[");
                name = replace(name, ",)" => "]");
                name = replace(name, ")" => "]");
                name = k * name
                isa(v[i], Nothing) && println(v, i, v[i])
                push!(value, Float64(v[i]))
                push!(names, name)
            elseif isa(v[i], Array)
                name = k * string(Turing.Utilities.ind2sub(size(v), i))
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
    # RandomVariables.params_nt(spl.state.vi, spl)
    
    nms, vals = _params_to_array(ts, spl)

    # Get the extra field names from the sampler state type.
    # This handles things like :lp or :weight.
    extra_params = additional_parameters(T)

    # Get the values of the extra parameters.
    extra_values = vcat(map(t -> [getproperty(t, p) for p in extra_params], ts))

    # Extract names & construct param array.
    nms = string.(vcat(nms..., string.(extra_params)...))
    parray = vcat([hcat(vals[i]..., extra_values[i]...) for i in 1:length(ts)]...)

    # If the state field has final_logevidence, grab that.
    le = :final_logevidence in fieldnames(typeof(spl.state)) ?
        getproperty(spl.state, :final_logevidence) :
        missing

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

transition_type(::Sampler{alg}) where alg = transition_type(alg)
