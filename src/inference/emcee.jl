###
### Sampler states
###

struct Emcee{space, P} <: InferenceAlgorithm 
    ensemble::AMH.Ensemble{P}
end

function Emcee(n_walkers::Int, d::MvNormal, stretch_length=2.0)
    prop = AMH.StretchProposal(d, stretch_length)
    ensemble = AMH.Ensemble(n_walkers, prop)
    return Emcee{(), typeof(prop)}(ensemble)
end

function Sampler(
    alg::Emcee,
    model::Model,
    s::Selector=Selector()
)
    # Set up info dict.
    info = Dict{Symbol, Any}()

    # Set up state struct.
    state = SamplerState(VarInfo(model))

    # Generate a sampler.
    return Sampler(alg, info, s, state)
end

alg_str(::Sampler{<:Emcee}) = "Emcee"

function AbstractMCMC.sample_init!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:Emcee},
    N::Integer;
    verbose::Bool=true,
    resume_from=nothing,
    kwargs...
)
    # Resume the sampler.
    set_resume!(spl; resume_from=resume_from, kwargs...)

    # Get `init_theta`
    initialize_parameters!(spl; verbose=verbose, kwargs...)

    # If we're doing random walk with a covariance matrix,
    # just link everything before sampling.
    link!(spl.state.vi, spl)
end

function AbstractMCMC.sample_end!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:Emcee},
    N::Integer,
    transitions;
    kwargs...
)
    # We are doing a random walk, so we unlink everything when we're done.
    invlink!(spl.state.vi, spl)
end

function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:Emcee},
    N::Integer,
    transition::Nothing;
    kwargs...
)
    # Generate a log joint function.
    densitymodel = AMH.DensityModel(gen_logπ(spl.state.vi, spl, model))

    # Make the first transition.
    transition = AbstractMCMC.step!(rng, densitymodel, spl.alg.ensemble, 1)

    return transition
end

function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:Emcee},
    N::Integer,
    transition;
    kwargs...
)
    # Generate a log joint function.
    densitymodel = AMH.DensityModel(gen_logπ(spl.state.vi, spl, model))

    # Make the first transition.
    transition = AbstractMCMC.step!(rng, densitymodel, spl.alg.ensemble, 1, transition)

    return transition
end

function transform_transition(spl::Sampler{<:Emcee}, ts, t::Int, i::Int)
    trans = ts[i][t]
    DynamicPPL.link!(spl.state.vi, spl)
    spl.state.vi[spl] = trans.params
    DynamicPPL.invlink!(spl.state.vi, spl)
    setlogp!(spl.state.vi, trans.lp)

    if t == 1
        println()
        println(trans.params)
        println(spl.state.vi[spl])
    end

    return Transition(spl, (walker_id=t, iteration=i))
end

function AbstractMCMC.bundle_samples(
    rng::AbstractRNG,
    model::AbstractModel,
    spl::Sampler{<:Emcee},
    N::Integer,
    ts::Vector,
    chain_type::Type{MCMCChains.Chains};
    save_state = false,
    kwargs...
)
    # Transform the transitions.
    ts_transform = mapreduce(
        i -> map(t -> transform_transition(spl, ts, t, i), 1:spl.alg.ensemble.n_walkers),
        vcat,
        1:length(ts)
    )

    # Convert transitions to array format.
    # Also retrieve the variable names.
    nms, vals = _params_to_array(ts_transform)

    # Get the values of the extra parameters in each transition.
    extra_params, extra_values = get_transition_extras(ts_transform)
    
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


####
#### Compiler interface, i.e. tilde operators.
####
function DynamicPPL.assume(
    spl::Sampler{<:Emcee},
    dist::Distribution,
    vn::VarName,
    vi,
)
    updategid!(vi, vn, spl)
    r = vi[vn]
    return r, logpdf_with_trans(dist, r, istrans(vi, vn))
end

function DynamicPPL.dot_assume(
    spl::Sampler{<:Emcee},
    dist::MultivariateDistribution,
    vn::VarName,
    var::AbstractMatrix,
    vi,
)
    @assert dim(dist) == size(var, 1)
    getvn = i -> VarName(vn, vn.indexing * "[:,$i]")
    vns = getvn.(1:size(var, 2))
    updategid!.(Ref(vi), vns, Ref(spl))
    r = vi[vns]
    var .= r
    return var, sum(logpdf_with_trans(dist, r, istrans(vi, vns[1])))
end
function DynamicPPL.dot_assume(
    spl::Sampler{<:Emcee},
    dists::Union{Distribution, AbstractArray{<:Distribution}},
    vn::VarName,
    var::AbstractArray,
    vi,
)
    getvn = ind -> VarName(vn, vn.indexing * "[" * join(Tuple(ind), ",") * "]")
    vns = getvn.(CartesianIndices(var))
    updategid!.(Ref(vi), vns, Ref(spl))
    r = reshape(vi[vec(vns)], size(var))
    var .= r
    return var, sum(logpdf_with_trans.(dists, r, istrans(vi, vns[1])))
end

function DynamicPPL.observe(
    spl::Sampler{<:Emcee},
    d::Distribution,
    value,
    vi,
)
    return DynamicPPL.observe(SampleFromPrior(), d, value, vi)
end

function DynamicPPL.dot_observe(
    spl::Sampler{<:Emcee},
    ds::Union{Distribution, AbstractArray{<:Distribution}},
    value::AbstractArray,
    vi,
)
    return DynamicPPL.dot_observe(SampleFromPrior(), ds, value, vi)
end
