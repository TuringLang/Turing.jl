"""
    isgibbscomponent(alg::Union{InferenceAlgorithm, AbstractMCMC.AbstractSampler})

Return a boolean indicating whether `alg` is a valid component for a Gibbs sampler.

Defaults to `false` if no method has been defined for a particular algorithm type.
"""
isgibbscomponent(::InferenceAlgorithm) = false
isgibbscomponent(spl::Sampler) = isgibbscomponent(spl.alg)

isgibbscomponent(::ESS) = true
isgibbscomponent(::HMC) = true
isgibbscomponent(::HMCDA) = true
isgibbscomponent(::NUTS) = true
isgibbscomponent(::MH) = true
isgibbscomponent(::PG) = true

isgibbscomponent(spl::RepeatSampler) = isgibbscomponent(spl.sampler)

isgibbscomponent(spl::ExternalSampler) = isgibbscomponent(spl.sampler)
isgibbscomponent(::AdvancedHMC.HMC) = true
isgibbscomponent(::AdvancedMH.MetropolisHastings) = true

# Basically like a `DynamicPPL.FixedContext` but
# 1. Hijacks the tilde pipeline to fix variables.
# 2. Computes the log-probability of the fixed variables.
#
# Purpose: avoid triggering resampling of variables we're conditioning on.
# - Using standard `DynamicPPL.condition` results in conditioned variables being treated
#   as observations in the truest sense, i.e. we hit `DynamicPPL.tilde_observe`.
# - But `observe` is overloaded by some samplers, e.g. `CSMC`, which can lead to
#   undesirable behavior, e.g. `CSMC` triggering a resampling for every conditioned variable
#   rather than only for the "true" observations.
# - `GibbsContext` allows us to perform conditioning while still hit the `assume` pipeline
#   rather than the `observe` pipeline for the conditioned variables.
"""
    GibbsContext{VNs}(global_varinfo, context)

A context used in the implementation of the Turing.jl Gibbs sampler.

There will be one `GibbsContext` for each iteration of a component sampler.

`VNs` is a a tuple of symbols for `VarName`s that the current component
sampler is sampling. For those `VarName`s, `GibbsContext` will just pass `tilde_assume`
calls to its child context. For other variables, their values will be fixed to the values
they have in `global_varinfo`.

The naive implementation of `GibbsContext` would simply have a field `target_varnames` that
would be a collection of `VarName`s that the current component sampler is sampling. The
reason we instead have a `Tuple` type parameter listing `Symbol`s is to allow
`is_target_varname` to benefit from compile time constant propagation. This is important
for type stability of `tilde_assume`.

# Fields
$(FIELDS)
"""
struct GibbsContext{VNs,GVI<:Ref{<:AbstractVarInfo},Ctx<:DynamicPPL.AbstractContext} <:
       DynamicPPL.AbstractContext
    """
    a `Ref` to the global `AbstractVarInfo` object that holds values for all variables, both
    those fixed and those being sampled. We use a `Ref` because this field may need to be
    updated if new variables are introduced.
    """
    global_varinfo::GVI
    """
    the child context that tilde calls will eventually be passed onto.
    """
    context::Ctx

    function GibbsContext{VNs}(global_varinfo, context) where {VNs}
        if !(DynamicPPL.NodeTrait(context) isa DynamicPPL.IsLeaf)
            error("GibbsContext can only wrap a leaf context, not a $(context).")
        end
        return new{VNs,typeof(global_varinfo),typeof(context)}(global_varinfo, context)
    end

    function GibbsContext(target_varnames, global_varinfo, context)
        if !(DynamicPPL.NodeTrait(context) isa DynamicPPL.IsLeaf)
            error("GibbsContext can only wrap a leaf context, not a $(context).")
        end
        if any(vn -> DynamicPPL.getoptic(vn) != identity, target_varnames)
            msg =
                "All Gibbs target variables must have identity lenses. " *
                "For example, you can't have `@varname(x.a[1])` as a target variable, " *
                "only `@varname(x)`."
            error(msg)
        end
        vn_sym = tuple(unique((DynamicPPL.getsym(vn) for vn in target_varnames))...)
        return new{vn_sym,typeof(global_varinfo),typeof(context)}(global_varinfo, context)
    end
end

function GibbsContext(target_varnames, global_varinfo)
    return GibbsContext(target_varnames, global_varinfo, DynamicPPL.DefaultContext())
end

DynamicPPL.NodeTrait(::GibbsContext) = DynamicPPL.IsParent()
DynamicPPL.childcontext(context::GibbsContext) = context.context
function DynamicPPL.setchildcontext(context::GibbsContext{VNs}, childcontext) where {VNs}
    return GibbsContext{VNs}(Ref(context.global_varinfo[]), childcontext)
end

get_global_varinfo(context::GibbsContext) = context.global_varinfo[]

function set_global_varinfo!(context::GibbsContext, new_global_varinfo)
    context.global_varinfo[] = new_global_varinfo
    return nothing
end

# has and get
function has_conditioned_gibbs(context::GibbsContext, vn::VarName)
    return DynamicPPL.haskey(get_global_varinfo(context), vn)
end
function has_conditioned_gibbs(context::GibbsContext, vns::AbstractArray{<:VarName})
    num_conditioned = count(Iterators.map(Base.Fix1(has_conditioned_gibbs, context), vns))
    if (num_conditioned != 0) && (num_conditioned != length(vns))
        error(
            "Some but not all of the variables in `vns` have been conditioned on. " *
            "Having mixed conditioning like this is not supported in GibbsContext.",
        )
    end
    return num_conditioned > 0
end

function get_conditioned_gibbs(context::GibbsContext, vn::VarName)
    return get_global_varinfo(context)[vn]
end
function get_conditioned_gibbs(context::GibbsContext, vns::AbstractArray{<:VarName})
    return map(Base.Fix1(get_conditioned_gibbs, context), vns)
end

is_target_varname(::GibbsContext{VNs}, ::VarName{sym}) where {VNs,sym} = sym in VNs

function is_target_varname(context::GibbsContext, vns::AbstractArray{<:VarName})
    num_target = count(Iterators.map(Base.Fix1(is_target_varname, context), vns))
    if (num_target != 0) && (num_target != length(vns))
        error(
            "Some but not all of the variables in `vns` are target variables. " *
            "Having mixed targeting like this is not supported in GibbsContext.",
        )
    end
    return num_target > 0
end

# Tilde pipeline
function DynamicPPL.tilde_assume(context::GibbsContext, right, vn, vi)
    child_context = DynamicPPL.childcontext(context)
    return if is_target_varname(context, vn)
        # Fall back to the default behavior.
        DynamicPPL.tilde_assume(child_context, right, vn, vi)
    elseif has_conditioned_gibbs(context, vn)
        # Short-circuit the tilde assume if `vn` is present in `context`.
        value, lp, _ = DynamicPPL.tilde_assume(
            child_context, right, vn, get_global_varinfo(context)
        )
        value, lp, vi
    else
        # If the varname has not been conditioned on, nor is it a target variable, its
        # presumably a new variable that should be sampled from its prior. We need to add
        # this new variable to the global `varinfo` of the context, but not to the local one
        # being used by the current sampler.
        value, lp, new_global_vi = DynamicPPL.tilde_assume(
            child_context,
            DynamicPPL.SampleFromPrior(),
            right,
            vn,
            get_global_varinfo(context),
        )
        set_global_varinfo!(context, new_global_vi)
        value, lp, vi
    end
end

# As above but with an RNG.
function DynamicPPL.tilde_assume(
    rng::Random.AbstractRNG, context::GibbsContext, sampler, right, vn, vi
)
    # See comment in the above, rng-less version of this method for an explanation.
    child_context = DynamicPPL.childcontext(context)
    return if is_target_varname(context, vn)
        DynamicPPL.tilde_assume(rng, child_context, sampler, right, vn, vi)
    elseif has_conditioned_gibbs(context, vn)
        value, lp, _ = DynamicPPL.tilde_assume(
            child_context, right, vn, get_global_varinfo(context)
        )
        value, lp, vi
    else
        value, lp, new_global_vi = DynamicPPL.tilde_assume(
            rng,
            child_context,
            DynamicPPL.SampleFromPrior(),
            right,
            vn,
            get_global_varinfo(context),
        )
        set_global_varinfo!(context, new_global_vi)
        value, lp, vi
    end
end

# Like the above tilde_assume methods, but with dot_tilde_assume and broadcasting of logpdf.
# See comments there for more details.
function DynamicPPL.dot_tilde_assume(context::GibbsContext, right, left, vns, vi)
    child_context = DynamicPPL.childcontext(context)
    return if is_target_varname(context, vns)
        DynamicPPL.dot_tilde_assume(child_context, right, left, vns, vi)
    elseif has_conditioned_gibbs(context, vns)
        value, lp, _ = DynamicPPL.dot_tilde_assume(
            child_context, right, left, vns, get_global_varinfo(context)
        )
        value, lp, vi
    else
        value, lp, new_global_vi = DynamicPPL.dot_tilde_assume(
            child_context,
            DynamicPPL.SampleFromPrior(),
            right,
            left,
            vns,
            get_global_varinfo(context),
        )
        set_global_varinfo!(context, new_global_vi)
        value, lp, vi
    end
end

# As above but with an RNG.
function DynamicPPL.dot_tilde_assume(
    rng::Random.AbstractRNG, context::GibbsContext, sampler, right, left, vns, vi
)
    child_context = DynamicPPL.childcontext(context)
    return if is_target_varname(context, vns)
        DynamicPPL.dot_tilde_assume(rng, child_context, sampler, right, left, vns, vi)
    elseif has_conditioned_gibbs(context, vns)
        value, lp, _ = DynamicPPL.dot_tilde_assume(
            child_context, right, left, vns, get_global_varinfo(context)
        )
        value, lp, vi
    else
        value, lp, new_global_vi = DynamicPPL.dot_tilde_assume(
            rng,
            child_context,
            DynamicPPL.SampleFromPrior(),
            right,
            left,
            vns,
            get_global_varinfo(context),
        )
        set_global_varinfo!(context, new_global_vi)
        value, lp, vi
    end
end

"""
    make_conditional(model, target_variables, varinfo)

Return a new, conditioned model for a component of a Gibbs sampler.

# Arguments
- `model::DynamicPPL.Model`: The model to condition.
- `target_variables::AbstractVector{<:VarName}`: The target variables of the component
sampler. These will _not_ be conditioned.
- `varinfo::DynamicPPL.AbstractVarInfo`: Values for all variables in the model. All the
values in `varinfo` but not in `target_variables` will be conditioned to the values they
have in `varinfo`.

# Returns
- A new model with the variables _not_ in `target_variables` conditioned.
- The `GibbsContext` object that will be used to condition the variables. This is necessary
because evaluation can mutate its `global_varinfo` field, which we need to access later.
"""
function make_conditional(
    model::DynamicPPL.Model, target_variables::AbstractVector{<:VarName}, varinfo
)
    # Insert the `GibbsContext` just before the leaf.
    # 1. Extract the `leafcontext` from `model` and wrap in `GibbsContext`.
    gibbs_context_inner = GibbsContext(
        target_variables, Ref(varinfo), DynamicPPL.leafcontext(model.context)
    )
    # 2. Set the leaf context to be the `GibbsContext` wrapping `leafcontext(model.context)`.
    gibbs_context = DynamicPPL.setleafcontext(model.context, gibbs_context_inner)
    return DynamicPPL.contextualize(model, gibbs_context), gibbs_context_inner
end

# All samplers are given the same Selector, so that they will sample all variables
# given to them by the Gibbs sampler. This avoids conflicts between the new and the old way
# of choosing which sampler to use.
function set_selector(x::DynamicPPL.Sampler)
    return DynamicPPL.Sampler(x.alg, DynamicPPL.Selector(0))
end
function set_selector(x::RepeatSampler)
    return RepeatSampler(set_selector(x.sampler), x.num_repeat)
end
set_selector(x::InferenceAlgorithm) = DynamicPPL.Sampler(x, DynamicPPL.Selector(0))

to_varname(vn::VarName) = vn
to_varname(s::Symbol) = VarName(s)

to_varname_list(x::Union{VarName,Symbol}) = [to_varname(x)]
# Any other value is assumed to be an iterable of VarNames and Symbols.
to_varname_list(t) = collect(map(to_varname, t))

"""
    Gibbs

A type representing a Gibbs sampler.

# Constructors

`Gibbs` needs to be given a set of pairs of variable names and samplers. Instead of a single
variable name per sampler, one can also give an iterable of variables, all of which are
sampled by the same component sampler.

Each variable name can be given as either a `Symbol` or a `VarName`.

Some examples of valid constructors are:
```julia
Gibbs(:x => NUTS(), :y => MH())
Gibbs(@varname(x) => NUTS(), @varname(y) => MH())
Gibbs((@varname(x), :y) => NUTS(), :z => MH())
```

Currently only variable names without indexing are supported, so for instance
`Gibbs(@varname(x[1]) => NUTS())` does not work. This will hopefully change in the future.

# Fields
$(TYPEDFIELDS)
"""
struct Gibbs{V<:AbstractVector{<:AbstractVector{<:VarName}},A<:AbstractVector} <:
       InferenceAlgorithm
    "varnames representing variables for each sampler"
    varnames::V
    "samplers for each entry in `varnames`"
    samplers::A

    function Gibbs(varnames, samplers)
        if length(varnames) != length(samplers)
            throw(ArgumentError("Number of varnames and samplers must match."))
        end

        for spl in samplers
            if !isgibbscomponent(spl)
                msg = "All samplers must be valid Gibbs components, $(spl) is not."
                throw(ArgumentError(msg))
            end
        end

        # Ensure that samplers have the same selector, and that varnames are lists of
        # VarNames.
        samplers = map(set_selector ∘ drop_space, samplers)
        varnames = map(to_varname_list, varnames)
        return new{typeof(varnames),typeof(samplers)}(varnames, samplers)
    end
end

function Gibbs(algs::Pair...)
    return Gibbs(map(first, algs), map(last, algs))
end

# The below two constructors only provide backwards compatibility with the constructor of
# the old Gibbs sampler. They are deprecated and will be removed in the future.
function Gibbs(alg1::InferenceAlgorithm, other_algs::InferenceAlgorithm...)
    algs = [alg1, other_algs...]
    varnames = map(algs) do alg
        space = getspace(alg)
        if (space isa VarName)
            space
        elseif (space isa Symbol)
            VarName{space}()
        else
            tuple((s isa Symbol ? VarName{s}() : s for s in space)...)
        end
    end
    msg = (
        "Specifying which sampler to use with which variable using syntax like " *
        "`Gibbs(NUTS(:x), MH(:y))` is deprecated and will be removed in the future. " *
        "Please use `Gibbs(; x=NUTS(), y=MH())` instead. If you want different iteration " *
        "counts for different subsamplers, use e.g. " *
        "`Gibbs(@varname(x) => RepeatSampler(NUTS(), 2), @varname(y) => MH())`"
    )
    Base.depwarn(msg, :Gibbs)
    return Gibbs(varnames, map(set_selector ∘ drop_space, algs))
end

function Gibbs(algs_with_iters::Tuple{<:InferenceAlgorithm,Int}...)
    algs = Iterators.map(first, algs_with_iters)
    iters = Iterators.map(last, algs_with_iters)
    algs_duplicated = Iterators.flatten((
        Iterators.repeated(alg, iter) for (alg, iter) in zip(algs, iters)
    ))
    # This calls the other deprecated constructor from above, hence no need for a depwarn
    # here.
    return Gibbs(algs_duplicated...)
end

# TODO: Remove when no longer needed.
DynamicPPL.getspace(::Gibbs) = ()

struct GibbsState{V<:DynamicPPL.AbstractVarInfo,S}
    vi::V
    states::S
end

varinfo(state::GibbsState) = state.vi

function DynamicPPL.initialstep(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::DynamicPPL.Sampler{<:Gibbs},
    vi_base::DynamicPPL.AbstractVarInfo;
    initial_params=nothing,
    kwargs...,
)
    alg = spl.alg
    varnames = alg.varnames
    samplers = alg.samplers

    # Run the model once to get the varnames present + initial values to condition on.
    vi = DynamicPPL.VarInfo(rng, model)
    if initial_params !== nothing
        vi = DynamicPPL.unflatten(vi, initial_params)
    end

    # Initialise each component sampler in turn, collect all their states.
    states = []
    for (varnames_local, sampler_local) in zip(varnames, samplers)
        # Get the initial values for this component sampler.
        initial_params_local = if initial_params === nothing
            nothing
        else
            DynamicPPL.subset(vi, varnames_local)[:]
        end

        # Construct the conditioned model.
        model_local, context_local = make_conditional(model, varnames_local, vi)

        # Take initial step.
        _, new_state_local = AbstractMCMC.step(
            rng,
            model_local,
            sampler_local;
            # FIXME: This will cause issues if the sampler expects initial params in unconstrained space.
            # This is not the case for any samplers in Turing.jl, but will be for external samplers, etc.
            initial_params=initial_params_local,
            kwargs...,
        )
        new_vi_local = varinfo(new_state_local)
        # Merge in any new variables that were introduced during the step, but that
        # were not in the domain of the current sampler.
        vi = merge(vi, get_global_varinfo(context_local))
        # Merge the new values for all the variables sampled by the current sampler.
        vi = merge(vi, new_vi_local)
        push!(states, new_state_local)
    end
    return Transition(model, vi), GibbsState(vi, states)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::DynamicPPL.Sampler{<:Gibbs},
    state::GibbsState;
    kwargs...,
)
    vi = varinfo(state)
    alg = spl.alg
    varnames = alg.varnames
    samplers = alg.samplers
    states = state.states
    @assert length(samplers) == length(state.states)

    # TODO: move this into a recursive function so we can unroll when reasonable?
    for index in 1:length(samplers)
        # Take the inner step.
        sampler_local = samplers[index]
        state_local = states[index]
        varnames_local = varnames[index]
        vi, new_state_local = gibbs_step_inner(
            rng, model, varnames_local, sampler_local, state_local, vi; kwargs...
        )
        states = Accessors.setindex(states, new_state_local, index)
    end
    return Transition(model, vi), GibbsState(vi, states)
end

"""
    setparams_varinfo!!(model, sampler::Sampler, state, params::AbstractVarInfo)

A lot like AbstractMCMC.setparams!!, but instead of taking a vector of parameters, takes an
`AbstractVarInfo` object. Also takes the `sampler` as an argument. By default, falls back to
`AbstractMCMC.setparams!!(model, state, params[:])`.

`model` is typically a `DynamicPPL.Model`, but can also be e.g. an
`AbstractMCMC.LogDensityModel`.
"""
function setparams_varinfo!!(model, ::Sampler, state, params::AbstractVarInfo)
    return AbstractMCMC.setparams!!(model, state, params[:])
end

function setparams_varinfo!!(
    model::DynamicPPL.Model,
    sampler::Sampler{<:MH},
    state::AbstractVarInfo,
    params::AbstractVarInfo,
)
    # The state is already a VarInfo, so we can just return `params`, but first we need to
    # update its logprob.
    # NOTE: Using `leafcontext(model.context)` here is a no-op, as it will be concatenated
    # with `model.context` before hitting `model.f`.
    return last(DynamicPPL.evaluate!!(model, params, DynamicPPL.leafcontext(model.context)))
end

function setparams_varinfo!!(
    model::DynamicPPL.Model,
    sampler::Sampler{<:ESS},
    state::AbstractVarInfo,
    params::AbstractVarInfo,
)
    # The state is already a VarInfo, so we can just return `params`, but first we need to
    # update its logprob. To do this, we have to call evaluate!! with the sampler, rather
    # than just a context, because ESS is peculiar in how it uses LikelihoodContext for
    # some variables and DefaultContext for others.
    return last(DynamicPPL.evaluate!!(model, params, SamplingContext(sampler)))
end

function setparams_varinfo!!(
    model::DynamicPPL.Model,
    sampler::Sampler{<:ExternalSampler},
    state::TuringState,
    params::AbstractVarInfo,
)
    logdensity = DynamicPPL.setmodel(state.logdensity, model, sampler.alg.adtype)
    new_inner_state = setparams_varinfo!!(
        AbstractMCMC.LogDensityModel(logdensity), sampler, state.state, params
    )
    return TuringState(new_inner_state, logdensity)
end

function setparams_varinfo!!(
    model::DynamicPPL.Model,
    sampler::Sampler{<:Hamiltonian},
    state::HMCState,
    params::AbstractVarInfo,
)
    θ_new = params[:]
    hamiltonian = get_hamiltonian(model, sampler, params, state, length(θ_new))

    # Update the parameter values in `state.z`.
    # TODO: Avoid mutation
    z = state.z
    resize!(z.θ, length(θ_new))
    z.θ .= θ_new
    return HMCState(params, state.i, state.kernel, hamiltonian, z, state.adaptor)
end

function setparams_varinfo!!(
    model::DynamicPPL.Model, sampler::Sampler{<:PG}, state::PGState, params::AbstractVarInfo
)
    return PGState(params, state.rng)
end

"""
    match_linking!!(varinfo_local, prev_state_local, model)

Make sure the linked/invlinked status of varinfo_local matches that of the previous
state for this sampler. This is relevant when multilple samplers are sampling the same
variables, and one might need it to be linked while the other doesn't.
"""
function match_linking!!(varinfo_local, prev_state_local, model)
    prev_varinfo_local = varinfo(prev_state_local)
    was_linked = DynamicPPL.istrans(prev_varinfo_local)
    is_linked = DynamicPPL.istrans(varinfo_local)
    if was_linked && !is_linked
        varinfo_local = DynamicPPL.link!!(varinfo_local, model)
    elseif !was_linked && is_linked
        varinfo_local = DynamicPPL.invlink!!(varinfo_local, model)
    end
    # TODO(mhauru) The above might run into trouble if some variables are linked and others
    # are not. `istrans(varinfo)` returns an `all` over the individual variables. This could
    # especially be a problem with dynamic models, where new variables may get introduced,
    # but also in cases where component samplers have partial overlap in their target
    # variables. The below is how I would like to implement this, but DynamicPPL at this
    # time does not support linking individual variables selected by `VarName`. It soon
    # should though, so come back to this.
    # Issue ref: https://github.com/TuringLang/Turing.jl/issues/2401
    # prev_links_dict = Dict(vn => DynamicPPL.istrans(prev_varinfo_local, vn) for vn in keys(prev_varinfo_local))
    # any_linked = any(values(prev_links_dict))
    # for vn in keys(varinfo_local)
    #     was_linked = if haskey(prev_varinfo_local, vn)
    #         prev_links_dict[vn]
    #     else
    #         # If the old state didn't have this variable, we assume it was linked if _any_
    #         # of the variables of the old state were linked.
    #         any_linked
    #     end
    #     is_linked = DynamicPPL.istrans(varinfo_local, vn)
    #     if was_linked && !is_linked
    #         varinfo_local = DynamicPPL.invlink!!(varinfo_local, vn)
    #     elseif !was_linked && is_linked
    #         varinfo_local = DynamicPPL.link!!(varinfo_local, vn)
    #     end
    # end
    return varinfo_local
end

function gibbs_step_inner(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    varnames_local,
    sampler_local,
    state_local,
    global_vi;
    kwargs...,
)
    # Construct the conditional model and the varinfo that this sampler should use.
    model_local, context_local = make_conditional(model, varnames_local, global_vi)
    varinfo_local = subset(global_vi, varnames_local)
    varinfo_local = match_linking!!(varinfo_local, state_local, model)

    # TODO(mhauru) The below may be overkill. If the varnames for this sampler are not
    # sampled by other samplers, we don't need to `setparams`, but could rather simply
    # recompute the log probability. More over, in some cases the recomputation could also
    # be avoided, if e.g. the previous sampler has done all the necessary work already.
    # However, we've judged that doing any caching or other tricks to avoid this now would
    # be premature optimization. In most use cases of Gibbs a single model call here is not
    # going to be a significant expense anyway.
    # Set the state of the current sampler, accounting for any changes made by other
    # samplers.
    state_local = setparams_varinfo!!(
        model_local, sampler_local, state_local, varinfo_local
    )

    # Take a step with the local sampler.
    new_state_local = last(
        AbstractMCMC.step(rng, model_local, sampler_local, state_local; kwargs...)
    )

    new_vi_local = varinfo(new_state_local)
    # Merge the latest values for all the variables in the current sampler.
    new_global_vi = merge(get_global_varinfo(context_local), new_vi_local)
    new_global_vi = setlogp!!(new_global_vi, getlogp(new_vi_local))
    return new_global_vi, new_state_local
end
