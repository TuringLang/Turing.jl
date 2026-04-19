###################################################
# Interface for other samplers to work with Gibbs #
###################################################

"""
    isgibbscomponent(spl::AbstractSampler)

Return a boolean indicating whether `spl` is a valid component for a Gibbs sampler.

Defaults to `true` if no method has been defined for a particular sampler.
"""
isgibbscomponent(::AbstractSampler) = true
isgibbscomponent(spl::RepeatSampler) = isgibbscomponent(spl.sampler)
isgibbscomponent(spl::ExternalSampler) = isgibbscomponent(spl.sampler)
isgibbscomponent(::Prior) = false
isgibbscomponent(::Emcee) = false
isgibbscomponent(::SGLD) = false
isgibbscomponent(::SGHMC) = false
isgibbscomponent(::SMC) = false

"""
    Turing.Inference.gibbs_get_raw_values(state)

Return a `VarNamedTuple` containing the raw values of all variables in the sampler state.
"""
function gibbs_get_raw_values(state::AbstractVarInfo)
    return DynamicPPL.get_raw_values(state)
end

"""
    Turing.Inference.gibbs_update_state!!(
        sampler::AbstractSampler, state, model::Model, global_vals::VarNamedTuple
    )

Update the state of a Gibbs component sampler to be consistent with the new values in
`global_vals`. The exact meaning of this depends on what the sampler state contains.

Each sampler should implement a method for its respective state type.
"""
function gibbs_update_state!! end

"""
    gibbs_recompute_ldf_and_params(
        old_ldf, model, vector_vnt, global_vals, extra_accs
    )

Shared helper that is used in `gibbs_update_state!!` for any sampler that uses a
LogDensityFunction.

Creates a new `LogDensityFunction` from the newly conditioned `model` (using a cached
`vector_vnt` to avoid an extra model evaluation), then reevaluates the model to obtain the
correct vectorised parameters corresponding to the raw values in `global_vals`.

If extra information is needed (e.g. log-probabilities), `extra_accs` can be used to pass in
other accumulators to be used in the same model evaluation, to avoid having to recompute
them later.

Returns `(new_ldf, new_params, accs)` where `accs` is the accumulator VarInfo after
evaluation, from which extra accumulators (e.g. `LogLikelihoodAccumulator`) can be read.
"""
function gibbs_recompute_ldf_and_params(
    old_ldf::DynamicPPL.LogDensityFunction,
    model::DynamicPPL.Model,
    vector_vnt::DynamicPPL.VarNamedTuple,
    global_vals::DynamicPPL.VarNamedTuple,
    extra_accs::NTuple{N,<:DynamicPPL.AbstractAccumulator}=(),
) where {N}
    new_ldf = DynamicPPL.LogDensityFunction(
        model, old_ldf._getlogdensity, vector_vnt; adtype=old_ldf.adtype
    )
    accs = DynamicPPL.OnlyAccsVarInfo(
        DynamicPPL.VectorParamAccumulator(new_ldf), extra_accs...
    )
    init_strategy = DynamicPPL.InitFromParams(global_vals, nothing)
    _, accs = DynamicPPL.init!!(
        new_ldf.model, accs, init_strategy, new_ldf.transform_strategy
    )
    new_params = DynamicPPL.get_vector_params(accs)
    return new_ldf, new_params, accs
end

###############################
# Gibbs implementation itself #
###############################

can_be_wrapped(::DynamicPPL.AbstractContext) = true
can_be_wrapped(::DynamicPPL.AbstractParentContext) = false
can_be_wrapped(ctx::DynamicPPL.PrefixContext) = can_be_wrapped(DynamicPPL.childcontext(ctx))

# Basically like a `DynamicPPL.FixedContext` but
# 1. Hijacks the tilde pipeline to fix variables.
# 2. Computes the log-probability of the fixed variables.
#
# Purpose: avoid triggering resampling of variables we're conditioning on.
# - Using standard `DynamicPPL.condition` results in conditioned variables being treated
#   as observations in the truest sense, i.e. we hit `DynamicPPL.tilde_observe!!`.
# - But `observe` is overloaded by some samplers, e.g. `CSMC`, which can lead to
#   undesirable behavior, e.g. `CSMC` triggering a resampling for every conditioned variable
#   rather than only for the "true" observations.
# - `GibbsContext` allows us to perform conditioning while still hit the `assume` pipeline
#   rather than the `observe` pipeline for the conditioned variables.
"""
    GibbsContext(target_varnames, global_vnt, context)

A context used in the implementation of the Turing.jl Gibbs sampler.

There will be one `GibbsContext` for each iteration of a component sampler.

`target_varnames` is a a tuple of `VarName`s that the current component sampler is sampling.
For those `VarName`s, `GibbsContext` will just pass `tilde_assume!!` calls to its child
context. For other variables, their values will be fixed to the values they have in
`global_vnt`.

# Fields
$(FIELDS)
"""
struct GibbsContext{
    VNs<:Tuple{Vararg{VarName}},
    GV<:Ref{<:DynamicPPL.VarNamedTuple},
    Ctx<:DynamicPPL.AbstractContext,
} <: DynamicPPL.AbstractParentContext
    """
    the VarNames being sampled
    """
    target_varnames::VNs
    """
    a `Ref` to the global `VarNamedTuple` object that holds raw values for all variables,
    both those fixed and those being sampled. We use a `Ref` because this field may need
    to be updated if new variables are introduced.
    """
    global_vnt::GV
    """
    the child context that tilde calls will eventually be passed onto.
    """
    context::Ctx

    function GibbsContext(target_varnames, global_vnt, context)
        if !can_be_wrapped(context)
            error("GibbsContext can only wrap a leaf or prefix context, not a $(context).")
        end
        target_varnames = tuple(target_varnames...)  # Allow vectors.
        return new{typeof(target_varnames),typeof(global_vnt),typeof(context)}(
            target_varnames, global_vnt, context
        )
    end
end

function GibbsContext(target_varnames, global_vnt)
    return GibbsContext(target_varnames, global_vnt, DynamicPPL.DefaultContext())
end

DynamicPPL.childcontext(context::GibbsContext) = context.context
function DynamicPPL.setchildcontext(context::GibbsContext, childcontext)
    return GibbsContext(context.target_varnames, context.global_vnt, childcontext)
end

get_global_vnt(context::GibbsContext) = context.global_vnt[]

function set_global_vnt!(context::GibbsContext, new_global_varinfo)
    context.global_vnt[] = new_global_varinfo
    return nothing
end

# has and get
function has_conditioned_gibbs(context::GibbsContext, vn::VarName)
    return DynamicPPL.haskey(get_global_vnt(context), vn)
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
    return get_global_vnt(context)[vn]
end
function get_conditioned_gibbs(context::GibbsContext, vns::AbstractArray{<:VarName})
    return map(Base.Fix1(get_conditioned_gibbs, context), vns)
end

function is_target_varname(ctx::GibbsContext, vn::VarName)
    return any(Base.Fix2(AbstractPPL.subsumes, vn), ctx.target_varnames)
end

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

# Copied from DynamicPPL to avoid having to export
optic_skip_length(::AbstractPPL.Iden) = 0
optic_skip_length(c::AbstractPPL.Index) = 1 + optic_skip_length(c.child)
optic_skip_length(c::AbstractPPL.Property) = 1 + optic_skip_length(c.child)

# Tilde pipeline
function DynamicPPL.tilde_assume!!(
    context::GibbsContext,
    right::Distribution,
    vn::VarName,
    template::Any,
    vi::DynamicPPL.AbstractVarInfo,
)
    child_context = DynamicPPL.childcontext(context)

    # Note that `child_context` may contain `PrefixContext`s -- in which case
    # we need to make sure that vn is appropriately prefixed before we handle
    # the `GibbsContext` behaviour below. For example, consider the following:
    #      @model inner() = x ~ Normal()
    #      @model outer() = a ~ to_submodel(inner())
    # If we run this with `Gibbs(@varname(a.x) => MH())`, then when we are
    # executing the submodel, the `context` will contain the `@varname(a.x)`
    # variable; `child_context` will contain `PrefixContext(@varname(a))`; and
    # `vn` will just be `@varname(x)`. If we just simply run
    # `is_target_varname(context, vn)`, it will return false, and everything
    # will be messed up.
    # TODO(penelopeysm): This 'problem' could be solved if we made GibbsContext a
    # leaf context and wrapped the PrefixContext _above_ the GibbsContext, so
    # that the prefixing would be handled by tilde_assume(::PrefixContext, ...)
    # _before_ we hit this method.
    # In the current state of GibbsContext, doing this would require
    # special-casing the way PrefixContext is used to wrap the leaf context.
    # This is very inconvenient because PrefixContext's behaviour is defined in
    # DynamicPPL, and we would basically have to create a new method in Turing
    # and override it for GibbsContext. Indeed, a better way to do this would
    # be to make GibbsContext a leaf context. In this case, we would be able to
    # rely on the existing behaviour of DynamicPPL.make_evaluate_args_and_kwargs
    # to correctly wrap the PrefixContext around the GibbsContext. This is very
    # tricky to correctly do now, but once we remove the other leaf contexts
    # (i.e. PriorContext and LikelihoodContext), we should be able to do this.
    # This is already implemented in
    # https://github.com/TuringLang/DynamicPPL.jl/pull/885/ but not yet
    # released. Exciting!
    child_context, aggregated_prefixes = DynamicPPL.extract_prefixes(child_context)
    if aggregated_prefixes !== nothing
        vn = AbstractPPL.prefix(vn, aggregated_prefixes)
        n = optic_skip_length(AbstractPPL.getoptic(aggregated_prefixes)) + 1
        template = DynamicPPL.SkipTemplate{n}(template)
    end

    return if is_target_varname(context, vn)
        # Fall back to the default behavior.
        DynamicPPL.tilde_assume!!(child_context, right, vn, template, vi)
    elseif has_conditioned_gibbs(context, vn)
        # This branch means that a different sampler is supposed to handle this
        # variable. From the perspective of this sampler, this variable is
        # conditioned on, so we can just treat it as an observation.
        # The only catch is that the value that we need is to be obtained from
        # the global VNT (since the local VarInfo has no knowledge of it).
        # Note that tilde_observe!! will trigger resampling in particle methods
        # for variables that are handled by other Gibbs component samplers.
        val = get_conditioned_gibbs(context, vn)
        DynamicPPL.tilde_observe!!(child_context, right, val, vn, template, vi)
    else
        # If the varname has not been conditioned on, nor is it a target variable, its
        # presumably a new variable that should be sampled from its prior. We need to add
        # this new variable to the global `varinfo` of the context, but not to the local one
        # being used by the current sampler.
        #
        # TODO(penelopeysm): How is the RNG controlled here?
        value = rand(right)
        vnt = get_global_vnt(context)
        vnt = DynamicPPL.templated_setindex!!(vnt, value, vn, template)
        set_global_vnt!(context, vnt)
        # Return the value (so that it can be used in the model), plus the unmodified local
        # varinfo
        value, vi
    end
end

"""
    make_conditional(model, target_variables, global_vnt)

Return a new, conditioned model for a component of a Gibbs sampler.

# Arguments

- `model::DynamicPPL.Model`: The model to condition.

- `target_variables::AbstractVector{<:VarName}`: The target variables of the component
  sampler. These will _not_ be conditioned.

- `global_vnt::DynamicPPL.VarNamedTuple`: Raw values for all variables in the model, which
  are used for all variables that are *not* in `target_variables`.

# Returns

- A new model with the variables _not_ in `target_variables` conditioned.

- The `GibbsContext` object that will be used to condition the variables. This is necessary
because evaluation can mutate its `global_varinfo` field, which we need to access later.
"""
function make_conditional(
    model::DynamicPPL.Model,
    target_variables::AbstractVector{<:VarName},
    global_vnt::DynamicPPL.VarNamedTuple,
)
    # Insert the `GibbsContext` just before the leaf.
    # 1. Extract the `leafcontext` from `model` and wrap in `GibbsContext`.
    gibbs_context_inner = GibbsContext(
        target_variables, Ref(global_vnt), DynamicPPL.leafcontext(model.context)
    )
    # 2. Set the leaf context to be the `GibbsContext` wrapping `leafcontext(model.context)`.
    gibbs_context = DynamicPPL.setleafcontext(model.context, gibbs_context_inner)
    return DynamicPPL.contextualize(model, gibbs_context), gibbs_context_inner
end

to_varname(x::VarName) = x
to_varname(x::Symbol) = VarName{x}()
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

# Fields
$(TYPEDFIELDS)
"""
struct Gibbs{N,V<:NTuple{N,AbstractVector{<:VarName}},A<:NTuple{N,Any}} <: AbstractSampler
    # TODO(mhauru) Revisit whether A should have a fixed element type.
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

        samplers = tuple(samplers...)
        varnames = tuple(map(to_varname_list, varnames)...)
        return new{length(samplers),typeof(varnames),typeof(samplers)}(varnames, samplers)
    end
end

function Gibbs(algs::Pair...)
    return Gibbs(map(first, algs), map(last, algs))
end

struct GibbsState{V<:DynamicPPL.VarNamedTuple,S}
    vnt::V
    states::S
end

function check_all_variables_handled(vns, spl::Gibbs)
    handled_vars = Iterators.flatten(spl.varnames)
    missing_vars = [
        vn for vn in vns if !any(hv -> AbstractPPL.subsumes(hv, vn), handled_vars)
    ]
    if !isempty(missing_vars)
        msg =
            "The Gibbs sampler does not have a component sampler for: $(join(missing_vars, ", ")). " *
            "Please assign a component sampler to each variable in the model."
        throw(ArgumentError(msg))
    end
end

function Turing._check_model(model::DynamicPPL.Model, spl::Gibbs)
    # TODO(penelopeysm): Could be smarter: subsamplers may not allow discrete variables.
    Turing._check_model(model, !Turing.allow_discrete_variables(spl))
    varnames = keys(rand(model))
    return check_all_variables_handled(varnames, spl)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::Gibbs;
    initial_params=Turing.Inference.init_strategy(spl),
    discard_sample=false,
    kwargs...,
)
    varnames = spl.varnames
    samplers = spl.samplers
    accs = DynamicPPL.OnlyAccsVarInfo(DynamicPPL.RawValueAccumulator(false))
    _, accs = DynamicPPL.init!!(rng, model, accs, initial_params, DynamicPPL.UnlinkAll())
    vnt = DynamicPPL.get_raw_values(accs)

    vnt, states = gibbs_initialstep_recursive(
        rng,
        model,
        AbstractMCMC.step,
        varnames,
        samplers,
        vnt;
        initial_params=initial_params,
        kwargs...,
    )
    transition = if discard_sample
        nothing
    else
        DynamicPPL.ParamsWithStats(DynamicPPL.InitFromParams(vnt), model)
    end
    return transition, GibbsState(vnt, states)
end

function AbstractMCMC.step_warmup(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::Gibbs;
    initial_params=Turing.Inference.init_strategy(spl),
    discard_sample=false,
    kwargs...,
)
    varnames = spl.varnames
    samplers = spl.samplers
    # Sample a set of initial values
    accs = DynamicPPL.OnlyAccsVarInfo(DynamicPPL.RawValueAccumulator(false))
    _, accs = DynamicPPL.init!!(rng, model, accs, initial_params, DynamicPPL.UnlinkAll())
    vnt = DynamicPPL.get_raw_values(accs)

    vnt, states = gibbs_initialstep_recursive(
        rng,
        model,
        AbstractMCMC.step_warmup,
        varnames,
        samplers,
        vnt;
        initial_params=initial_params,
        kwargs...,
    )
    transition = if discard_sample
        nothing
    else
        DynamicPPL.ParamsWithStats(DynamicPPL.InitFromParams(vnt), model)
    end
    return transition, GibbsState(vnt, states)
end

"""
Take the first step of MCMC for the first component sampler, and call the same function
recursively on the remaining samplers, until no samplers remain. Return the global VarInfo
and a tuple of initial states for all component samplers.

The `step_function` argument should always be either AbstractMCMC.step or
AbstractMCMC.step_warmup.
"""
function gibbs_initialstep_recursive(
    rng,
    model,
    step_function::Function,
    varname_vecs,
    samplers,
    vnt,
    states=();
    initial_params,
    kwargs...,
)
    # End recursion
    if isempty(varname_vecs) && isempty(samplers)
        return vnt, states
    end

    varnames, varname_vecs_tail... = varname_vecs
    sampler, samplers_tail... = samplers

    # Construct the conditioned model.
    conditioned_model, context = make_conditional(model, varnames, vnt)

    # Take initial step with the current sampler.
    _, new_state = step_function(
        rng,
        conditioned_model,
        sampler;
        # FIXME: This will cause issues if the sampler expects initial params in unconstrained space.
        # This is not the case for any samplers in Turing.jl, but will be for external samplers, etc.
        initial_params=initial_params,
        kwargs...,
        discard_sample=true,
    )
    new_vnt_local = gibbs_get_raw_values(new_state)
    # Merge in any new variables that were introduced during the step, but that
    # were not in the domain of the current sampler.
    vnt = merge(vnt, get_global_vnt(context))
    # Merge the new values for all the variables sampled by the current sampler.
    vnt = merge(vnt, new_vnt_local)

    states = (states..., new_state)
    return gibbs_initialstep_recursive(
        rng,
        model,
        step_function,
        varname_vecs_tail,
        samplers_tail,
        vnt,
        states;
        initial_params=initial_params,
        kwargs...,
    )
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::Gibbs,
    state::GibbsState;
    discard_sample=false,
    kwargs...,
)
    varnames = spl.varnames
    samplers = spl.samplers
    states = state.states
    @assert length(samplers) == length(state.states)

    vnt, states = gibbs_step_recursive(
        rng, model, AbstractMCMC.step, varnames, samplers, states, state.vnt; kwargs...
    )

    transition = if discard_sample
        nothing
    else
        DynamicPPL.ParamsWithStats(DynamicPPL.InitFromParams(vnt), model)
    end
    return transition, GibbsState(vnt, states)
end

function AbstractMCMC.step_warmup(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::Gibbs,
    state::GibbsState;
    discard_sample=false,
    kwargs...,
)
    varnames = spl.varnames
    samplers = spl.samplers
    states = state.states
    @assert length(samplers) == length(state.states)

    vnt, states = gibbs_step_recursive(
        rng,
        model,
        AbstractMCMC.step_warmup,
        varnames,
        samplers,
        states,
        state.vnt;
        kwargs...,
    )
    transition = if discard_sample
        nothing
    else
        DynamicPPL.ParamsWithStats(DynamicPPL.InitFromParams(vnt), model)
    end
    return transition, GibbsState(vnt, states)
end

"""
Run a Gibbs step for the first varname/sampler/state tuple, and recursively call the same
function on the tail, until there are no more samplers left.

The `step_function` argument should always be either AbstractMCMC.step or
AbstractMCMC.step_warmup.
"""
function gibbs_step_recursive(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    step_function::Function,
    varname_vecs,
    samplers,
    states,
    global_vnt,
    new_states=();
    kwargs...,
)
    # End recursion.
    if isempty(varname_vecs) && isempty(samplers) && isempty(states)
        return global_vnt, new_states
    end

    varnames, varname_vecs_tail... = varname_vecs
    sampler, samplers_tail... = samplers
    state, states_tail... = states

    # Construct the conditional model that this sampler should use.
    conditioned_model, context = make_conditional(model, varnames, global_vnt)
    # Update the sampler's state based on global values that were provided by other
    # samplers.
    state = gibbs_update_state!!(sampler, state, conditioned_model, global_vnt)

    # Take a step with the local sampler. We don't need the actual sample, only the state.
    # Note that we pass `discard_sample=true` after `kwargs...`, because AbstractMCMC will
    # tell Gibbs that _this Gibbs sample_ should be kept, and so `kwargs` will actually
    # contain `discard_sample=false`!
    _, new_state = step_function(
        rng, conditioned_model, sampler, state; kwargs..., discard_sample=true
    )

    # The current sampler will return some raw values, which we update the global VNT with.
    new_vnt_local = gibbs_get_raw_values(new_state)
    new_global_vnt = merge(get_global_vnt(context), new_vnt_local)

    new_states = (new_states..., new_state)
    return gibbs_step_recursive(
        rng,
        model,
        step_function,
        varname_vecs_tail,
        samplers_tail,
        states_tail,
        new_global_vnt,
        new_states;
        kwargs...,
    )
end
