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

function can_be_wrapped(ctx::DynamicPPL.AbstractContext)
    return DynamicPPL.NodeTrait(ctx) isa DynamicPPL.IsLeaf
end
can_be_wrapped(ctx::DynamicPPL.PrefixContext) = can_be_wrapped(ctx.context)

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
    GibbsContext(target_varnames, global_varinfo, context)

A context used in the implementation of the Turing.jl Gibbs sampler.

There will be one `GibbsContext` for each iteration of a component sampler.

`target_varnames` is a a tuple of `VarName`s that the current component sampler
is sampling. For those `VarName`s, `GibbsContext` will just pass `tilde_assume`
calls to its child context. For other variables, their values will be fixed to
the values they have in `global_varinfo`.

# Fields
$(FIELDS)
"""
struct GibbsContext{
    VNs<:Tuple{Vararg{VarName}},GVI<:Ref{<:AbstractVarInfo},Ctx<:DynamicPPL.AbstractContext
} <: DynamicPPL.AbstractContext
    """
    the VarNames being sampled
    """
    target_varnames::VNs
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

    function GibbsContext(target_varnames, global_varinfo, context)
        if !can_be_wrapped(context)
            error("GibbsContext can only wrap a leaf or prefix context, not a $(context).")
        end
        target_varnames = tuple(target_varnames...)  # Allow vectors.
        return new{typeof(target_varnames),typeof(global_varinfo),typeof(context)}(
            target_varnames, global_varinfo, context
        )
    end
end

function GibbsContext(target_varnames, global_varinfo)
    return GibbsContext(target_varnames, global_varinfo, DynamicPPL.DefaultContext())
end

DynamicPPL.NodeTrait(::GibbsContext) = DynamicPPL.IsParent()
DynamicPPL.childcontext(context::GibbsContext) = context.context
function DynamicPPL.setchildcontext(context::GibbsContext, childcontext)
    return GibbsContext(
        context.target_varnames, Ref(context.global_varinfo[]), childcontext
    )
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

# Tilde pipeline
function DynamicPPL.tilde_assume(context::GibbsContext, right, vn, vi)
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
    vn, child_context = DynamicPPL.prefix_and_strip_contexts(child_context, vn)

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
    vn, child_context = DynamicPPL.prefix_and_strip_contexts(child_context, vn)

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

wrap_in_sampler(x::AbstractMCMC.AbstractSampler) = x
wrap_in_sampler(x::InferenceAlgorithm) = DynamicPPL.Sampler(x)

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
struct Gibbs{N,V<:NTuple{N,AbstractVector{<:VarName}},A<:NTuple{N,Any}} <:
       InferenceAlgorithm
    # TODO(mhauru) Revisit whether A should have a fixed element type once
    # InferenceAlgorithm/Sampler types have been cleaned up.
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

        samplers = tuple(map(wrap_in_sampler, samplers)...)
        varnames = tuple(map(to_varname_list, varnames)...)
        return new{length(samplers),typeof(varnames),typeof(samplers)}(varnames, samplers)
    end
end

"""
    GibbsConditional(sampler, varnames)
    GibbsConditional(sym, conditional)

A conditional sampler for use within a Gibbs sampler. Has two forms:

1. **Sampler-based**: `GibbsConditional(sampler, varnames)` samples variables in `varnames` 
   conditionally on all other variables using the provided `sampler`.

2. **Analytical**: `GibbsConditional(sym, conditional)` samples the variable `sym` according 
   to the analytical conditional `conditional`, which must be a function from a NamedTuple 
   of the conditioned variables to a Distribution.

# Arguments (Sampler-based)
- `sampler`: The underlying sampler to use for sampling the specified variables.
- `varnames`: A single variable name or a collection of variable names to be sampled.
  Each variable name can be given as either a `Symbol` or a `VarName`.

# Arguments (Analytical)
- `sym`: A single variable name (Symbol or VarName) to be sampled.
- `conditional`: A function from a NamedTuple of conditioned variables to a Distribution.

# Examples
```julia
# Sampler-based: Sample x and y jointly using MH, conditioned on all other variables
GibbsConditional(MH(), [:x, :y])

# Sampler-based: Sample only z using NUTS, conditioned on all other variables  
GibbsConditional(NUTS(), :z)

# Analytical: Sample λ according to a conditional function
function cond_λ(c)
    α_n = α_0 + (N - 1) / 2 + 1
    β_n = s2 * N / 2 + c.m^2 / 2 + inv(θ_0)
    return Gamma(α_n, inv(β_n))
end
GibbsConditional(:λ, cond_λ)

# Use in a Gibbs sampler
Gibbs(
    GibbsConditional(:λ, cond_λ),
    GibbsConditional(:m, cond_m)
)
```
"""
struct GibbsConditional{S,V,C}
    "The underlying sampler (Nothing for analytical conditionals)"
    sampler::S
    "Variable names to be sampled"
    varnames::V
    "Analytical conditional function (Nothing for sampler-based conditionals)"
    conditional::C
    
    # Constructor for sampler-based conditionals
    function GibbsConditional(sampler, varnames)
        # Ensure the sampler is a valid Gibbs component
        if !isgibbscomponent(sampler)
            msg = "The provided sampler $(sampler) is not a valid Gibbs component."
            throw(ArgumentError(msg))
        end
        # Convert varnames to a consistent format
        varnames_vec = to_varname_list(varnames)
        # Wrap sampler if needed
        wrapped_sampler = wrap_in_sampler(sampler)
        return new{typeof(wrapped_sampler),typeof(varnames_vec),Nothing}(wrapped_sampler, varnames_vec, nothing)
    end
    
    # Constructor for analytical conditionals
    function GibbsConditional(sym::Union{Symbol, VarName, AbstractVector{<:Union{Symbol, VarName}}}, conditional)
        # Validate conditional is callable
        if !isa(conditional, Function)
            throw(ArgumentError("conditional must be a function that takes a NamedTuple and returns a Distribution"))
        end
        
        # Convert symbol to consistent format - analytical conditionals should work with single variables
        if isa(sym, AbstractVector) && length(sym) != 1
            throw(ArgumentError("Analytical conditionals currently support only single variables"))
        end
        varname_vec = to_varname_list(sym)
        
        return new{Nothing,typeof(varname_vec),typeof(conditional)}(nothing, varname_vec, conditional)
    end
end

# Make GibbsConditional a valid Gibbs component
isgibbscomponent(::GibbsConditional) = true

# Helper functions to determine the type of GibbsConditional
is_analytical_conditional(gc::GibbsConditional) = gc.conditional !== nothing
is_sampler_conditional(gc::GibbsConditional) = gc.sampler !== nothing

# Add wrap_in_sampler method for analytical GibbsConditional
wrap_in_sampler(gc::GibbsConditional) = is_analytical_conditional(gc) ? gc : wrap_in_sampler(gc.sampler)

# Original Pair constructor
function Gibbs(algs::Pair...)
    return Gibbs(map(first, algs), map(last, algs))
end

# Special handling for two arguments where at least one is GibbsConditional
function Gibbs(arg1::Union{Pair, GibbsConditional}, arg2::Union{Pair, GibbsConditional})
    # If both are pairs, use the optimized path
    if arg1 isa Pair && arg2 isa Pair
        return Gibbs((first(arg1), first(arg2)), (last(arg1), last(arg2)))
    end
    
    # Otherwise, process the arguments
    varnames = []
    samplers = []
    for arg in (arg1, arg2)
        if arg isa Pair
            push!(varnames, first(arg))
            push!(samplers, last(arg))
        elseif arg isa GibbsConditional
            push!(varnames, arg.varnames)
            # For analytical conditionals, store the GibbsConditional itself as the "sampler"
            if is_analytical_conditional(arg)
                push!(samplers, arg)  # Store the whole GibbsConditional
            else
                push!(samplers, arg.sampler)  # Store the wrapped sampler
            end
        end
    end
    return Gibbs(tuple(varnames...), tuple(samplers...))
end

# Allow GibbsConditional (and mixed with Pairs) in Gibbs constructor
function Gibbs(arg1, arg2, args...)
    # Collect all arguments
    all_args = (arg1, arg2, args...)
    
    # If all arguments are Pairs, use the optimized Pair constructor
    if all(arg isa Pair for arg in all_args)
        return Gibbs(map(first, all_args), map(last, all_args))
    end
    
    # Otherwise process each argument - could be a Pair or a GibbsConditional
    varnames = []
    samplers = []
    for arg in all_args
        if arg isa Pair
            push!(varnames, first(arg))
            push!(samplers, last(arg))
        elseif arg isa GibbsConditional
            push!(varnames, arg.varnames)
            # For analytical conditionals, store the GibbsConditional itself as the "sampler"
            # so we can identify it later in the stepping process
            if is_analytical_conditional(arg)
                push!(samplers, arg)  # Store the whole GibbsConditional
            else
                push!(samplers, arg.sampler)  # Store the wrapped sampler
            end
        else
            throw(ArgumentError("Gibbs arguments must be Pairs or GibbsConditional objects, got $(typeof(arg))"))
        end
    end
    return Gibbs(tuple(varnames...), tuple(samplers...))
end

struct GibbsState{V<:DynamicPPL.AbstractVarInfo,S}
    vi::V
    states::S
end

varinfo(state::GibbsState) = state.vi

# State type for analytical conditionals
struct AnalyticalConditionalState{V<:DynamicPPL.AbstractVarInfo}
    vi::V
    value::Any
    logp::Real
end

varinfo(state::AnalyticalConditionalState) = state.vi

"""
Initialise a VarInfo for the Gibbs sampler.

This is straight up copypasta from DynamicPPL's src/sampler.jl. It is repeated here to
support calling both step and step_warmup as the initial step. DynamicPPL initialstep is
incompatible with step_warmup.
"""
function initial_varinfo(rng, model, spl, initial_params)
    vi = DynamicPPL.default_varinfo(rng, model, spl)

    # Update the parameters if provided.
    if initial_params !== nothing
        vi = DynamicPPL.initialize_parameters!!(vi, initial_params, model)

        # Update joint log probability.
        # This is a quick fix for https://github.com/TuringLang/Turing.jl/issues/1588
        # and https://github.com/TuringLang/Turing.jl/issues/1563
        # to avoid that existing variables are resampled
        vi = last(DynamicPPL.evaluate!!(model, vi, DynamicPPL.DefaultContext()))
    end
    return vi
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::DynamicPPL.Sampler{<:Gibbs};
    initial_params=nothing,
    kwargs...,
)
    alg = spl.alg
    varnames = alg.varnames
    samplers = alg.samplers
    vi = initial_varinfo(rng, model, spl, initial_params)

    vi, states = gibbs_initialstep_recursive(
        rng,
        model,
        AbstractMCMC.step,
        varnames,
        samplers,
        vi;
        initial_params=initial_params,
        kwargs...,
    )
    return Transition(model, vi), GibbsState(vi, states)
end

function AbstractMCMC.step_warmup(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::DynamicPPL.Sampler{<:Gibbs};
    initial_params=nothing,
    kwargs...,
)
    alg = spl.alg
    varnames = alg.varnames
    samplers = alg.samplers
    vi = initial_varinfo(rng, model, spl, initial_params)

    vi, states = gibbs_initialstep_recursive(
        rng,
        model,
        AbstractMCMC.step_warmup,
        varnames,
        samplers,
        vi;
        initial_params=initial_params,
        kwargs...,
    )
    return Transition(model, vi), GibbsState(vi, states)
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
    vi,
    states=();
    initial_params=nothing,
    kwargs...,
)
    # End recursion
    if isempty(varname_vecs) && isempty(samplers)
        return vi, states
    end

    varnames, varname_vecs_tail... = varname_vecs
    sampler, samplers_tail... = samplers

    # Get the initial values for this component sampler.
    initial_params_local = if initial_params === nothing
        nothing
    else
        DynamicPPL.subset(vi, varnames)[:]
    end

    # Check if this is an analytical conditional
    if sampler isa GibbsConditional && is_analytical_conditional(sampler)
        # For analytical conditionals, we need to initialize with a sample
        # Create a dummy state for the first sample  
        dummy_state = AnalyticalConditionalState(vi, nothing, 0.0)
        new_vi_local, new_state = sample_analytical_conditional(
            rng, sampler, vi, varnames, dummy_state
        )
        # For analytical conditionals, new_vi_local contains the updated subset
        # Note: we need to update the global vi with the new values later
    else
        # Construct the conditioned model.
        conditioned_model, context = make_conditional(model, varnames, vi)

        # Take initial step with the current sampler.
        _, new_state = step_function(
            rng,
            conditioned_model,
            sampler;
            # FIXME: This will cause issues if the sampler expects initial params in unconstrained space.
            # This is not the case for any samplers in Turing.jl, but will be for external samplers, etc.
            initial_params=initial_params_local,
            kwargs...,
        )
        new_vi_local = varinfo(new_state)
        # Merge in any new variables that were introduced during the step, but that
        # were not in the domain of the current sampler.
        vi = merge(vi, get_global_varinfo(context))
    end
    # Merge the new values for all the variables sampled by the current sampler.
    vi = merge(vi, new_vi_local)

    states = (states..., new_state)
    return gibbs_initialstep_recursive(
        rng,
        model,
        step_function,
        varname_vecs_tail,
        samplers_tail,
        vi,
        states;
        initial_params=initial_params,
        kwargs...,
    )
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

    vi, states = gibbs_step_recursive(
        rng, model, AbstractMCMC.step, varnames, samplers, states, vi; kwargs...
    )
    return Transition(model, vi), GibbsState(vi, states)
end

function AbstractMCMC.step_warmup(
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

    vi, states = gibbs_step_recursive(
        rng, model, AbstractMCMC.step_warmup, varnames, samplers, states, vi; kwargs...
    )
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
    logdensity = DynamicPPL.LogDensityFunction(
        model, state.ldf.varinfo, state.ldf.context; adtype=sampler.alg.adtype
    )
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

"""
Sample from an analytical conditional distribution.

This function evaluates the conditional function with the current variable values
and samples from the resulting distribution.
"""
function sample_analytical_conditional(
    rng::Random.AbstractRNG,
    gc::GibbsConditional,
    global_vi::DynamicPPL.AbstractVarInfo,
    target_varnames,
    state
)
    # Extract current values of all variables except the target
    var_dict = Dict{Symbol,Any}()
    for vn in keys(global_vi)
        # Convert VarName to Symbol for the NamedTuple
        sym = Symbol(vn)
        if sym ∉ [Symbol(tvn) for tvn in target_varnames]  # Don't include target variable
            var_dict[sym] = global_vi[vn]
        end
    end
    
    # Convert to NamedTuple as expected by the conditional function
    conditioning_vars = NamedTuple(var_dict)
    
    # Evaluate the conditional function to get the distribution
    try
        conditional_dist = gc.conditional(conditioning_vars)
        if !isa(conditional_dist, Distribution)
            throw(ArgumentError("Conditional function must return a Distribution, got $(typeof(conditional_dist))"))
        end
        
        # Sample from the conditional distribution
        new_value = rand(rng, conditional_dist)
        
        # Create a local VarInfo with just the target variable
        # This follows the pattern used by other samplers in the Gibbs framework
        local_vi = DynamicPPL.subset(global_vi, target_varnames)
        target_vn = target_varnames[1]  # Analytical conditionals are single-variable
        local_vi = DynamicPPL.setindex!!(local_vi, new_value, target_vn)
        
        # Update log probability with the conditional distribution's logpdf
        logp_new = logpdf(conditional_dist, new_value)
        local_vi = setlogp!!(local_vi, logp_new)
        
        # Create a proper state for the analytical conditional that contains the VarInfo
        analytical_state = AnalyticalConditionalState(local_vi, new_value, logp_new)
        
        return local_vi, analytical_state
        
    catch e
        if isa(e, MethodError)
            throw(ArgumentError("Conditional function failed: $(e). Make sure it accepts a NamedTuple and returns a Distribution."))
        else
            rethrow(e)
        end
    end
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
    global_vi,
    new_states=();
    kwargs...,
)
    # End recursion.
    if isempty(varname_vecs) && isempty(samplers) && isempty(states)
        return global_vi, new_states
    end

    varnames, varname_vecs_tail... = varname_vecs
    sampler, samplers_tail... = samplers
    state, states_tail... = states

    # Construct the conditional model and the varinfo that this sampler should use.
    conditioned_model, context = make_conditional(model, varnames, global_vi)
    vi = DynamicPPL.subset(global_vi, varnames)
    vi = match_linking!!(vi, state, model)

    # TODO(mhauru) The below may be overkill. If the varnames for this sampler are not
    # sampled by other samplers, we don't need to `setparams`, but could rather simply
    # recompute the log probability. More over, in some cases the recomputation could also
    # be avoided, if e.g. the previous sampler has done all the necessary work already.
    # However, we've judged that doing any caching or other tricks to avoid this now would
    # be premature optimization. In most use cases of Gibbs a single model call here is not
    # going to be a significant expense anyway.
    # Check if this is an analytical conditional
    if sampler isa GibbsConditional && is_analytical_conditional(sampler)
        # Handle analytical conditional sampling
        new_vi_local, new_state = sample_analytical_conditional(
            rng, sampler, global_vi, varnames, state
        )
    else
        # Set the state of the current sampler, accounting for any changes made by other
        # samplers.
        state = setparams_varinfo!!(conditioned_model, sampler, state, vi)

        # Take a step with the local sampler.
        new_state = last(step_function(rng, conditioned_model, sampler, state; kwargs...))

        new_vi_local = varinfo(new_state)
    end
    # Merge the latest values for all the variables in the current sampler.
    new_global_vi = merge(get_global_varinfo(context), new_vi_local)
    new_global_vi = setlogp!!(new_global_vi, getlogp(new_vi_local))

    new_states = (new_states..., new_state)
    return gibbs_step_recursive(
        rng,
        model,
        step_function,
        varname_vecs_tail,
        samplers_tail,
        states_tail,
        new_global_vi,
        new_states;
        kwargs...,
    )
end
