#
# Gibbs sampling
#

# Gibbs partitions the model's variables into blocks, one component sampler each, and sweeps
# the blocks in turn: to step a block, the model is `condition`ed on the current values of
# every variable outside it, so the component samples that block's full conditional.
#
# Those values are threaded explicitly as an immutable `VarNamedTuple` -- each component's
# step returns the values it owns, which are merged to form the next component's conditioning
# set. Nothing is written in place, so everything in a sweep reads one frozen snapshot.

#
# Interface for other samplers to work with Gibbs
#

"""
    is_gibbs_component(spl::AbstractSampler)

Return a boolean indicating whether `spl` is a valid component for a Gibbs sampler.

Defaults to `true` if no method has been defined for a particular sampler.
"""
is_gibbs_component(::AbstractSampler) = true
is_gibbs_component(spl::RepeatSampler) = is_gibbs_component(spl.sampler)
is_gibbs_component(spl::ExternalSampler) = is_gibbs_component(spl.sampler)
is_gibbs_component(::Prior) = false
is_gibbs_component(::Emcee) = false
is_gibbs_component(::SGLD) = false
is_gibbs_component(::SGHMC) = false
is_gibbs_component(::SMC) = false

"""
    Turing.Inference.gibbs_get_raw_values(state)

Return a `VarNamedTuple` containing the raw values of all variables in the sampler state.

Turing's Gibbs sampler maintains, at all points during the sampling process, a single global
`VarNamedTuple` that contains the **raw** values for all variables in the model. During the
sampling process, it calls each component sampler in turn and updates the global
`VarNamedTuple` with the new raw values returned by each sampler.

This function is used to pass that information *from* a component sampler *to* the Gibbs
sampler. Note that this means that the `VarNamedTuple` returned by this function should
**only** contain raw values for the variables that the component sampler is responsible for
sampling, and should not contain any values for other variables.
"""
function gibbs_get_raw_values end

"""
    Turing.Inference.gibbs_get_raw_values(state::AbstractVarInfo)

If your sampler state is an `AbstractVarInfo`, there is a default method available for this,
which reads the values stored in its `RawValueAccumulator`. (This means that the `VarInfo`
used for evaluation in the component sampler *must* contain a `RawValueAccumulator`.)
"""
function gibbs_get_raw_values(state::AbstractVarInfo)
    return DynamicPPL.get_raw_values(state)
end

"""
    Turing.Inference.gibbs_get_stats(state)

Return a `NamedTuple` of sampler statistics (acceptance rates, step sizes, and so on) for
the last step taken from `state`.

Gibbs discards its component samplers' transitions -- reading parameters off them would cost
a model re-evaluation -- so a component that wants its statistics to reach the chain has to
carry them on its state. Defaults to no statistics.
"""
gibbs_get_stats(::Any) = NamedTuple()

"""
    Turing.Inference.gibbs_update_state!!(
        sampler::AbstractSampler, state, model::Model, global_vals::VarNamedTuple
    )

Update the state of a Gibbs component sampler to be consistent with the new values in
`global_vals`. Each sampler should implement a method for its respective state type.

Note that the `model` argument passed in here will be 'conditioned' on the *new* values
inside `global_vals`. Thus, evaluating it will reflect the log-probability associated with
the new values.

Exactly what this function should do will depend on what the sampler state contains, but
for example, it will often mean:

- Updating any raw or vectorised values stored in the sampler state to be consistent with
  `global_vals`.
- Reevaluating the (new) model to update any cached log-probabilities.
- Updating any log-density callables (such as a `DynamicPPL.LogDensityFunction`) stored in
  the sampler state, to be consistent with the new model.

For examples of this, please see the implementations of this function for the samplers in
Turing.jl. In particular, the `HMC` and `ExternalSampler` implementations work with
`LogDensityFunction` and demonstrate how information such as that can be updated based on
the new model.
"""
function gibbs_update_state!! end

"""
    gibbs_recompute_ldf_and_params(
        old_ldf::LogDensityFunction,
        model::Model,
        global_vals::VarNamedTuple,
        extra_accs=()
    )

Shared helper that is used in `gibbs_update_state!!` for any sampler that uses a
LogDensityFunction.

Creates a new `LogDensityFunction` from the newly conditioned `model`, then reevaluates the
model to obtain the correct vectorised parameters corresponding to the raw values in
`global_vals`.

If extra information is needed (e.g. log-probabilities), `extra_accs` can be used to pass in
other accumulators to be used in the same model evaluation, to avoid having to recompute
them later.

Returns `(new_ldf, new_params, accs)` where `accs` is the set of accumulators after
evaluation, from which extra accumulators (e.g. `LogLikelihoodAccumulator`) can be read.

!!! warning
    This assumes that `old_ldf.model` (i.e., the model conditioned on the previous values)
    and `model` (i.e., the model conditioned on the new values) have the same structure, i.e.,
    all other components of the LogDensityFunction can be reused.
"""
function gibbs_recompute_ldf_and_params(
    old_ldf::DynamicPPL.LogDensityFunction,
    model::DynamicPPL.Model,
    global_vals::DynamicPPL.VarNamedTuple,
    extra_accs::NTuple{N,<:DynamicPPL.AbstractAccumulator}=(),
) where {N}
    new_ldf = DynamicPPL.LogDensityFunction(
        model,
        DynamicPPL.get_logdensity_callable(old_ldf),
        DynamicPPL.get_all_ranges_and_transforms(old_ldf),
        DynamicPPL.get_sample_input_vector(old_ldf);
        adtype=old_ldf.adtype,
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

#
# Gibbs implementation itself
#

"""
    conditioned_values(global_vnt, target_variables)

Return the values in `global_vnt` for every variable *not* sampled by this Gibbs component,
i.e. the ones it conditions on.

Conditioned variables reach `tilde_observe!!`, so particle samplers reweight on them. That is
what makes the component's target distribution correct: a conditioned variable the target
depends on must reweight the sweep, and one it does not contributes the same increment to
every particle, which ESS-gated resampling ignores.
"""
function conditioned_values(
    global_vnt::DynamicPPL.VarNamedTuple, target_variables::AbstractVector{<:VarName}
)
    # Overlap is tested in both directions because a key can be finer than a target (`x`
    # covers `x[1]`) or coarser than one (a component owning `x[1]` writes back the whole
    # `x`); conditioning a target on its own stale value would leave nothing to sample.
    overlaps(a, b) = AbstractPPL.subsumes(a, b) || AbstractPPL.subsumes(b, a)
    conditioned = Tuple(
        vn for vn in keys(global_vnt) if !any(t -> overlaps(t, vn), target_variables)
    )
    return DynamicPPL.subset(global_vnt, conditioned)
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

Note that all variables in the model should be handled by one or more samplers. The
behaviour of Gibbs when there are unhandled variables is undefined: depending on the version
of Turing, it may either crash, or it may sample once from the prior and not update values
after that. See https://github.com/TuringLang/Turing.jl/issues/2810 for more information.

There is currently no way to specify a different initialisation strategy for each component
sampler individually. When sampling with Gibbs, `initial_params` applies to the model as a
whole.

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
            if !is_gibbs_component(spl)
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

"""
    GibbsInitStrategy(varnames, strategies)

Initialise each variable with the strategy of the component sampler that samples it, so that
e.g. an `HMC` component still gets its `InitFromUniform` starting point inside Gibbs. A
variable no component claims falls back to the prior.
"""
struct GibbsInitStrategy{V,S} <: DynamicPPL.AbstractInitStrategy
    varnames::V
    strategies::S
end

function DynamicPPL.init(
    rng::Random.AbstractRNG, vn::VarName, dist::Distribution, strategy::GibbsInitStrategy
)
    for (vns, component_strategy) in zip(strategy.varnames, strategy.strategies)
        if any(Base.Fix2(AbstractPPL.subsumes, vn), vns)
            return DynamicPPL.init(rng, vn, dist, component_strategy)
        end
    end
    return DynamicPPL.init(rng, vn, dist, DynamicPPL.InitFromPrior())
end

function init_strategy(spl::Gibbs)
    return GibbsInitStrategy(spl.varnames, map(init_strategy, spl.samplers))
end

"""
    component_stats(spl::Gibbs, states)

Collect the component samplers' statistics into one `NamedTuple`, prefixing each with the
symbols of the variables that component samples, so that two components reporting e.g.
`acceptance_rate` do not collide. Components sampling the same symbols are further
distinguished by their index.

The prefix uses each variable's symbol rather than its whole `VarName` because chain packages
read variable structure back out of these names: `MCMCChains.namesingroup(chn, :x)` matches
anything beginning `x[`, so a statistic named `x[1]_acceptance_rate` would be served up as one
of `x`'s draws. A symbol carries no optic, so it cannot be parsed as part of another variable.
"""
function component_stats(spl::Gibbs, states)
    prefixes = map(vns -> join(unique(map(AbstractPPL.getsym, vns)), "_"), spl.varnames)
    stats = NamedTuple()
    for (i, (prefix, state)) in enumerate(zip(prefixes, states))
        component = gibbs_get_stats(state)
        isempty(component) && continue
        name = count(==(prefix), prefixes) > 1 ? string(prefix, "_", i) : prefix
        names = map(k -> Symbol(name, "_", k), keys(component))
        stats = merge(stats, NamedTuple{names}(values(component)))
    end
    return stats
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
        DynamicPPL.ParamsWithStats(
            DynamicPPL.InitFromParams(vnt), model, component_stats(spl, states)
        )
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
        DynamicPPL.ParamsWithStats(
            DynamicPPL.InitFromParams(vnt), model, component_stats(spl, states)
        )
    end
    return transition, GibbsState(vnt, states)
end

"""
Take the first step of MCMC for the first component sampler, and call the same function
recursively on the remaining samplers, until no samplers remain. Return the global VNT
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
    conditioned_model = DynamicPPL.condition(model, conditioned_values(vnt, varnames))

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
    # New values for the variables this sampler is responsible for, plus any variable it
    # encountered that no component owns yet: both arrive in its own raw values.
    vnt = merge(vnt, gibbs_get_raw_values(new_state))

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
        DynamicPPL.ParamsWithStats(
            DynamicPPL.InitFromParams(vnt), model, component_stats(spl, states)
        )
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
        DynamicPPL.ParamsWithStats(
            DynamicPPL.InitFromParams(vnt), model, component_stats(spl, states)
        )
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
    conditioned_model = DynamicPPL.condition(
        model, conditioned_values(global_vnt, varnames)
    )
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
    new_global_vnt = merge(global_vnt, gibbs_get_raw_values(new_state))

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
