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
# Reading what to condition off that snapshot's keys is equivalent to deciding it per tilde
# statement, where the model's own `VarName` is available, while two invariants hold:
#
#   - The partition is exact: no component's variable strictly contains another's. Under
#     `Gibbs(@varname(x[1]) => MH(), @varname(x) => HMC(0.05, 3))` the second component writes
#     `x` back as a unit, so freeing it for the first component to sample `x[1]` would free
#     `x[2]` as well. `conditioned_values` throws rather than hand a component a larger block
#     than it owns. Sampling such a partition means writing a sampler against the AbstractMCMC
#     interface directly, since this one cannot express conditioning on part of a value.
#   - A block whose variables change between sweeps is sampled only by a component built for
#     it. A variable the model reaches only on some sweeps -- a new name, `z` in
#     `x ~ Normal(); y ~ Normal(); if x > 0; z ~ Normal(); end`, or a new element, `x[5]`
#     inside a branch -- turns up in whichever component's step gets there first.
#     `adopt_new_variables` hands it to the component declared for it, or, if none was, to the
#     component that found it, and throws unless that component can sample a set of variables
#     that changes between sweeps. Either way the variable stays in the snapshot even in
#     sweeps where the model never reaches it.
#
#     Owning such a variable is necessary but not sufficient: the variables that decide
#     whether it exists have to be in the same block. Under
#     `Gibbs(@varname(x) => MH(), @varname(z) => PG(20))` on
#     `x ~ Normal(); if x > 0; z ~ Normal(); end`, the `x` component conditions on `z` while
#     proposing an `x` for which `z` does not exist, so its acceptance ratio compares two
#     different supports; the chain samples but is biased, measurably (P(x > 0) drops from
#     1/2 to about 1/20). `Gibbs((@varname(x), @varname(z)) => PG(20))` is the correct
#     partition. Gibbs cannot detect the difference, because which variables a branch
#     depends on is not something it can see.
#
# That last part has no fix at this level: a block's shape is discovered while stepping it, so
# Gibbs cannot tell in advance which variables a sweep will reach. Knowing that beforehand
# needs the model's trace type separated from its values.

#
# Interface for other samplers to work with Gibbs
#

"""
    supports_gibbs(spl::AbstractSampler)

Return a boolean indicating whether `spl` is a valid component for a Gibbs sampler.

Defaults to `true` if no method has been defined for a particular sampler.
"""
function supports_gibbs(spl::AbstractSampler)
    if which(isgibbscomponent, Tuple{typeof(spl)}) !== _ISGIBBSCOMPONENT_DEFAULT
        Base.depwarn(
            "`Turing.Inference.isgibbscomponent` is deprecated, define " *
            "`Turing.Inference.supports_gibbs` instead.",
            :supports_gibbs,
        )
        return isgibbscomponent(spl)
    end
    return true
end
supports_gibbs(spl::RepeatSampler) = supports_gibbs(spl.sampler)
supports_gibbs(spl::ExternalSampler) = supports_gibbs(spl.sampler)
supports_gibbs(::Prior) = false
supports_gibbs(::Emcee) = false
supports_gibbs(::SGLD) = false
supports_gibbs(::SGHMC) = false
supports_gibbs(::SMC) = false

"""
    isgibbscomponent(spl::AbstractSampler)

Deprecated name for [`supports_gibbs`](@ref), still honoured so that a sampler written
against it keeps working.
"""
isgibbscomponent(::AbstractSampler) = true

# An `isgibbscomponent` overload is any method more specific than the one above, which is what
# `supports_gibbs`'s fallback looks for. `Base.depwarn` is silent unless asked for
# (`--depwarn=yes`, as package test suites use), which is right here: the sampler's author has
# to act on it, not whoever runs it.
const _ISGIBBSCOMPONENT_DEFAULT = which(isgibbscomponent, Tuple{AbstractSampler})

"""
    Turing.Inference.allow_varying_dimension(spl::AbstractSampler)

Whether `spl` can sample a block whose set of variables changes between Gibbs sweeps.

Defaults to `false`, because sampling a target whose dimension changes takes a construction
built for it. A sampler reusing a `LogDensityFunction` through
[`gibbs_recompute_ldf_and_params`](@ref) has no slot for a variable that appears later, and a
Metropolis-Hastings ratio between states of different dimension is not the one the algorithm
assumes. `PG` and `CSMC` rebuild their trace each sweep, drawing whatever the model reaches,
so they can.

Returning `true` is not by itself enough for an array whose length varies and whose elements
are drawn one at a time (`for i in 1:n; x[i] ~ ...; end` under a random `n`). Those are stored
as a `PartialArray` backed by a plain `Vector`, and merging two of different lengths throws
`ArgumentError: Cannot merge PartialArrays with different axes` from DynamicPPL, in either
direction, before this trait is consulted. A single draw of the whole vector,
`x ~ MvNormal(zeros(n), I)`, is replaced key and all and does work.
"""
allow_varying_dimension(::AbstractSampler) = false
# `RepeatSampler` hands `gibbs_update_state!!` to the sampler it wraps, so the inner answer is
# the right one. `ExternalSampler` does not: its own `gibbs_update_state!!` goes through
# `gibbs_recompute_ldf_and_params`, so it keeps the `false` default whatever it wraps.
allow_varying_dimension(spl::RepeatSampler) = allow_varying_dimension(spl.sampler)
allow_varying_dimension(::PG) = true

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
    is_target(vn) = any(t -> AbstractPPL.subsumes(t, vn), target_variables)
    conditioned = VarName[]
    for vn in keys(global_vnt)
        if is_target(vn)
            # Wholly owned by this component, so it must be left free to sample.
        elseif any(t -> AbstractPPL.subsumes(vn, t), target_variables)
            # The values store `vn` as a unit but the component owns only part of it, so
            # leaving it free also frees the rest. That is only safe if the component owns
            # every leaf; otherwise it would silently sample a larger block than it was given.
            for leaf in AbstractPPL.varname_leaves(vn, global_vnt[vn])
                is_target(leaf) || throw(
                    ArgumentError(
                        "Gibbs cannot condition on part of $(vn): the values store it as a" *
                        " unit, but this component does not sample $(leaf). Give the" *
                        " component all of $(vn), or split the variable in the model.",
                    ),
                )
            end
        else
            push!(conditioned, vn)
        end
    end
    return DynamicPPL.subset(global_vnt, Tuple(conditioned))
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

Every variable in the model must be handled by at least one component sampler, and several
components may sample the same variable. What they may not do is split a value that is stored
as a unit: if one component declares `x` and another `x[1]`, whether Gibbs can express that
depends on how the values arrive. Written element by element -- `x[1] ~ Normal(); x[2] ~
Normal()`, sampled by MH -- each element is a key of its own and the partition works. Written
as one draw, `x ~ MvNormal(...)`, or handed back as a unit by a component that vectorises its
parameters, such as HMC, `x` is a single key: Gibbs cannot free part of it and throws. So the
same pair of declared varnames may sample or throw depending on the model and on the other
component's sampler.

Variables the model only reaches on some sweeps -- a `z` inside `if x > 0`, or an `x[5]`
inside a branch -- are handled a little differently. One that appears part-way through a run
is taken on by the component declared for it, or, if you declared none, by the component that
first reached it, which then keeps sampling it. Either way that component has to be able to
sample a set of variables that changes between sweeps, such as `PG`; Gibbs throws if it
cannot, since the variable would otherwise be drawn once and then conditioned on for the rest
of the run.

Each component sampler initialises the variables it samples with its own default strategy, so
e.g. an `HMC` component starts from its `InitFromUniform`. A user-supplied `initial_params`
overrides that and applies to the model as a whole; it cannot yet be set per component.

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
            if !supports_gibbs(spl)
                msg = "All samplers must be valid Gibbs components, $(nameof(typeof(spl))) is not."
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

struct GibbsState{V<:DynamicPPL.VarNamedTuple,S,A}
    vnt::V
    states::S
    """
    per component, the variables it picked up mid-run because no component was declared for
    them; kept so the component that discovered one keeps sampling it (see
    [`adopt_new_variables`](@ref)).
    """
    adopted::A
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

"""
    adopt_new_variables(spl, sampler, adopted, old_vnt, new_vnt)

Settle what happens to variables that appeared during one component's step, returning the
variables this component has adopted.

A variable the model reaches only on some sweeps shows up in whichever component's step gets
there first. If a component was declared for it, that component must be able to sample a set
of variables that changes between sweeps. If none was, the component that found it adopts it,
on the same condition, so that it keeps sampling the variable instead of conditioning on a
single draw for the rest of the run. Otherwise this throws.

The variables the model starts with are [`check_all_variables_handled`](@ref)'s business;
this covers only what a step adds to the snapshot.
"""
function adopt_new_variables(
    spl::Gibbs,
    sampler,
    adopted,
    old_vnt::DynamicPPL.VarNamedTuple,
    new_vnt::DynamicPPL.VarNamedTuple,
)
    # Walk to the leaves rather than compare keys: an array stored under a single key can grow
    # an element, and an `x[5]` appearing inside a branch is as unhandled as a new `z`.
    for vn in keys(new_vnt), leaf in AbstractPPL.varname_leaves(vn, new_vnt[vn])
        DynamicPPL.hasvalue(old_vnt, leaf) && continue
        owner = findfirst(
            vns -> any(hv -> AbstractPPL.subsumes(hv, leaf), vns), spl.varnames
        )
        owning_sampler = owner === nothing ? sampler : spl.samplers[owner]
        if !allow_varying_dimension(owning_sampler)
            name = nameof(typeof(owning_sampler))
            role = owner === nothing ? "that found it" : "declared for it"
            throw(
                ArgumentError(
                    "The variable $(leaf) appeared during sampling, and would be sampled " *
                    "by $(name), the component $(role). $(name) fixes the set of variables " *
                    "it samples at its first step. Give $(leaf) to a component that can " *
                    "sample a varying set of variables, such as `PG`, and put the variables " *
                    "that decide whether $(leaf) exists in that same block: a component " *
                    "that conditions on $(leaf) while proposing a state where $(leaf) does " *
                    "not exist is comparing two different supports, which biases the chain.",
                ),
            )
        end
        owner === nothing && (adopted = vcat(adopted, leaf))
    end
    return adopted
end

function check_all_variables_handled(vns, spl::Gibbs)
    handled_vars = Iterators.flatten(spl.varnames)
    missing_vars = [
        vn for vn in vns if !any(hv -> AbstractPPL.subsumes(hv, vn), handled_vars)
    ]
    if !isempty(missing_vars)
        msg =
            "The Gibbs sampler has no component for $(join(missing_vars, ", ")). Assign " *
            "every variable the model reaches to a component; one left out is never " *
            "sampled, and every component conditions on a single draw of it for the whole " *
            "run. A variable the model reaches only inside a branch, so that it is there on " *
            "some sweeps and not others, has to go to a component that can sample a varying " *
            "set of variables, such as `PG`."
        throw(ArgumentError(msg))
    end
end

"""
    gibbs_initial_values(rng, model, spl, initial_params)

Return the values the sweep starts from: one draw of the model, its `:=` quantities included.

`:=` quantities are not variables, so [`check_all_variables_handled`](@ref) runs before they
are added. Components report them among their own values, and having them here keeps the sweep
from taking them for variables that appeared while sampling.
"""
function gibbs_initial_values(rng, model, spl::Gibbs, initial_params)
    accs = DynamicPPL.OnlyAccsVarInfo(DynamicPPL.RawValueAccumulator(false))
    _, accs = DynamicPPL.init!!(rng, model, accs, initial_params, DynamicPPL.UnlinkAll())
    vnt = DynamicPPL.get_raw_values(accs)
    check_all_variables_handled(keys(vnt), spl)
    accs = DynamicPPL.OnlyAccsVarInfo(DynamicPPL.RawValueAccumulator(true))
    _, accs = DynamicPPL.init!!(
        rng, model, accs, DynamicPPL.InitFromParams(vnt), DynamicPPL.UnlinkAll()
    )
    return DynamicPPL.get_raw_values(accs)
end

function Turing._check_model(model::DynamicPPL.Model, spl::Gibbs)
    # TODO(penelopeysm): Could be smarter: subsamplers may not allow discrete variables.
    return Turing._check_model(model, !Turing.allow_discrete_variables(spl))
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
    vnt = gibbs_initial_values(rng, model, spl, initial_params)

    vnt, states, adopted = gibbs_initialstep_recursive(
        rng,
        model,
        AbstractMCMC.step,
        spl,
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
    return transition, GibbsState(vnt, states, adopted)
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
    vnt = gibbs_initial_values(rng, model, spl, initial_params)

    vnt, states, adopted = gibbs_initialstep_recursive(
        rng,
        model,
        AbstractMCMC.step_warmup,
        spl,
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
    return transition, GibbsState(vnt, states, adopted)
end

"""
Take the first step of MCMC for the first component sampler, and call the same function
recursively on the remaining samplers, until no samplers remain. Return the global VNT
and a tuple of initial states for all component samplers, plus what each has adopted.

The `step_function` argument should always be either AbstractMCMC.step or
AbstractMCMC.step_warmup.
"""
function gibbs_initialstep_recursive(
    rng,
    model,
    step_function::Function,
    spl::Gibbs,
    varname_vecs,
    samplers,
    vnt,
    states=(),
    adopted_vecs=();
    initial_params,
    kwargs...,
)
    # End recursion
    if isempty(varname_vecs) && isempty(samplers)
        return vnt, states, adopted_vecs
    end

    varnames, varname_vecs_tail... = varname_vecs
    sampler, samplers_tail... = samplers

    # Construct the conditioned model.
    conditioned_model = DynamicPPL.condition(model, conditioned_values(vnt, varnames))

    # Take initial step with the current sampler. Everything it does not sample is
    # conditioned, so it only initialises its own targets and can use its own strategy
    # directly -- `GibbsInitStrategy` picks an owner per variable, which is ambiguous when
    # components overlap. A user-supplied `initial_params` is passed through untouched.
    component_init = if initial_params isa GibbsInitStrategy
        init_strategy(sampler)
    else
        initial_params
    end
    _, new_state = step_function(
        rng,
        conditioned_model,
        sampler;
        # FIXME: This will cause issues if the sampler expects initial params in unconstrained space.
        # This is not the case for any samplers in Turing.jl, but will be for external samplers, etc.
        initial_params=component_init,
        kwargs...,
        discard_sample=true,
    )
    # New values for the variables this sampler is responsible for, plus any variable it
    # encountered that no component owns yet: both arrive in its own raw values.
    new_vnt = merge(vnt, gibbs_get_raw_values(new_state))
    adopted = adopt_new_variables(spl, sampler, VarName[], vnt, new_vnt)

    states = (states..., new_state)
    adopted_vecs = (adopted_vecs..., adopted)
    return gibbs_initialstep_recursive(
        rng,
        model,
        step_function,
        spl,
        varname_vecs_tail,
        samplers_tail,
        new_vnt,
        states,
        adopted_vecs;
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

    vnt, states, adopted = gibbs_step_recursive(
        rng,
        model,
        AbstractMCMC.step,
        spl,
        varnames,
        samplers,
        states,
        state.adopted,
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
    return transition, GibbsState(vnt, states, adopted)
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

    vnt, states, adopted = gibbs_step_recursive(
        rng,
        model,
        AbstractMCMC.step_warmup,
        spl,
        varnames,
        samplers,
        states,
        state.adopted,
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
    return transition, GibbsState(vnt, states, adopted)
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
    spl::Gibbs,
    varname_vecs,
    samplers,
    states,
    adopted_vecs,
    global_vnt,
    new_states=(),
    new_adopted=();
    kwargs...,
)
    # End recursion.
    if isempty(varname_vecs) && isempty(samplers) && isempty(states)
        return global_vnt, new_states, new_adopted
    end

    varnames, varname_vecs_tail... = varname_vecs
    sampler, samplers_tail... = samplers
    state, states_tail... = states
    adopted, adopted_tail... = adopted_vecs

    # A component samples what it was declared for plus anything it has adopted.
    targets = isempty(adopted) ? varnames : vcat(varnames, adopted)
    # Construct the conditional model that this sampler should use.
    conditioned_model = DynamicPPL.condition(model, conditioned_values(global_vnt, targets))
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
    adopted = adopt_new_variables(spl, sampler, adopted, global_vnt, new_global_vnt)

    new_states = (new_states..., new_state)
    new_adopted = (new_adopted..., adopted)
    return gibbs_step_recursive(
        rng,
        model,
        step_function,
        spl,
        varname_vecs_tail,
        samplers_tail,
        states_tail,
        adopted_tail,
        new_global_vnt,
        new_states,
        new_adopted;
        kwargs...,
    )
end
