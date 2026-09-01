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
#     Every variable of the model has to be declared for a component, and a change in the set
#     of variables the model reaches is allowed only when both of these hold: the component
#     taking the step declares `allow_varying_dimension`, and the variable that came or went is
#     one that same component samples. The first holds because during its step everything it
#     does not sample is conditioned, so only its own proposal can have moved; the second
#     because otherwise the component that does sample the variable conditions on one the
#     proposed state lacks. Together: the variable and whatever decides its existence belong in
#     one block. `check_variable_set` checks appearance and disappearance alike, which is why
#     the result does not depend on whether the initialising draw happened to reach it.
#
#     That rejects the partitions where the deciding variable sits in another block. Under
#     `Gibbs(@varname(b) => MH(), @varname(θ) => PG(20))` on a model where `b` decides whether
#     `θ[2]` exists, `MH`'s step is what makes `θ[2]` come and go, so `MH` is asked and refuses;
#     asking only `θ`'s declared `PG` let the chain come back biased (P(b=1) = 0.11 against an
#     exact 0.39). `Gibbs((@varname(b), @varname(θ)) => PG(20))` is the correct partition.
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

Defined in terms of `supports_gibbs` rather than as a constant `true`, so that a wrapper
delegating through this name -- `isgibbscomponent(w::MyWrapper) = isgibbscomponent(w.inner)` --
still gets the right answer for a sampler Turing declares unusable.
"""
isgibbscomponent(spl::AbstractSampler) = supports_gibbs(spl)

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

Returning `true` is a claim about the algorithm, and it carries an obligation: the component
must have coherent semantics for a variable it samples going away and later coming back. `PG`
and `CSMC` do. The reference particle replays the retained trajectory's *values*, so it
reproduces that execution exactly and cannot reach an address the trajectory lacks, which it
would otherwise refuse; the remaining particles draw from the prior and are free to reach a
different set of addresses, which is what lets the block move between supports at all.

Returning `true` is not by itself enough. For an array whose length varies
(`for i in 1:n; x[i] ~ ...; end` under a random `n`), the block has to hold `n` as well: with
`n` in another component, the conditioned `x[i]` become observations whose number depends on
it, and `PG` refuses with "the number of observations must not be random".
"""
allow_varying_dimension(::AbstractSampler) = false
# `RepeatSampler` hands `gibbs_update_state!!` to the sampler it wraps, so the inner answer is
# the right one. `ExternalSampler` does not: its own `gibbs_update_state!!` goes through
# `gibbs_recompute_ldf_and_params`, so it keeps the `false` default whatever it wraps.
allow_varying_dimension(spl::RepeatSampler) = allow_varying_dimension(spl.sampler)
allow_varying_dimension(::PG) = true

# For an error about an unsupported component, the wrapper's own name says nothing: both
# wrappers forward `supports_gibbs`, so the sampler that answered is the one inside.
_component_name(spl::AbstractSampler) = nameof(typeof(spl))
_component_name(spl::Union{RepeatSampler,ExternalSampler}) = _component_name(spl.sampler)

"""
    Turing.Inference.gibbs_get_parameter_values(state)

Return a `VarNamedTuple` containing the parameter values of all variables in the sampler
state.

Turing's Gibbs sampler maintains, at all points during the sampling process, a single global
`VarNamedTuple` that contains the **raw** values for all variables in the model. During the
sampling process, it calls each component sampler in turn and updates the global
`VarNamedTuple` with the new raw values returned by each sampler.

This function is used to pass that information *from* a component sampler *to* the Gibbs
sampler. Note that this means that the `VarNamedTuple` returned by this function should
**only** contain raw values for the variables that the component sampler is responsible for
sampling, and should not contain any values for other variables. In particular it must leave
out `:=` quantities: they are not variables, so Gibbs would take one appearing inside a branch
for a variable that appeared mid-run. `DynamicPPL.get_parameter_values` returns exactly the
`~` values of a state whose accumulator holds both.

A step need not reach every variable the component samples: a target inside a branch the model
did not take has no value that sweep. Both answers are allowed -- leave it out, or keep the
value the component last held, if the algorithm accounts for variables it is not currently
sampling -- but the choice is what Gibbs sees. It reads any change in the reported set as a
change of support and asks [`allow_varying_dimension`](@ref) of the component that made it, so
a component that keeps inactive values never trips that check and owns the correctness of
doing so. Gibbs infers nothing further from presence or absence.
"""
function gibbs_get_parameter_values(state)
    # This has to be the only entry point, not one method among several: a method specialised
    # on the state type would win dispatch and the deprecated name would never be consulted.
    # An `isgibbscomponent`-era overload is any method more specific than the forwarder below.
    if which(gibbs_get_raw_values, Tuple{typeof(state)}) !== _GIBBS_GET_RAW_VALUES_FORWARDER
        Base.depwarn(
            "`Turing.Inference.gibbs_get_raw_values` is deprecated, define " *
            "`Turing.Inference.gibbs_get_parameter_values` instead.",
            :gibbs_get_parameter_values,
        )
        return gibbs_get_raw_values(state)
    end
    return _default_parameter_values(state)
end

"""
    Turing.Inference._default_parameter_values(state)

The answer for a state with no `gibbs_get_parameter_values` method of its own.

An `AbstractVarInfo` carries its `~` values in a `RawValueAccumulator`, so there is one; any
other state has to say for itself.
"""
_default_parameter_values(state) = throw(MethodError(gibbs_get_parameter_values, (state,)))
_default_parameter_values(state::AbstractVarInfo) = DynamicPPL.get_parameter_values(state)

"""
    Turing.Inference.gibbs_get_raw_values(state)

Deprecated name for [`gibbs_get_parameter_values`](@ref), still honoured so that a sampler
written against it keeps working. Calls through to the new name, so a state that defines only
the new one can still be asked by the old.
"""
gibbs_get_raw_values(state) = gibbs_get_parameter_values(state)

# The forwarder above is what `gibbs_get_parameter_values` compares against: any method more
# specific than it is a sampler's own, written against the old name.
const _GIBBS_GET_RAW_VALUES_FORWARDER = which(gibbs_get_raw_values, Tuple{Any})

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
                msg = "All samplers must be valid Gibbs components, $(_component_name(spl)) is not."
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
        clashes = filter(in(keys(stats)), names)
        isempty(clashes) || throw(
            ArgumentError(
                "Gibbs cannot name the statistics of these components apart: " *
                "$(join(clashes, ", ")) would stand for two different diagnostics, because " *
                "joining a component's variables and a statistic's name with `_` does not " *
                "distinguish `$(prefix)` from another component whose name and statistic " *
                "concatenate to the same thing. Rename the model's variables, or the " *
                "sampler's statistics, so the two differ.",
            ),
        )
        stats = merge(stats, NamedTuple{names}(values(component)))
    end
    return stats
end

"""
    check_variable_set(spl, sampler, varnames, old_vnt, component_vnt)

Check that a change in the set of variables is one the component that made it may make.

Two conditions have to hold together, and anything else throws:

  - the component taking the step declares [`allow_varying_dimension`](@ref); and
  - the variable that came or went is one that same component samples.

The first is needed because during a component's step every variable it does not sample is
conditioned, and so cannot move: if the set of variables the model reaches changed, it was that
component's own proposal that moved between two different supports. The second is needed
because a variable coming and going under one component's proposal while another component
samples it means that other component conditions on a variable absent from the state being
proposed -- the two blocks disagree about the support, and the chain comes back biased even
though both samplers can handle varying dimension on their own.

Together they say: the variable and whatever decides whether it exists belong in one block,
sampled by a component built for it. The cost of getting this wrong is not a crash but a wrong
answer: on the `dynamic_bernoulli_normal` model in the tests, where `b` decides whether `θ[2]`
exists, `Gibbs(@varname(b) => MH(), @varname(θ) => PG(20))` used to sample and return
P(b=1) between 0.0 and 0.18 against an exact 0.394, while
`Gibbs((@varname(b), @varname(θ)) => PG(20))` gives 0.38.

Both directions are checked. A leaf the component reports that was not in the snapshot has
appeared; a leaf of `varnames` that was in the snapshot and the component no longer reports has
gone.

The verdict does not depend on the draw Gibbs initialises with, and that rests on the snapshot
being rebuilt after each step rather than merged into (see the note above
[`gibbs_step_recursive`](@ref)). A leaf a component stops reporting leaves the snapshot, so
when the model reaches it again it is *assumed* during the step of whichever component decides
its existence, and shows up in that component's own report -- where the appearance branch sees
it and names the right component. Were the snapshot merged into instead, the leaf would linger
at a stale value, be conditioned into that component's step forever, and the appearance branch
would be dead for it: a split partition would then be rejected only for those seeds whose
initialising draw happened to miss the variable.
"""
function _require_varying_dimension(sampler, leaf, what::String)
    allow_varying_dimension(sampler) && return nothing
    name = nameof(typeof(sampler))
    return throw(
        ArgumentError(
            "The variable $(leaf) $(what) during a step of $(name), so that step is " *
            "proposing between states with different sets of variables. $(name) fixes the " *
            "set of variables it samples at its first step, and its acceptance ratio " *
            "between two different supports is not the one the algorithm assumes. Put " *
            "$(leaf) and the variables that decide whether it exists in one block, sampled " *
            "by a component that can handle a varying set of variables, such as `PG`.",
        ),
    )
end

function check_variable_set(
    spl::Gibbs,
    sampler,
    varnames,
    old_vnt::DynamicPPL.VarNamedTuple,
    component_vnt::DynamicPPL.VarNamedTuple,
)
    # Walk to the leaves rather than compare keys: an array stored under a single key can grow
    # or lose an element, and an `x[5]` inside a branch matters as much as a whole `z`.
    for vn in keys(component_vnt), leaf in AbstractPPL.varname_leaves(vn, component_vnt[vn])
        DynamicPPL.hasvalue(old_vnt, leaf) && continue
        if !any(hv -> AbstractPPL.subsumes(hv, leaf), varnames)
            owner = findfirst(
                vns -> any(hv -> AbstractPPL.subsumes(hv, leaf), vns), spl.varnames
            )
            throw(
                ArgumentError(
                    if owner === nothing
                        "The variable $(leaf) appeared during a step of " *
                        "$(nameof(typeof(sampler))) and no component samples it, so every " *
                        "component would condition on a single draw of it for the rest of " *
                        "the run. Assign it to a component."
                    else
                        "The variable $(leaf) appeared during a step of " *
                        "$(nameof(typeof(sampler))), which does not sample it: " *
                        "$(_component_name(spl.samplers[owner])) does. The component whose " *
                        "step decides whether $(leaf) exists has to be the one that samples " *
                        "it, or that other component conditions on a variable the state " *
                        "being proposed does not have, and the chain comes back biased. Put " *
                        "$(leaf) and the variables that decide whether it exists in one " *
                        "block."
                    end,
                ),
            )
        end
        _require_varying_dimension(sampler, leaf, "appeared")
    end
    for vn in keys(old_vnt), leaf in AbstractPPL.varname_leaves(vn, old_vnt[vn])
        any(hv -> AbstractPPL.subsumes(hv, leaf), varnames) || continue
        DynamicPPL.hasvalue(component_vnt, leaf) && continue
        _require_varying_dimension(sampler, leaf, "stopped existing")
    end
    return nothing
end

"""
    check_all_variables_handled(vnt, spl::Gibbs)

Check that every variable in `vnt` belongs to a component.

A key of `vnt` no declared varname subsumes is examined leaf by leaf, because ownership of a
value stored as one key is a property of its leaves. One component owning every leaf can be
handed the whole value, so that partition is fine; several owning parts of it is one Gibbs
cannot express, since freeing one part of a single stored value is exactly what
[`conditioned_values`](@ref) cannot do; and a leaf no component owns is one the user left out.
"""
function check_all_variables_handled(vnt::DynamicPPL.VarNamedTuple, spl::Gibbs)
    missing_vars = VarName[]
    split_vars = VarName[]
    for vn in keys(vnt)
        any(hv -> AbstractPPL.subsumes(hv, vn), Iterators.flatten(spl.varnames)) && continue
        owners = Set{Int}()
        for leaf in AbstractPPL.varname_leaves(vn, vnt[vn])
            owner = findfirst(
                vns -> any(hv -> AbstractPPL.subsumes(hv, leaf), vns), spl.varnames
            )
            owner === nothing ? push!(missing_vars, leaf) : push!(owners, owner)
        end
        length(owners) > 1 && push!(split_vars, vn)
    end
    if !isempty(missing_vars) || !isempty(split_vars)
        if isempty(missing_vars)
            throw(
                ArgumentError(
                    "Components claim parts of $(join(split_vars, ", ")) separately, but " *
                    "the model stores each as one value, so Gibbs cannot free one part and " *
                    "condition on the rest. Give the whole variable to a single component.",
                ),
            )
        end
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

Return the values the sweep starts from, and check that every one of them has a component.
"""
function gibbs_initial_values(rng, model, spl::Gibbs, initial_params)
    accs = DynamicPPL.OnlyAccsVarInfo(DynamicPPL.RawValueAccumulator(false))
    _, accs = DynamicPPL.init!!(rng, model, accs, initial_params, DynamicPPL.UnlinkAll())
    vnt = DynamicPPL.get_raw_values(accs)
    check_all_variables_handled(vnt, spl)
    return vnt
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

    vnt, states = gibbs_initialstep_recursive(
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
    vnt = gibbs_initial_values(rng, model, spl, initial_params)

    vnt, states = gibbs_initialstep_recursive(
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
    spl::Gibbs,
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
    conditioned = conditioned_values(vnt, varnames)
    conditioned_model = DynamicPPL.condition(model, conditioned)

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
    component_vnt = gibbs_get_parameter_values(new_state)
    check_variable_set(spl, sampler, varnames, vnt, component_vnt)
    new_vnt = merge(conditioned, component_vnt)

    states = (states..., new_state)
    return gibbs_initialstep_recursive(
        rng,
        model,
        step_function,
        spl,
        varname_vecs_tail,
        samplers_tail,
        new_vnt,
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
        rng, model, AbstractMCMC.step, spl, varnames, samplers, states, state.vnt; kwargs...
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
        spl,
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
# ──────────────────────────────────────────────────────────────────────────────
# The snapshot after a component's step is what it conditioned on plus what it reported --
# rebuilt, not merged into. `merge` only ever overwrites a key, so a variable the component
# stopped reaching would keep a stale value: it would then be conditioned into the step of
# whichever component decides its existence, so that step would never assume it again and
# `check_variable_set` could never see it reappear. Rebuilding lets it leave the snapshot, so
# the sweep that brings it back is an appearance in the deciding component's own report.
# ──────────────────────────────────────────────────────────────────────────────

function gibbs_step_recursive(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    step_function::Function,
    spl::Gibbs,
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
    conditioned = conditioned_values(global_vnt, varnames)
    conditioned_model = DynamicPPL.condition(model, conditioned)
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
    component_vnt = gibbs_get_parameter_values(new_state)
    check_variable_set(spl, sampler, varnames, global_vnt, component_vnt)
    new_global_vnt = merge(conditioned, component_vnt)

    new_states = (new_states..., new_state)
    return gibbs_step_recursive(
        rng,
        model,
        step_function,
        spl,
        varname_vecs_tail,
        samplers_tail,
        states_tail,
        new_global_vnt,
        new_states;
        kwargs...,
    )
end
