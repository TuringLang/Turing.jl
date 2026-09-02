#
# Gibbs sampling
#

# Gibbs partitions the model's variables into blocks, one component sampler each, and sweeps
# the blocks in turn: to step a block, the model is `condition`ed on the current values of
# every variable outside it, so the component samples that block's full conditional.
#
# Those values are threaded explicitly as an immutable `VarNamedTuple`. Nothing is written in
# place, and nothing accumulates: after each component's step the snapshot is the set of
# variables a model evaluation reaches at the proposed values, so a variable the model stopped
# reaching leaves the snapshot instead of lingering at a stale value.
#
# Reading what to condition off that snapshot's keys is equivalent to deciding it per tilde
# statement, where the model's own `VarName` is available, so long as two things hold: no
# component is given part of a value the snapshot stores as a unit, and a change in the set of
# variables the model reaches belongs to the component that made it. Neither is a blanket rule
# about the declared varnames -- one component may own `x[1]` while another owns `x` and still
# partition cleanly, when the values keep an element per key -- so what is enforced is the
# expressibility of each particular case. The contract a user meets is in the `Gibbs` and
# `allow_varying_dimension` docstrings; why each holds is beside the code that enforces it, in
# `conditioned_values` and `check_variable_set`.
#
# Neither is decidable in advance: a block's shape is discovered while stepping it, so Gibbs
# cannot tell beforehand which variables a sweep will reach. Knowing that would need the
# model's trace type separated from its values.

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

Whether `spl` can move between supports *within* one of its own steps, that is, sample a block
whose set of variables its own proposal changes.

Defaults to `false`, because proposing between two supports takes a construction built for it.
A Metropolis-Hastings ratio between states of different dimension is not the one the algorithm
assumes, and a `LogDensityFunction`'s layout is fixed for the whole of a step, so it has no
slot for a variable the proposal reaches part-way through: a leapfrog step that crosses into
another branch raises `KeyError` from DynamicPPL rather than reaching this check at all. `PG`
and `CSMC` rebuild their trace each sweep, drawing whatever the model reaches, so they can.

This is a different question from whether a component can be handed a block at a new shape from
one sweep to the next, which no trait settles: it depends on whether the component caches a
layout, so it is checked where such a layout is reused, in
[`_require_same_block_shape`](@ref). `MH` answers `false` here and yet samples a block whose
shape changes between sweeps perfectly well.

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

# Whichever sampler answered `allow_varying_dimension`, which `ExternalSampler` does not
# forward. Unwrapping there would name a sampler that may well declare the trait it is being
# said to lack.
_trait_owner_name(spl::AbstractSampler) = nameof(typeof(spl))
_trait_owner_name(spl::RepeatSampler) = _trait_owner_name(spl.sampler)
_trait_owner_name(spl::ExternalSampler) = "externalsampler($(nameof(typeof(spl.sampler))))"

"""
    Turing.Inference.gibbs_get_parameter_values(state)

Return a `VarNamedTuple` containing the parameter values of all variables in the sampler
state.

Turing's Gibbs sampler maintains, at all points during the sampling process, a single global
`VarNamedTuple` that contains the **raw** values for all variables in the model. During the
sampling process, it calls each component sampler in turn and rebuilds that `VarNamedTuple`
around the new raw values returned by each sampler.

This function is used to pass that information *from* a component sampler *to* the Gibbs
sampler. Note that this means that the `VarNamedTuple` returned by this function should
**only** contain raw values for the variables that the component sampler is responsible for
sampling, and should not contain any values for other variables. In particular it must leave
out `:=` quantities: they are not variables, so Gibbs would take one appearing inside a branch
for a variable that appeared mid-run. `DynamicPPL.get_parameter_values` returns exactly the
`~` values of a state whose accumulator holds both.

A step need not reach every variable the component samples: a target inside a branch the model
did not take has no value that sweep, and leaving it out is the right answer. A component that
wants to remember such a value -- to reuse it if the branch comes back -- should keep it in its
own state and not report it. Gibbs does not read the report to decide which variables exist:
it evaluates the model (see [`reached_values`](@ref)), because existence is a property of the
model at the current values and not of a component's bookkeeping. Reporting a value for a
variable the model no longer reaches has no effect on the snapshot, and reporting one for a
variable the component does not sample is an error.
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

This requires that `old_ldf.model` (i.e., the model conditioned on the previous values) and
`model` (i.e., the model conditioned on the new values) reach the same variables, so that the
flat layout can be reused. [`_require_same_block_shape`](@ref) checks that rather than assuming
it, and throws with an explanation if another component's step has changed the block's shape.
The layout is reused rather than rebuilt because its ordering is what the component's own
state, such as a Hamiltonian metric, is indexed by.
"""
function gibbs_recompute_ldf_and_params(
    old_ldf::DynamicPPL.LogDensityFunction,
    model::DynamicPPL.Model,
    global_vals::DynamicPPL.VarNamedTuple,
    extra_accs::NTuple{N,<:DynamicPPL.AbstractAccumulator}=(),
) where {N}
    layout = DynamicPPL.get_all_ranges_and_transforms(old_ldf)
    _require_same_block_shape(layout, model, global_vals)
    # Built without `adtype` first, so that nothing is prepared at the previous step's point:
    # `layout` guarantees the same variables, not the same position, and a taped backend
    # records its tape wherever it is prepared. The second construction prepares at the
    # current point and is the only one that prepares at all.
    new_ldf = DynamicPPL.LogDensityFunction(
        model,
        DynamicPPL.get_logdensity_callable(old_ldf),
        layout,
        DynamicPPL.get_sample_input_vector(old_ldf);
        adtype=nothing,
    )
    accs = DynamicPPL.OnlyAccsVarInfo(
        DynamicPPL.VectorParamAccumulator(new_ldf), extra_accs...
    )
    init_strategy = DynamicPPL.InitFromParams(global_vals, nothing)
    _, accs = DynamicPPL.init!!(
        new_ldf.model, accs, init_strategy, new_ldf.transform_strategy
    )
    new_params = DynamicPPL.get_vector_params(accs)
    if old_ldf.adtype !== nothing
        new_ldf = DynamicPPL.LogDensityFunction(
            model,
            DynamicPPL.get_logdensity_callable(old_ldf),
            layout,
            new_params;
            adtype=old_ldf.adtype,
        )
    end
    return new_ldf, new_params, accs
end

"""
    _require_same_block_shape(layout, model, global_vals)

Throw unless the block `layout` describes still holds exactly the variables the component
samples.

A component reusing a `LogDensityFunction` keeps the flat layout its first step established,
and its adapted state -- a Hamiltonian metric, a step size -- is sized for that layout. So it
cannot be re-targeted at a block of a different shape, which is what another component's step
does when it decides whether one of this block's variables exists. Rebuilding the layout is not
enough on its own: `AdvancedHMC` then meets a position vector of one length and a metric of
another.

This is a different question from [`allow_varying_dimension`](@ref), which asks whether a
component may move between supports *within* its own step. A component can be unable to do
either, able to do both, or, like `MH`, unable to move between supports within a step yet
perfectly able to sample a block it is handed at a new shape each sweep -- `MH` caches no
layout, so it never reaches here.
"""
function _require_same_block_shape(
    layout::DynamicPPL.VarNamedTuple,
    model::DynamicPPL.Model,
    global_vals::DynamicPPL.VarNamedTuple,
)
    conditioned = DynamicPPL.conditioned(model)
    function fail(leaf, what)
        return throw(
            ArgumentError(
                "The variable $(leaf) $(what) the block this component samples between Gibbs " *
                "sweeps. A component that reuses a `LogDensityFunction` fixes the block's " *
                "layout at its first step and sizes its adapted state to it, so it cannot " *
                "sample a block whose shape changes. Put $(leaf) and the variables that " *
                "decide whether it exists in one block sampled by a component that rebuilds " *
                "its trace each sweep, such as `PG`, or sample this block with `MH`, which " *
                "caches no layout.",
            ),
        )
    end
    for vn in keys(layout)
        DynamicPPL.hasvalue(global_vals, vn) || fail(vn, "left")
    end
    # A layout holds one range per tilde statement, so its key can be a whole vector where the
    # values keep an element per leaf. Ask which key covers the leaf rather than looking the
    # leaf up, or a vector-valued `m ~` would look like `m[1]` having no slot.
    for vn in keys(global_vals), leaf in AbstractPPL.varname_leaves(vn, global_vals[vn])
        DynamicPPL.hasvalue(conditioned, leaf) && continue
        any(k -> AbstractPPL.subsumes(k, leaf), keys(layout)) && continue
        fail(leaf, "joined")
    end
    return nothing
end

#
# Gibbs implementation itself
#

"""
    reached_values(rng, model, proposed)

Return the values of the variables `model` reaches when evaluated at `proposed`.

This is how the snapshot after a component's step is built, rather than from the values the
component reports. Which variables exist is a property of the model at the current values, so
only an evaluation settles it, and a component's report cannot be trusted for it: a component
is free to keep a value for a variable it is not currently sampling, which is useful to it and
none of Gibbs's business, but were that value to enter the snapshot it would be conditioned
into the step of whichever component decides the variable's existence, and
[`check_variable_set`](@ref) would never see the variable come back.

The evaluation costs one model evaluation per component step, and draws from the prior only for
a variable `proposed` has no value for. That happens when a component under-reports a variable
it samples, or when a variable appeared during this step, and `check_variable_set` throws in
both cases -- so nothing that returns normally has consumed `rng`.
"""
function reached_values(rng, model, proposed::DynamicPPL.VarNamedTuple)
    accs = DynamicPPL.OnlyAccsVarInfo(DynamicPPL.RawValueAccumulator(false))
    _, accs = DynamicPPL.init!!(
        rng,
        model,
        accs,
        DynamicPPL.InitFromParams(proposed, DynamicPPL.InitFromPrior()),
        DynamicPPL.UnlinkAll(),
    )
    return DynamicPPL.get_raw_values(accs)
end

"""
    conditioned_values(global_vnt, target_variables)

Return the values in `global_vnt` for every variable *not* sampled by this Gibbs component,
i.e. the ones it conditions on.

A component may own part of a value the model stores as a unit, which cannot be expressed:
under `x ~ MvNormal(zeros(2), I)` the values hold `x` as one key, so freeing it for a component
that samples `x[1]` would free `x[2]` too. This throws rather than hand a component a larger
block than it owns. Whether a partition is expressible is a property of the model's tilde
statements and not of the samplers -- element-wise `x[1] ~` and `x[2] ~` give a key each and
split cleanly, because the snapshot comes from a model evaluation rather than from what a
component reports. Sampling a genuinely unsplittable block means writing a sampler against the
AbstractMCMC interface directly.

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

Variables the model only reaches on some sweeps need care:

```julia
@model function f()
    x ~ Normal()
    y ~ Normal()
    if x > 0
        z ~ Normal()
    end
end

# `x` decides whether `z` exists, so they share a block, sampled by `PG`, which can handle a
# set of variables that changes between sweeps.
sample(f(), Gibbs(@varname(y) => MH(), (@varname(x), @varname(z)) => PG(20)), 1000)
```

Three partitions of that model are rejected, each naming the variable and the components:

  - `Gibbs(@varname(x) => MH(), @varname(y) => MH())` -- nothing samples `z`. Every variable the
    model reaches must belong to a component, whether or not it is always reached.
  - `Gibbs(@varname(x) => MH(), @varname(y) => MH(), @varname(z) => PG(20))` -- `z` is split from
    `x`, which decides whether it exists. The `x` component's step is what makes `z` come and go,
    so that step proposes between states with different sets of variables while a *different*
    component samples `z`, leaving that one conditioning on a `z` the proposed state does not
    have. The chain still runs but comes back biased, which is why this is refused rather than
    warned about. Giving `z` to `PG` does not help: it is never asked, because it is not the
    component whose step changed anything.
  - `Gibbs(@varname(y) => MH(), (@varname(x), @varname(z)) => MH())` -- the block is right, but
    `MH` fixes the set of variables it samples at its first step, and its acceptance ratio
    between two different supports is not the one the algorithm assumes.

The same applies to a new element rather than a new name: an `x[5]` first reached inside a
branch is treated exactly like `z` above.

When each is caught varies, and cannot be made uniform. A component that cannot serve in Gibbs
at all is refused when the `Gibbs` sampler is constructed, and anything already visible in the
first draw is refused at the first step. The rest can only surface while sampling, because
whether `z` exists depends on `x`, which is drawn during the run: the same partition may be
refused at the first sweep or fifty sweeps in, depending on the draws. Sampling stops with an
error at that point rather than carrying on, since the chain from an invalid partition is
biased rather than merely noisy.

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
# A `Gibbs` component would need `gibbs_get_parameter_values` and `gibbs_update_state!!` for
# `GibbsState`, which do not exist; without this, nesting fails with a `MethodError` several
# steps in rather than at construction.
supports_gibbs(::Gibbs) = false
# TODO(penelopeysm): `allow_discrete_variables(::Gibbs)` could be smarter, since a component
# may not allow discrete variables even though Gibbs itself does.

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

function _require_varying_dimension(sampler, leaf, what::String)
    allow_varying_dimension(sampler) && return nothing
    # The name has to be whichever sampler answered the trait, which is not always the one
    # inside: `RepeatSampler` forwards `allow_varying_dimension` but `ExternalSampler` does
    # not, so naming the wrapped sampler there would state the opposite of what it declares.
    # The reason a given sampler cannot do this is per-sampler, so it lives in
    # `allow_varying_dimension`'s docstring rather than here.
    name = _trait_owner_name(sampler)
    remedy = if sampler isa ExternalSampler
        # Suggesting a component that declares the trait would be useless: the wrapper
        # overrides whatever it wraps.
        "`externalsampler` fixes the parameter layout whatever it wraps, so sampling this " *
        "block takes a sampler used through the `AbstractMCMC` interface directly rather " *
        "than wrapped."
    else
        "Put $(leaf) and the variables that decide whether it exists in one block, sampled " *
        "by a component that can, such as `PG`."
    end
    return throw(
        ArgumentError(
            "The variable $(leaf) $(what) during a step of $(name), so that step is " *
            "proposing between states with different sets of variables, which $(name) does " *
            "not declare it can sample (see `allow_varying_dimension`). " *
            remedy,
        ),
    )
end

"""
    check_variable_set(spl, sampler, varnames, old_vnt, new_vnt, proposed)

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

Both directions are checked, and symmetrically: `old_vnt` and `new_vnt` are both the set of
variables a model evaluation reached (see [`reached_values`](@ref)), so a leaf in one and not
the other came or went during this step, and nothing else can have caused it -- every variable
outside the component's block was conditioned.

Taking both sets from the model, rather than from what the component reported, is what makes
the verdict independent of the draw Gibbs initialises with and of the component's own
bookkeeping. A component is free to keep a value for a variable it is not currently sampling;
were that value taken into the snapshot it would be conditioned into the step of whichever
component decides the variable's existence, that step would never assume it again, and the
appearance branch would be dead for it -- so a split partition would be rejected only for
those seeds whose initialising draw happened to miss the variable.
"""
function check_variable_set(
    spl::Gibbs,
    sampler,
    varnames,
    old_vnt::DynamicPPL.VarNamedTuple,
    new_vnt::DynamicPPL.VarNamedTuple,
    proposed::DynamicPPL.VarNamedTuple,
)
    # Walk to the leaves rather than compare keys: an array stored under a single key can grow
    # or lose an element, and an `x[5]` inside a branch matters as much as a whole `z`.
    for (from, to, what) in
        ((new_vnt, old_vnt, "appeared"), (old_vnt, new_vnt, "stopped existing"))
        for vn in keys(from), leaf in AbstractPPL.varname_leaves(vn, from[vn])
            DynamicPPL.hasvalue(to, leaf) && continue
            _require_owned(spl, sampler, varnames, leaf, what)
            _require_varying_dimension(sampler, leaf, what)
        end
    end
    # Anything the model reaches that the component did not report was drawn from the prior by
    # `reached_values`, just to let the evaluation finish. The loops above have already ruled
    # on whether the variable may come or go, so what is left is a component that sampled a
    # variable and failed to return it, and its value would silently be a prior draw.
    for vn in keys(new_vnt), leaf in AbstractPPL.varname_leaves(vn, new_vnt[vn])
        DynamicPPL.hasvalue(proposed, leaf) && continue
        throw(
            ArgumentError(
                "The component sampling $(join(varnames, ", ")) did not report a value for " *
                "$(leaf), which it samples and which the model reaches, so the value would " *
                "be a draw from the prior. `gibbs_get_parameter_values` has to return every " *
                "variable the component sampled this step.",
            ),
        )
    end
    return nothing
end

"""
    _require_owned(spl, sampler, varnames, leaf, what)

Throw unless `leaf`, whose existence just changed, is sampled by the component that changed it.

A variable coming or going under one component's proposal while another samples it means that
other component conditions on a variable absent from the state being proposed: the two blocks
disagree about the support, and the chain comes back biased even though both samplers can
handle varying dimension on their own.
"""
function _require_owned(spl::Gibbs, sampler, varnames, leaf, what::String)
    any(hv -> AbstractPPL.subsumes(hv, leaf), varnames) && return nothing
    owner = findfirst(vns -> any(hv -> AbstractPPL.subsumes(hv, leaf), vns), spl.varnames)
    return throw(
        ArgumentError(
            if owner === nothing
                "The variable $(leaf) $(what) during a step of " *
                "$(_component_name(sampler)) and no component samples it, so every " *
                "component would condition on a single draw of it for the rest of the run. " *
                "Assign it to a component."
            else
                # Name the components by what they sample, not by their sampler type: two
                # components of the same type would otherwise give "during a step of MH,
                # which does not sample it: MH does".
                "The variable $(leaf) $(what) during a step of the component sampling " *
                "$(join(varnames, ", ")), which does not sample it. The component sampling " *
                "$(join(spl.varnames[owner], ", ")) does. Whichever component's step " *
                "decides whether $(leaf) exists has to be the one that samples it, or that " *
                "other component conditions on a variable the state being proposed does " *
                "not have, and the chain comes back biased. Put $(leaf) and the variables " *
                "that decide whether it exists in one block."
            end,
        ),
    )
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
    proposed = merge(conditioned, gibbs_get_parameter_values(new_state))
    new_vnt = reached_values(rng, model, proposed)
    check_variable_set(spl, sampler, varnames, vnt, new_vnt, proposed)

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
# The snapshot after a component's step is the set of variables the model reaches at the
# proposed values, not the component's report merged into the old snapshot. Only an evaluation
# settles which variables exist, and a report cannot be trusted for it: a component may keep a
# value for a variable it is not currently sampling, and `merge` would leave that stale value
# in the snapshot, conditioned into the step of whichever component decides its existence, so
# that step would never assume it again and `check_variable_set` could never see it reappear.
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

    proposed = merge(conditioned, gibbs_get_parameter_values(new_state))
    new_global_vnt = reached_values(rng, model, proposed)
    check_variable_set(spl, sampler, varnames, global_vnt, new_global_vnt, proposed)

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
