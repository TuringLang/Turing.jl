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

# TODO(#2863): remove the `isgibbscomponent` bridge, and the `gibbs_get_raw_values` one below,
# once the deprecation period is over. Consulting a deprecated name from the new one keeps a
# third-party overload working, but it means two names can disagree until then.
#
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
A `LogDensityFunction`'s layout is fixed for the whole of a step, so it has no slot for a
variable the proposal reaches part-way through: a leapfrog step that crosses into another
branch raises `KeyError` from DynamicPPL rather than reaching this check at all. `PG` and
`CSMC` rebuild their trace each sweep, drawing whatever the model reaches, so they can.

`MH` answers `true`, but not because every crossing is safe for it. Whether one is depends on
the variable that moved: drawn from its prior, the proposal density cancels against the prior
and the ratio collapses to a likelihood ratio, which is defined across dimensions; proposed
from, it does not cancel. That is a fact about one variable in one evaluation, which a trait
asked once cannot supply, so `MH` answers `true` here and refuses the invalid crossings itself.
A component whose answer really is uniform should give it here.

This is a different question from being *handed* a block at a new shape between one's own
steps, which another component's step can do and which a component answers by implementing
[`gibbs_update_state!!`](@ref) for a [`ReshapedBlock`](@ref).

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
# `RepeatSampler` hands the step to the sampler it wraps, so the inner answer is the right one.
# `ExternalSampler` keeps the `false` default whatever it wraps: the wrapped state is opaque, so
# a claim made about the sampler inside says nothing about what survives the wrapper.
allow_varying_dimension(spl::RepeatSampler) = allow_varying_dimension(spl.sampler)
allow_varying_dimension(::PG) = true
# Whether a crossing is valid for `MH` is decided per variable and per evaluation: a variable
# drawn from its prior cancels against `p` in the acceptance ratio whatever its dimension, one
# proposed from does not. A trait asked once, before any evaluation, cannot know which applies,
# and reading the declared proposal keys instead is wrong in both directions, since a key may
# match no tilde statement and so be declared without ever being used. So `MH` answers `true`
# and polices each crossing itself in `_require_same_variables`, where the evaluations have
# recorded which proposals were used.
allow_varying_dimension(::MH) = true

# For an error about an unsupported component, the wrapper's own name says nothing: both
# wrappers forward `supports_gibbs`, so the sampler that answered is the one inside.
_component_name(spl::AbstractSampler) = nameof(typeof(spl))
_component_name(spl::Union{RepeatSampler,ExternalSampler}) = _component_name(spl.sampler)

# Whichever sampler is the constraint, which for `ExternalSampler` is the wrapper itself: it
# forwards neither `allow_varying_dimension` nor `gibbs_update_state!!` for a `ReshapedBlock`,
# so unwrapping would name a sampler that may well be capable of what it is said to lack.
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

See also the five-argument form, [`gibbs_update_state!!(sampler, state, model, global_vals,
reshaped::ReshapedBlock)`](@ref), which Gibbs calls in place of this one when the block has a
different set of variables from the one the sampler last stepped.
"""
function gibbs_update_state!! end

"""
    Turing.Inference.gibbs_update_state!!(
        sampler::AbstractSampler, state, model::Model, global_vals::VarNamedTuple,
        reshaped::ReshapedBlock
    )

Update the state of a Gibbs component sampler whose block now holds a different set of
variables from the one it last stepped.

That happens when another component samples the variable deciding whether one of this block's
variables exists:

```julia
@model function f(y)
    b ~ Bernoulli(0.3)
    θ = zeros(2)
    θ[1] ~ Normal()
    b == 1 && (θ[2] ~ Normal())
    return y ~ Normal(sum(θ), 0.5)
end

Gibbs((@varname(b), @varname(θ)) => PG(20), @varname(θ) => HMC(0.1, 5))
```

`PG` samples `b` and `θ` together, then `HMC` samples `θ` again. When `PG`'s step flips `b`,
`θ[2]` appears or vanishes, so by the time `HMC` next steps its block has a different shape
from the one it last saw. Each step still sees one fixed shape, so the scheme is valid; `HMC`
only has to rebuild the parameter layout and phasepoint it cached for the old shape. Without a
flip, `θ` never changes shape and the case never arises.

Defaults to throwing, because most components carry something so sized: an adapted mass
matrix, a flat parameter layout, a set of prior means. Implementing this method is how a
sampler says that it copes -- the implementation is the declaration, so there is no separate
trait that could fall out of step with what the sampler actually does, and a sampler written
before this method existed keeps the safe answer.

Turing's own answers: `MH` and `PG` delegate to the four-argument form, caching nothing shaped
like the block. `HMC` and the other [`StaticHamiltonian`](@ref)s rebuild both the parameter
layout and the phasepoint. `NUTS` and `HMCDA` do not implement it, since `gen_metric` renews an
[`AdaptiveHamiltonian`](@ref)'s metric from `state.adaptor`, whose mass matrix is sized for the
shape it adapted to; resetting the adaptation instead would discard it mid-chain, which is a
decision for the user rather than a detail of conditioning. Nor does `ESS`, whose prior means
are gathered for the block it was built on, nor `externalsampler`, whose wrapped state is
opaque.
"""
function _reshape_description(r::ReshapedBlock)
    r.change === :joined && return "$(r.variable) joined it"
    r.change === :left && return "$(r.variable) left it"
    return "$(r.variable) is now drawn from a distribution that links differently"
end

function gibbs_update_state!!(
    sampler, state, model::DynamicPPL.Model, global_vals, reshaped::ReshapedBlock
)
    name = _trait_owner_name(sampler)
    return throw(
        ArgumentError(
            "The block sampled by $(name) changed shape between its steps: " *
            _reshape_description(reshaped) *
            ", because another component's step decides that. $(name) does not implement " *
            "`gibbs_update_state!!` for a reshaped block, so state it carries would still " *
            "be sized for the old one. Sample this block with `MH`, `HMC` or `PG`, which " *
            "do, or put $(reshaped.variable) and whatever decides its shape in one block.",
        ),
    )
end
function gibbs_update_state!!(
    sampler::RepeatSampler, state, model::DynamicPPL.Model, global_vals, r::ReshapedBlock
)
    return gibbs_update_state!!(sampler.sampler, state, model, global_vals, r)
end

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

The flat layout is built from `global_vals` rather than reused from `old_ldf`. A layout is keyed
by tilde statement and carries each variable's transform, so a reused one is stale the moment
the model takes a different branch -- not only when a variable joins or leaves the block, but
also when the same variables arrive under different keys, or with a different transform and so
a different linked dimension. Evaluating against a stale layout fails inside DynamicPPL naming
no variable. Built from the values, the layout is identical whenever the execution path is,
which is exactly when the component's own state depends on the ordering, and differs only when
that state was going to be wrong anyway.
"""
function gibbs_recompute_ldf_and_params(
    old_ldf::DynamicPPL.LogDensityFunction,
    model::DynamicPPL.Model,
    global_vals::DynamicPPL.VarNamedTuple,
    extra_accs::NTuple{N,<:DynamicPPL.AbstractAccumulator}=(),
) where {N}
    init_strategy = DynamicPPL.InitFromParams(global_vals, nothing)
    # Establish the layout the newly conditioned model actually has, at the current values.
    # `InitFromParams` with no fallback draws nothing, so this consumes no randomness -- unlike
    # building the layout through the `transform_strategy` constructor, which draws from the
    # prior with whatever the default rng happens to be.
    probe = DynamicPPL.OnlyAccsVarInfo(DynamicPPL.VectorValueAccumulator())
    _, probe = DynamicPPL.init!!(model, probe, init_strategy, old_ldf.transform_strategy)
    # Built without `adtype` first, so that nothing is prepared at a point we are about to
    # replace: a taped backend records its tape wherever it is prepared. The construction
    # below prepares at the current point and is the only one that prepares at all.
    new_ldf = DynamicPPL.LogDensityFunction(
        model, DynamicPPL.get_logdensity_callable(old_ldf), probe; adtype=nothing
    )
    accs = DynamicPPL.OnlyAccsVarInfo(
        DynamicPPL.VectorParamAccumulator(new_ldf), extra_accs...
    )
    _, accs = DynamicPPL.init!!(
        new_ldf.model, accs, init_strategy, new_ldf.transform_strategy
    )
    new_params = DynamicPPL.get_vector_params(accs)
    if old_ldf.adtype !== nothing
        # `new_ldf`'s own layout, not `layout`: they differ when the block was reshaped.
        new_ldf = DynamicPPL.LogDensityFunction(
            model,
            DynamicPPL.get_logdensity_callable(old_ldf),
            DynamicPPL.get_all_ranges_and_transforms(new_ldf),
            new_params;
            adtype=old_ldf.adtype,
        )
    end
    return new_ldf, new_params, accs
end

#
# Gibbs implementation itself
#

"""
    Snapshot(values, supports)

The values Gibbs threads through a sweep, with the support signature of each tilde statement
that produced them.

Both halves come from one model evaluation and [`reached_values`](@ref) is the only thing that
builds one, which is what lets [`check_variable_set`](@ref) compare the supports either side of
a step without evaluating the model again: it needs the supports from before the step as well
as after, and the evaluation that produced the previous snapshot had already computed them.
"""
struct Snapshot{V<:DynamicPPL.VarNamedTuple,S<:DynamicPPL.VarNamedTuple}
    values::V
    supports::S
end

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
    _, accs = DynamicPPL.init!!(
        rng,
        model,
        _snapshot_accs(),
        DynamicPPL.InitFromParams(proposed, DynamicPPL.InitFromPrior()),
        DynamicPPL.UnlinkAll(),
    )
    return _snapshot(accs)
end

# Raw values, plus the support signature of each tilde statement that produced them.
function _snapshot_accs()
    return DynamicPPL.OnlyAccsVarInfo(
        DynamicPPL.RawValueAccumulator(false), SupportSignatureAccumulator()
    )
end

function _snapshot(accs)
    return Snapshot(
        DynamicPPL.get_raw_values(accs),
        DynamicPPL.getacc(accs, Val(SUPPORT_ACC_NAME)).values,
    )
end

"""
    _support_signature(dist)

A value that differs whenever a distribution's support or its form does.

The snapshot Gibbs threads holds raw values, which say nothing about the set a variable is
drawn from, so this has to come from the tilde's own distribution during the evaluation. Two
parts, because the rule asks for both to be unchanged:

  - the type name, for the distributional form. `Normal(0, 1)` and `TDist(3)` share the support
    of the whole line and are still a change.
  - `Bijectors.bijector`, for the support of a continuous variable. It is the
    constrained-to-unconstrained map, so it carries the support and nothing else: every
    `Normal(m, sigma)` gives the same `identity`, which an ordinary hierarchical
    `x ~ Normal(m, 1)` depends on, while `truncated(Normal(); lower=x)` gives a
    `TruncatedBijector` carrying `x` and so registers when `x` moves.
  - the bounds, for the support of a discrete one. See [`_support_bounds`](@ref).

The bijector is what makes the continuous case uniform across shapes, which bounds alone could
not be: bounds exist only for univariate distributions, so a signature built from them silently
ignored the support of everything else -- a
`product_distribution([truncated(Normal(); lower=a), ...])` whose bound came from another block
passed unnoticed while its univariate sibling was refused. The bounds supplement it rather than
replacing it.

Two kinds of support change get past all three parts, and both are reducible. One is a support
carried entirely by a distribution's parameters: `Categorical([0.5, 0.5, 0, 0])` and
`Categorical([0, 0, 0.5, 0.5])` are both supported on `1:4` as far as `Distributions` is
concerned. The other is a discrete distribution that reports no bounds, `Multinomial(n, p)`
among them, where the bijector is `identity` and [`_support_bounds`](@ref) has nothing to
return. Both pass, and a component moving another block's variable that way is accepted. This
signature is what the rule can enforce, not the whole of what the rule says.
"""
function _support_signature(d::Distributions.Distribution)
    # `bijector` is not part of what a model has to provide. A distribution implementing only
    # the vector linking interface, `Bijectors.VectorBijectors.to_linked_vec`, samples correctly
    # under every sampler, while Bijectors' generic univariate `bijector` reaches for
    # `minimum`/`maximum`, which such a distribution need not define. Asking for it anyway made
    # Gibbs the one sampler that could not take the model. Deriving the transform instead is no
    # good either: it is the expensive step `fix_transforms` exists to avoid, and doing it once
    # per tilde per sweep would defeat that.
    try
        return (nameof(typeof(d)), Bijectors.bijector(d), _support_bounds(d))
    catch e
        e isa MethodError || rethrow()
        # The form alone still catches a change of distribution, and misses only a same-type
        # change of support, which needs the bounds to express in the first place.
        return (nameof(typeof(d)),)
    end
end
_support_signature(d) = (nameof(typeof(d)),)

"""
    _support_bounds(dist)

The bounds of `dist`'s support, or `()` when it does not report them.

This is what carries the support of a *discrete* variable. A discrete variable is never linked,
so `Bijectors.bijector` is `identity` for the whole family and says nothing about which values it
can take: `DiscreteUniform(1, 2)` and `DiscreteUniform(3, 4)` are disjoint and would otherwise
share a signature. With `b` in one block choosing between them and `k` in another, the chain is
absorbed into whichever branch it started in -- P(b=1) came back 0.0 or 1.0 by seed against an
exact 0.5.

Multivariate distributions are asked too, and must be: restricting this to the univariate case
left every discrete *vector* with the signature `(family, identity, ())`, which is the same hole
one shape up. A `product_distribution` of `DiscreteUniform(1, 2)` and one of
`DiscreteUniform(3, 4)` are told apart only here.

Not every distribution reports bounds, and one that does not is left uncovered rather than
refused: `Multinomial(n, p)` has no `minimum`, so a component moving another block's variable
between `Multinomial(2, p)` and `Multinomial(4, p)` still passes unseen. Continuous
distributions lose nothing by it, since the bijector carries their support; discrete ones with no
bounds are the residue of this rule, alongside a support carried purely by parameters. Both are
recorded in the note in the `Gibbs` docstring.
"""
function _support_bounds(d::Distributions.Distribution)
    # Guarded here rather than by `_support_signature`'s own `try`, so that a distribution
    # without bounds keeps the rest of its signature instead of falling back to the family.
    return try
        (minimum(d), maximum(d))
    catch e
        e isa MethodError || rethrow()
        ()
    end
end

const SUPPORT_ACC_NAME = :GibbsSupportSignatures
_store_support(val, tval, logjac, vn, dist) = _support_signature(dist)
function SupportSignatureAccumulator()
    return DynamicPPL.VNTAccumulator{SUPPORT_ACC_NAME}(_store_support)
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
components may sample the same variable. What they may not do is split a value that the model
stores as a unit: if one component declares `x` and another `x[1]`, whether Gibbs can express
that depends on the model's tilde statements. Written element by element -- `x[1] ~ Normal();
x[2] ~ Normal()` -- each element is a key of its own and the partition works. Written as one
draw, `x ~ MvNormal(...)`, `x` is a single key: Gibbs cannot free part of it and throws. Which
sampler holds the containing block makes no difference, because the values Gibbs threads come
from evaluating the model rather than from what a component hands back.

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
  - `Gibbs(@varname(y) => MH(), (@varname(x), @varname(z)) => MH(@varname(z) => Normal()))` --
    the block is right, but `z` is given a proposal of its own, so its proposal density no
    longer cancels against its prior and the acceptance ratio between the two supports is not
    the one the algorithm assumes. The same partition with a plain `MH()` samples correctly,
    because proposing from the prior is what makes the cancellation hold.

The same applies to a new element rather than a new name: an `x[5]` first reached inside a
branch is treated exactly like `z` above.

When each is caught varies, and cannot be made uniform. A component that cannot serve in Gibbs
at all is refused when the `Gibbs` sampler is constructed, and anything already visible in the
first draw is refused at the first step. The rest can only surface while sampling, because
whether `z` exists depends on `x`, which is drawn during the run: the same partition may be
refused at the first sweep or fifty sweeps in, depending on the draws. Sampling stops with an
error at that point rather than carrying on, since the chain from an invalid partition is
biased rather than merely noisy.

A component's step can reach into another block in two ways, because a variable in one block
may decide something about a variable in another without ever sampling it. Both are refused by
one rule:

> A component may never change the dimension or the distributional form of a variable
> belonging to another component, unless the two share that variable.

Sharing is the remedy in either direction: put the deciding variable and what it decides in one
block, or write a component that samples both.

| What another component's step changes | Effect on this block             | Treatment  |
|---------------------------------------|----------------------------------|------------|
| whether a tilde statement executes    | variable appears or leaves       | refused    |
| which distribution a tilde draws from | its family or its support moves  | refused    |
| a distribution's parameters only      | neither; the form is unchanged   | allowed    |

"Distributional form" is the family together with the support, and never the parameters. Both
halves of it are compared, so `x ~ Normal(0, 1)` becoming `x ~ TDist(3)` is refused even though
the support is the whole line either way.

Excluding the parameters is what makes the third row work, and it is the common case: under
`m ~ Normal(0, 1)` in one block and `x ~ Normal(m, 1)` in another, `x`'s distribution differs
every sweep while its family and its support do not, and refusing that would refuse every
hierarchical model.

The second row is the one worth knowing about, because it was previously silent. With `b` in
one block and `x ~ Dirichlet(3)` under `b == 1` but `x ~ MvNormal(zeros(3), I)` otherwise, `x`'s
leaf set and its three raw values are the same either way. Each component's conditional is
individually correct, but a draw from the second branch almost never lies on the simplex, so
once the chain leaves `b == 1` it cannot return: it is reducible rather than biased, and used to
return P(b=1) near 0 or near 1 against an exact 0.6456.

!!! note
    That the decider and what it decides must share a block is an assumption of *this* Gibbs
    implementation, not a property of Gibbs sampling. A component's target is its full
    conditional whichever block the decider sits in; what breaks is this implementation's
    bookkeeping -- a snapshot of raw values cannot see a support change, and a component
    holding a flat parameter layout cannot be re-pointed at a different one.

    The rule is also deliberately conservative. Whether a particular support change is fatal
    is a question about irreducibility, which no single model evaluation can decide, so
    partitions that would have sampled correctly are refused with the rest: with `a` in one
    block and `x ~ truncated(Normal(); lower=a)` in another, the region stays connected and
    the split agreed with a single-block reference to 0.0015 before this rule refused it.
    Turing.jl#2801 is the same shape and had a regression test asserting it against a NUTS
    ground truth. Put such variables in one block -- which recovers that answer -- or write a
    sampler for them. A Gibbs built to track supports, or to reason about reachability, could
    relax this.

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

struct GibbsState{Sn<:Snapshot,S,B}
    snapshot::Sn
    states::S
    # The leaves each component's block held when that component last stepped, so that a block
    # reshaped by another component's step can be spotted without asking the component. Taken
    # right after each step rather than at the end of the sweep: a later component can reshape
    # an earlier one's block, so there is no single snapshot that answers for all of them.
    blocks::B
end

# The leaves of `vnt` that `varnames` claims, i.e. the shape of that component's block.
function block_leaves(vnt::DynamicPPL.VarNamedTuple, varnames)
    owned(leaf) = any(hv -> AbstractPPL.subsumes(hv, leaf), varnames)
    return Set(
        leaf for vn in keys(vnt) for
        leaf in AbstractPPL.varname_leaves(vn, vnt[vn]) if owned(leaf)
    )
end

"""
    check_block_nonempty(varnames, block)

Throw if the model currently reaches none of the variables a component samples.

A component with nothing to sample is refused rather than skipped. `MH` would in fact sample
such a partition correctly, and skipping the component for that sweep is the sensible
semantics, but a `LogDensityFunction` over no variables is ill-formed and fails deep inside
with `VectorEvaluator requires a vector of floating-point values`. Refusing uniformly is loud
and easy to revisit; letting it through for some samplers and not others is neither.
"""
function check_block_nonempty(varnames, block)
    isempty(block) || return nothing
    return throw(
        ArgumentError(
            "The component sampling $(join(varnames, ", ")) has nothing to sample: the " *
            "model currently reaches none of those variables. That means another " *
            "component's step decides whether they exist at all, which Gibbs does not " *
            "support -- put them and the variables that decide their existence in one " *
            "block, sampled by a component that can handle a varying set, such as `PG`.",
        ),
    )
end

"""
    block_fingerprint(snapshot, varnames)

The shape of a component's block: the leaves it holds, and the linking transform of each tilde
statement that produced them.

Leaves alone are not enough. A tilde statement that changes which distribution it draws from can
keep its leaf count and still move the block's *linked* dimension -- `x ~ Dirichlet(3)` occupies
two numbers and `x ~ MvNormal(zeros(3), I)` three, both with three leaves -- and a component
holding a parameter layout has to be told.

What is compared is the transform's *type*, not the transform. A change of type is what can move
the linked dimension; a change within a type cannot. That distinction is the whole of the
difference between this and [`check_variable_set`](@ref)'s support test, which compares the
transform itself: with `a` and `x` in one block and `x ~ truncated(Normal(); lower=a)` in
another that shares `x`, the bound moves every sweep and the support test is right to see it,
while the layout never changes and a gate that fired here would refuse a partition that samples
perfectly well.
"""
function block_fingerprint(snapshot::Snapshot, varnames)
    # Pairs rather than a `VarNamedTuple`: this is an internal fingerprint, compared and never
    # indexed into, and a tilde `VarName` carrying a `Colon` can only be written into a
    # `VarNamedTuple` with a template giving the array's shape, which there is nothing here to
    # supply. The order is the evaluation's, so it is stable between sweeps.
    layout = Pair{VarName,Any}[]
    for vn in keys(snapshot.supports)
        any(hv -> AbstractPPL.subsumes(hv, vn), varnames) || continue
        push!(layout, vn => _layout_key(snapshot.supports[vn]))
    end
    return (leaves=block_leaves(snapshot.values, varnames), layout=layout)
end

# The part of a support signature that can move the linked dimension: the distributional form,
# and the transform's type rather than the transform. See `block_fingerprint`.
_layout_key(sig::Tuple{Any}) = sig
_layout_key(sig::Tuple) = (sig[1], typeof(sig[2]))

# `nothing` if the block has the shape the component last saw, otherwise the variable that
# changed it. Which one is reported only shapes the error message, so any of them will do.
function block_reshaped(before, now)
    if !isequal(before.leaves, now.leaves)
        joined = setdiff(now.leaves, before.leaves)
        isempty(joined) || return ReshapedBlock(first(joined), :joined)
        return ReshapedBlock(first(setdiff(before.leaves, now.leaves)), :left)
    end
    # The tilde keys matter as much as what they map to. A branch writing `x` in one tilde and
    # `x[i]` in three gives identical leaves and two DISJOINT layouts, so a loop that skipped a
    # key the other side lacked saw nothing while the linked dimension moved from two to three.
    # A key on one side only is itself proof the layout changed.
    isequal(Set(before.layout), Set(now.layout)) && return nothing
    for (vn, key) in now.layout
        i = findfirst(p -> p.first == vn, before.layout)
        i === nothing && return ReshapedBlock(vn, :respecified)
        isequal(before.layout[i].second, key) || return ReshapedBlock(vn, :respecified)
    end
    for (vn, _) in before.layout
        any(p -> p.first == vn, now.layout) || return ReshapedBlock(vn, :respecified)
    end
    return nothing
end

"""
    GibbsInitStrategy(varnames, strategies)

Initialise each variable with the strategy of the component sampler that samples it, choosing
by ownership; a variable no component claims falls back to the prior.

This produces the single initial draw the first sweep starts from, and nothing else. Each
component's own initialisation does not go through it: `gibbs_initialstep_recursive` asks
`init_strategy(sampler)` directly, because ownership is ambiguous where components overlap. So
an `HMC` component gets its `InitFromUniform` starting point from there, not from here.
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
    check_reported_variables(varnames, conditioned, report)

Throw if `report` contains a value for a variable the component did not sample and was
conditioned on.

`gibbs_get_parameter_values`' contract is that a component reports only what it samples, and
`merge` gives the report priority, so a foreign value silently replaces another block's --
either frozen for the rest of the run, or re-sampled afterwards from a corrupted conditional,
which looks perfectly plausible in the chain.

A component may legitimately report a variable it does not sample when that variable did not
exist to be conditioned on and its own step brought it into being: it is assumed rather than
observed, so it lands in the component's accumulator. That case is not an error here --
[`check_variable_set`](@ref)'s appearance branch diagnoses it, and names the component that
should have sampled it. Only a variable that *was* conditioned can have been overwritten.
"""
function check_reported_variables(
    varnames, conditioned::DynamicPPL.VarNamedTuple, report::DynamicPPL.VarNamedTuple
)
    for vn in keys(report), leaf in AbstractPPL.varname_leaves(vn, report[vn])
        any(hv -> AbstractPPL.subsumes(hv, leaf), varnames) && continue
        DynamicPPL.hasvalue(conditioned, leaf) || continue
        throw(
            ArgumentError(
                "The component sampling $(join(varnames, ", ")) reported a value for " *
                "$(leaf), which it does not sample and which was conditioned on for its " *
                "step. Reporting it overwrites the value the component that does sample it " *
                "produced. `gibbs_get_parameter_values` must return only the variables its " *
                "own sampler is responsible for.",
            ),
        )
    end
    return nothing
end

"""
    check_variable_set(spl, sampler, varnames, old, new, proposed)

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

Both directions are checked, and symmetrically: `old` and `new` are both the set of variables a
model evaluation reached (see [`reached_values`](@ref)), so a leaf in one and not the other came
or went during this step, and nothing else can have caused it -- every variable outside the
component's block was conditioned.

A variable can also change while coming and going neither way, if another component's step
decided which distribution its tilde statement draws from. That is checked from the support
signatures the two snapshots carry, and against ownership alone: see the comment at the loop.

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
    old::Snapshot,
    new::Snapshot,
    proposed::DynamicPPL.VarNamedTuple,
)
    old_vnt, new_vnt = old.values, new.values
    # Walk to the leaves rather than compare keys: an array stored under a single key can grow
    # or lose an element, and an `x[5]` inside a branch matters as much as a whole `z`.
    for (from, to, what) in
        ((new_vnt, old_vnt, "appeared"), (old_vnt, new_vnt, "stopped existing"))
        for vn in keys(from), leaf in AbstractPPL.varname_leaves(vn, from[vn])
            DynamicPPL.hasvalue(to, leaf) && continue
            _require_owned(
                spl,
                sampler,
                varnames,
                leaf,
                what,
                "the other component conditions on a variable the state being proposed does " *
                "not have, and the chain comes back biased",
            )
            _require_varying_dimension(sampler, leaf, what)
        end
    end
    # A variable can also change without coming or going, if another component's step decided
    # which distribution its tilde statement draws from. The rule is the same as above and the
    # test is the same: a component may move the dimension or the distributional form of a
    # variable only if it samples that variable. The form is the family together with the
    # support and never the parameters, so a hierarchical `x ~ Normal(m, 1)` is untouched.
    # Compared per tilde statement, which is the granularity a distribution belongs to.
    for vn in keys(old.supports)
        DynamicPPL.hasvalue(new.supports, vn) || continue
        isequal(old.supports[vn], DynamicPPL.getvalue(new.supports, vn)) && continue
        # Ownership is the whole rule here. `allow_varying_dimension` is a different
        # question -- whether a proposal may move between two sets of variables -- and the set
        # has not changed, so asking it would refuse a component that samples both the decider
        # and what it decides, which is exactly the arrangement this check asks for.
        _require_owned(
            spl,
            sampler,
            varnames,
            vn,
            "changed distributional form or dimension",
            "the two components disagree about the set it is drawn from, and the chain can " *
            "be left unable to return to part of the space",
        )
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
function _require_owned(
    spl::Gibbs, sampler, varnames, leaf, what::String, consequence::String
)
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
                "decides this has to be the one that samples $(leaf), or $(consequence). " *
                "Put $(leaf) and whatever decides this in one block."
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
    _, accs = DynamicPPL.init!!(
        rng, model, _snapshot_accs(), initial_params, DynamicPPL.UnlinkAll()
    )
    snapshot = _snapshot(accs)
    check_all_variables_handled(snapshot.values, spl)
    return snapshot
end

# `step` and `step_warmup` differ only in which of the two the components are stepped with, so
# both pairs of methods forward to one body with `step_function` bound accordingly.
function _gibbs_initialstep(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::Gibbs,
    step_function::Function;
    initial_params=Turing.Inference.init_strategy(spl),
    discard_sample=false,
    kwargs...,
)
    snapshot = gibbs_initial_values(rng, model, spl, initial_params)
    snapshot, states, blocks = gibbs_initialstep_recursive(
        rng,
        model,
        step_function,
        spl,
        spl.varnames,
        spl.samplers,
        snapshot;
        initial_params=initial_params,
        kwargs...,
    )
    return _gibbs_transition(model, spl, snapshot, states, discard_sample),
    GibbsState(snapshot, states, blocks)
end

function _gibbs_transition(model, spl::Gibbs, snapshot::Snapshot, states, discard_sample)
    discard_sample && return nothing
    return DynamicPPL.ParamsWithStats(
        DynamicPPL.InitFromParams(snapshot.values), model, component_stats(spl, states)
    )
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG, model::DynamicPPL.Model, spl::Gibbs; kwargs...
)
    return _gibbs_initialstep(rng, model, spl, AbstractMCMC.step; kwargs...)
end

function AbstractMCMC.step_warmup(
    rng::Random.AbstractRNG, model::DynamicPPL.Model, spl::Gibbs; kwargs...
)
    return _gibbs_initialstep(rng, model, spl, AbstractMCMC.step_warmup; kwargs...)
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
    snapshot::Snapshot,
    states=(),
    blocks=();
    initial_params,
    kwargs...,
)
    # End recursion
    if isempty(varname_vecs) && isempty(samplers)
        return snapshot, states, blocks
    end

    varnames, varname_vecs_tail... = varname_vecs
    sampler, samplers_tail... = samplers

    # Construct the conditioned model.
    check_block_nonempty(varnames, block_leaves(snapshot.values, varnames))
    conditioned = conditioned_values(snapshot.values, varnames)
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
    report = gibbs_get_parameter_values(new_state)
    check_reported_variables(varnames, conditioned, report)
    proposed = merge(conditioned, report)
    new_snapshot = reached_values(rng, model, proposed)
    check_variable_set(spl, sampler, varnames, snapshot, new_snapshot, proposed)

    states = (states..., new_state)
    blocks = (blocks..., block_fingerprint(new_snapshot, varnames))
    return gibbs_initialstep_recursive(
        rng,
        model,
        step_function,
        spl,
        varname_vecs_tail,
        samplers_tail,
        new_snapshot,
        states,
        blocks;
        initial_params=initial_params,
        kwargs...,
    )
end

function _gibbs_step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::Gibbs,
    state::GibbsState,
    step_function::Function;
    discard_sample=false,
    kwargs...,
)
    @assert length(spl.samplers) == length(state.states)
    snapshot, states, blocks = gibbs_step_recursive(
        rng,
        model,
        step_function,
        spl,
        spl.varnames,
        spl.samplers,
        state.states,
        state.blocks,
        state.snapshot;
        kwargs...,
    )
    return _gibbs_transition(model, spl, snapshot, states, discard_sample),
    GibbsState(snapshot, states, blocks)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::Gibbs,
    state::GibbsState;
    kwargs...,
)
    return _gibbs_step(rng, model, spl, state, AbstractMCMC.step; kwargs...)
end

function AbstractMCMC.step_warmup(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::Gibbs,
    state::GibbsState;
    kwargs...,
)
    return _gibbs_step(rng, model, spl, state, AbstractMCMC.step_warmup; kwargs...)
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
    blocks,
    snapshot::Snapshot,
    new_states=(),
    new_blocks=();
    kwargs...,
)
    # End recursion.
    if isempty(varname_vecs) && isempty(samplers) && isempty(states)
        return snapshot, new_states, new_blocks
    end

    varnames, varname_vecs_tail... = varname_vecs
    sampler, samplers_tail... = samplers
    state, states_tail... = states
    block, blocks_tail... = blocks
    # Construct the conditional model that this sampler should use.
    conditioned = conditioned_values(snapshot.values, varnames)
    conditioned_model = DynamicPPL.condition(model, conditioned)
    # Update the sampler's state based on global values that were provided by other
    # samplers. A block another component's step has reshaped since this one last stepped
    # goes to the `ReshapedBlock` method, which throws unless the sampler implements it.
    current_block = block_fingerprint(snapshot, varnames)
    check_block_nonempty(varnames, current_block.leaves)
    reshaped = block_reshaped(block, current_block)
    state = if reshaped === nothing
        gibbs_update_state!!(sampler, state, conditioned_model, snapshot.values)
    else
        gibbs_update_state!!(sampler, state, conditioned_model, snapshot.values, reshaped)
    end

    # Take a step with the local sampler. We don't need the actual sample, only the state.
    # Note that we pass `discard_sample=true` after `kwargs...`, because AbstractMCMC will
    # tell Gibbs that _this Gibbs sample_ should be kept, and so `kwargs` will actually
    # contain `discard_sample=false`!
    _, new_state = step_function(
        rng, conditioned_model, sampler, state; kwargs..., discard_sample=true
    )

    report = gibbs_get_parameter_values(new_state)
    check_reported_variables(varnames, conditioned, report)
    proposed = merge(conditioned, report)
    new_snapshot = reached_values(rng, model, proposed)
    check_variable_set(spl, sampler, varnames, snapshot, new_snapshot, proposed)

    new_states = (new_states..., new_state)
    new_blocks = (new_blocks..., block_fingerprint(new_snapshot, varnames))
    return gibbs_step_recursive(
        rng,
        model,
        step_function,
        spl,
        varname_vecs_tail,
        samplers_tail,
        states_tail,
        blocks_tail,
        new_snapshot,
        new_states,
        new_blocks;
        kwargs...,
    )
end
