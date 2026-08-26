#
# Particle filtering and particle MCMC samplers: SMC, PG / conditional SMC
#

# A probabilistic model becomes a particle filter by reading each `observe` statement as one
# filtering step. Evaluated under `SMCContext`, every likelihood term calls `Libtask.produce`, so
# a *particle* is a suspended model execution: we `advance!` it to its next `observe`, take the
# produced log-likelihood as its weight, then resample. SMC is one such sweep; particle Gibbs
# (PG/CSMC) runs a *conditional* sweep -- one particle is a fixed reference trajectory -- inside
# an MCMC loop.
#
# The reference reproduces the retained trajectory by reusing its values rather than replaying its
# random draws (see the `reference` field for why); a fork forgets them in `reseed!`.
#
# Reference: Andrieu, Doucet & Holenstein, "Particle Markov chain Monte Carlo methods", Journal of
# the Royal Statistical Society: Series B 72(3), 269-342 (2010).

using StatsFuns: softmax, logsumexp
import Random123

#
# Particle random number generation
#

# Each particle owns a counter-based `Random123.Philox2x`. This section comes first because
# `Particle` names the generator type in its signature.

"A fresh counter-based generator for one particle, seeded from `rng`."
function particle_rng(rng::AbstractRNG=Random.default_rng())
    return Random.seed!(Random123.Philox2x(), rand(rng, Random.Sampler(rng, UInt64)))
end

# Derive a fresh seed from `key`. Re-seeding to split one generator into many can yield correlated
# streams (Steele et al., "Fast Splittable Pseudorandom Number Generators", OOPSLA 2014), and a
# `MersenneTwister` derivation is not stable across Julia versions, which drifted SMC/PG results
# even under a StableRNG (#2781, AdvancedPS.jl#110). Philox is counter-based with a fixed
# algorithm, so a seed derived through it is both decorrelated and version-stable.
split_key(key::Integer) = rand(Random.seed!(Random123.Philox2x(), key), typeof(key))

"Reseed from the generator's own current state (used between steps when not resampling)."
refresh!(rng::Random123.Philox2x) = Random.seed!(rng, split_key(rng.key))

#
# Model evaluation via Libtask
#

# A `Particle` is the only mutable state. It is stored as its `TapedTask`'s "taped globals",
# so the tilde overloads reach it from *inside* a running model via `get_taped_globals`.
# This keeps all state explicit on the particle -- no `task_local_storage`.

# Particle samplers replay executions in a fixed order, so they cannot run models whose
# evaluation order is nondeterministic (e.g. a threaded `observe` loop).
function error_if_threadsafe_eval(model::DynamicPPL.Model)
    if DynamicPPL.requires_threadsafe(model)
        throw(
            ArgumentError(
                "Particle sampling methods do not currently support models that need threadsafe evaluation.",
            ),
        )
    end
    return nothing
end

"""
    SMCContext

Leaf context marking a model evaluation as a particle-filter step: `tilde_assume!!` draws from
the prior using the particle's own generator -- or, for a conditional-SMC reference, reuses the
retained trajectory's value at that address -- and `tilde_observe!!` scores the observation and
`Libtask.produce`s the increment as the particle's weight.
"""
struct SMCContext <: DynamicPPL.AbstractContext end

# `OnlyAccsVarInfo` needs a parameter eltype; `Any` is fine here since particle MCMC never
# involves AD or tracer types (see the `get_param_eltype` docstring in DynamicPPL).
DynamicPPL.get_param_eltype(::DynamicPPL.AbstractVarInfo, ::SMCContext) = Any

"""
    Particle(model, rng)
    Particle(model, rng, reference::DynamicPPL.VarNamedTuple)

A single particle: a suspended `model` execution together with its `varinfo`, its own `rng`, and an
accumulated `logweight`.

Without `reference` the particle draws from the prior. Given a retained trajectory's raw values it
becomes a conditional-SMC reference pinned to that trajectory, erroring if its execution reaches an
address that trajectory lacks or finishes without reaching one it has.
"""
mutable struct Particle{RT<:AbstractRNG,WT<:Real}
    # Abstract on purpose: the varinfo type changes as the execution proceeds, since each new
    # address widens the raw-value accumulator's `VarNamedTuple`. Accesses go through Libtask's
    # (already type-unstable) taped globals, so this costs nothing extra.
    varinfo::DynamicPPL.AbstractVarInfo
    rng::RT
    # `logweight` tracks whatever `DynamicPPL.LogProbType` is, so weights follow suit if it
    # is ever changed.
    logweight::WT
    # `nothing` unless this particle is a CSMC reference; otherwise the retained trajectory's raw
    # values, which it reuses (`InitFromParams` in `tilde_assume!!`). Reusing the *value* rather than
    # replaying the RNG draw is what survives Gibbs re-conditioning: a draw is x = g(u; θ), so
    # replaying u after θ → θ' gives g(u; θ') ≠ x -- μ + Φ⁻¹(u) shifts by μ' − μ.
    reference::Union{Nothing,DynamicPPL.VarNamedTuple}
    task::Libtask.TapedTask
    # `task` is filled in once the particle exists, because the task must capture the
    # particle as its taped globals (a back-reference). This has to be an inner constructor
    # for that reason: `task` is left undefined here and set immediately after.
    function Particle(
        vi::DynamicPPL.AbstractVarInfo,
        rng::RT,
        reference::Union{Nothing,DynamicPPL.VarNamedTuple}=nothing,
    ) where {RT<:AbstractRNG}
        w = zero(DynamicPPL.LogProbType)
        return new{RT,typeof(w)}(vi, rng, w, reference)
    end
end

function Particle(
    model::DynamicPPL.Model,
    rng::AbstractRNG,
    reference::Union{Nothing,DynamicPPL.VarNamedTuple}=nothing,
)
    model = DynamicPPL.setleafcontext(model, SMCContext())
    varinfo = particle_varinfo()
    args, kwargs = DynamicPPL.make_evaluate_args_and_kwargs(model, varinfo)
    particle = Particle(varinfo, rng, reference)
    particle.task = Libtask.TapedTask(particle, model.f, args...; kwargs...)
    return particle
end

"""
    reseed!(particle, rng)

Restart `particle` as a fresh continuation seeded from `rng`, so that a particle descended from
the reference stops reusing retained values and samples afresh. Mutates and returns `particle`.
"""
function reseed!(particle::Particle, rng::AbstractRNG)
    Random.seed!(particle.rng, rand(rng, UInt64))
    # A fork samples fresh from here on, so it must forget the reference's remaining values.
    particle.reference = nothing
    return particle
end

"""
    fork(particle, rng)

Copy `particle` into an independent, reseeded continuation. `deepcopy` forks the underlying
`TapedTask` (Libtask defines `copy` as `deepcopy`) and preserves the task↔particle
back-reference; [`reseed!`](@ref) then gives it its own random stream.
"""
fork(particle::Particle, rng::AbstractRNG) = reseed!(deepcopy(particle), rng)

"""
Whether `particle` is a conditional-SMC reference, i.e. pinned to a retained trajectory. Carried
on the particle rather than inferred from its slot, so forking and resampling cannot get it wrong.
"""
isreference(particle::Particle) = particle.reference !== nothing

"""
    advance!(particle) -> Union{Real,Nothing}

Run the particle to its next `observe`, returning the incremental log-likelihood, or
`nothing` once the model finishes.
"""
function advance!(particle::Particle)
    score = Libtask.consume(particle.task)
    # `tilde_assume!!` already rejects an address the retained values lack, so the only
    # discrepancy left to catch here is a retained address the reference never visited (e.g. a
    # branch that stopped being taken after re-conditioning).
    reference = particle.reference
    if score === nothing && reference !== nothing
        dropped = setdiff(
            keys(reference), keys(DynamicPPL.get_raw_values(particle.varinfo))
        )
        isempty(dropped) || error(
            "the reference execution trace changed while replaying retained values " *
            "(retained addresses never reached: $(collect(dropped)))",
        )
    end
    return score
end

function DynamicPPL.tilde_assume!!(
    ::SMCContext, dist::Distribution, vn::VarName, template, vi::DynamicPPL.AbstractVarInfo
)
    particle = Libtask.get_taped_globals(Particle)
    # A reference reuses the retained value at every address it visits (see the `reference` field);
    # ordinary particles and forks have no expected set, so they draw from the prior. The two error
    # paths cover the two ways the trace can have moved: an address outside the retained set here,
    # and -- via the `nothing` fallback -- a retained address with no usable value.
    reference = particle.reference
    strategy = if reference === nothing
        DynamicPPL.InitFromPrior()
    else
        haskey(reference, vn) || error(
            "the reference execution trace changed while replaying retained values " *
            "(new address: $vn)",
        )
        DynamicPPL.InitFromParams(reference, nothing)
    end
    ctx = DynamicPPL.InitContext(particle.rng, strategy, DynamicPPL.UnlinkAll())
    x, vi = DynamicPPL.tilde_assume!!(ctx, dist, vn, template, vi)
    particle.varinfo = vi
    return x, vi
end

# Both tilde handlers thread the varinfo the model passed in, and mirror the result onto the
# particle for the sweep to read between produces. Reading `particle.varinfo` instead would discard
# whatever the model body accumulated since the previous tilde: `@addlogprob!` reaches the varinfo
# directly rather than through a handler, so its term would never reach a chain's `loglikelihood`.
#
# The weight is not emitted here -- `ProduceLogLikelihoodAccumulator` produces it as it accumulates,
# below.
function DynamicPPL.tilde_observe!!(
    ::SMCContext,
    dist::Distribution,
    left,
    vn::Union{VarName,Nothing},
    template,
    vi::DynamicPPL.AbstractVarInfo,
)
    particle = Libtask.get_taped_globals(Particle)
    left, vi = DynamicPPL.tilde_observe!!(
        DynamicPPL.DefaultContext(), dist, left, vn, template, vi
    )
    particle.varinfo = vi
    return left, vi
end

"""
    ProduceLogLikelihoodAccumulator{T} <: LogProbAccumulator{T}

A likelihood accumulator that `Libtask.produce`s each increment as it accumulates it, which is what
turns a model evaluation into a particle filter: one produce per likelihood term, so the sweep sees
one filtering step per `observe`. Substituting it for `LogLikelihoodAccumulator` is the only thing
that distinguishes a particle's varinfo.
"""
struct ProduceLogLikelihoodAccumulator{T<:Real} <: DynamicPPL.LogProbAccumulator{T}
    logp::T
end

DynamicPPL.accumulator_name(::Type{<:ProduceLogLikelihoodAccumulator}) = :LogLikelihood
DynamicPPL.logp(acc::ProduceLogLikelihoodAccumulator) = acc.logp

# The produce lives in the one place the likelihood is accumulated, so `val` *is* the increment:
# "produced weight equals accumulated increment" is structural rather than an invariant two call
# sites must keep in step. `observe` and `@addlogprob!` both arrive through exactly one `acclogp`,
# which is what gets `@addlogprob!` into the weight (issue #1996). `combine` adds the two `logp`s
# directly rather than through `acclogp`, so folding a submodel's varinfo into its parent's stays
# silent. `produce` suspends before the caller stores the updated varinfo, so a *suspended*
# particle's total lags this term; every varinfo read happens after the model has run to completion.
function DynamicPPL.acclogp(acc::ProduceLogLikelihoodAccumulator, val)
    Libtask.produce(val)
    return ProduceLogLikelihoodAccumulator(DynamicPPL.logp(acc) + val)
end

function DynamicPPL.accumulate_assume!!(
    acc::ProduceLogLikelihoodAccumulator, val, tval, logjac, vn, dist, template
)
    return acc
end
function DynamicPPL.accumulate_observe!!(
    acc::ProduceLogLikelihoodAccumulator, dist, left, vn, template
)
    return DynamicPPL.acclogp(acc, Distributions.loglikelihood(dist, left))
end

# `@addlogprob!` and `:=` update the varinfo without passing through a tilde handler, so a trailing
# one has nothing after it to mirror the result onto the particle, and the final varinfo is
# unreachable: `Libtask.consume` discards the evaluator's return value once the model is done. The
# methods below therefore mirror as the update happens, keyed on the produce-aware accumulator,
# which only a particle carries (`trajectory_varinfo` is the set without it) and so identifies a
# particle varinfo.
#
# `:=` is dispatched on the context, so its method below takes `SMCContext`. `@addlogprob!` has no
# such hook, so the two accumulator methods extend DynamicPPL's own and reproduce their bodies.
# They dispatch on `OnlyAccsVarInfo`, which every sampler uses, so they must stay behaviourally
# identical to DynamicPPL's; they belong there once it offers a context-dispatched entry point.
function mirror_onto_particle(vi::DynamicPPL.OnlyAccsVarInfo)
    acc_name = Val(:LogLikelihood)
    if DynamicPPL.hasacc(vi, acc_name) &&
        DynamicPPL.getacc(vi, acc_name) isa ProduceLogLikelihoodAccumulator
        Libtask.get_taped_globals(Particle).varinfo = vi
    end
    return vi
end

function acclogp_and_mirror!!(vi, acc_name, logp, ignore_missing_accumulator)
    if ignore_missing_accumulator && !DynamicPPL.hasacc(vi, acc_name)
        return vi
    end
    return mirror_onto_particle(
        DynamicPPL.map_accumulator!!(acc -> DynamicPPL.acclogp(acc, logp), vi, acc_name)
    )
end

function DynamicPPL.accloglikelihood!!(
    vi::DynamicPPL.OnlyAccsVarInfo, logp; ignore_missing_accumulator=false
)
    return acclogp_and_mirror!!(vi, Val(:LogLikelihood), logp, ignore_missing_accumulator)
end

function DynamicPPL.acclogprior!!(
    vi::DynamicPPL.OnlyAccsVarInfo, logp; ignore_missing_accumulator=false
)
    return acclogp_and_mirror!!(vi, Val(:LogPrior), logp, ignore_missing_accumulator)
end

function DynamicPPL.store_coloneq_value!!(
    ::SMCContext, vn::VarName, right, template, vi::DynamicPPL.AbstractVarInfo
)
    vi = DynamicPPL.store_coloneq_value!!(
        DynamicPPL.DefaultContext(), vn, right, template, vi
    )
    Libtask.get_taped_globals(Particle).varinfo = vi
    return vi
end

# Tell Libtask which calls may contain a `produce`, so it instruments them. The produce itself is in
# `acclogp`; everything else here is marked because it sits on a path that reaches it. Over-
# approximating is safe (a wrongly-marked call is merely instrumented); missing a real one is not.
#
#   observe:      tilde_observe!! -> accumulate_observe!! -> acclogp
#   @addlogprob!: accloglikelihood!! -> acclogp_and_mirror!! -> map_accumulator!! -> acclogp
#                 (the `@addlogprob! (; ...)` NamedTuple form routes through acclogp!! first)
#   Gibbs:        GibbsContext turns a tilde_assume!! into a tilde_observe!!
Libtask.@might_produce(DynamicPPL.tilde_observe!!)
Libtask.@might_produce(DynamicPPL.accumulate_observe!!)
Libtask.@might_produce(DynamicPPL.acclogp)
Libtask.@might_produce(DynamicPPL.tilde_assume!!)
Libtask.@might_produce(DynamicPPL.accloglikelihood!!)
Libtask.@might_produce(acclogp_and_mirror!!)
Libtask.@might_produce(DynamicPPL.map_accumulator!!)
Libtask.@might_produce(DynamicPPL.acclogp!!)
# Every model / submodel evaluator takes a `DynamicPPL.Model`, so this covers them all.
# See https://github.com/TuringLang/Libtask.jl/issues/217.
Libtask.might_produce_if_sig_contains(::Type{<:DynamicPPL.Model}) = true

# Add the raw sampled values (`:=` included, so a chain reports them as under any other sampler;
# they take no part in a sweep). `OnlyAccsVarInfo`'s default `LogPrior` and `LogJacobian` are kept
# on purpose: `ParamsWithStats` reads them off this varinfo to fill a chain's `logprior` and
# `logjoint` columns, and dropping them would silently omit those columns rather than error.
function trajectory_varinfo()
    return DynamicPPL.setacc!!(
        DynamicPPL.OnlyAccsVarInfo(), DynamicPPL.RawValueAccumulator(true)
    )
end

# A particle's varinfo is the same set with the produce-aware likelihood accumulator in place of the
# default one. Outside a sweep use `trajectory_varinfo`: there a `produce` has no task to suspend,
# and the mirroring keyed on that accumulator would write onto an unrelated particle.
function particle_varinfo()
    return DynamicPPL.setacc!!(trajectory_varinfo(), ProduceLogLikelihoodAccumulator())
end

#
# Resampling schemes
#

# For unconditional SMC, multinomial, stratified, and systematic resampling all have offspring
# counts satisfying `E[Oᵏ] = N·Wᵏ`. Multinomial resampling is broadly consistent, and
# stratified resampling is consistent under standard regularity conditions; systematic
# resampling is order-dependent and can fail to be consistent for arbitrary particle orderings
# (Gerber, Chopin & Whiteley, 2019), so stratified is the default unconditional scheme.
#
# Particle Gibbs additionally needs a valid *conditional* version of the chosen law, and
# pinning one ordered offspring is not it: for systematic resampling the conditional
# construction draws the grid offset from a weight-dependent mixture rather than `U[0,1]`, then
# randomly cycles the output so the reference lands in the pinned slot (Chopin & Singh, 2015,
# Algorithm 4, https://doi.org/10.3150/14-BEJ629; see also Finke, Johansen, Lee & Murray,
# "Resampling in conditional SMC algorithms", https://arxiv.org/abs/2606.25603). Independent
# multinomial draws stay valid once the reference ancestor is pinned, so `resample_propagate!`
# uses them for every scheme in a conditional sweep. That costs mixing -- Chopin & Singh find
# systematic resampling mixes noticeably better than multinomial in particle Gibbs -- so
# implementing the conditional schemes properly would be a genuine improvement.

##
## Resampler interface
##

abstract type AbstractResampler end

"""Whether to resample given the normalized `weights`. Bare schemes always resample."""
should_resample(::AbstractResampler, weights) = true

"""Draw `n` ancestor indices from `1:length(weights)` with probabilities `weights`."""
function resample_indices end

##
## Schemes
##

"Multinomial resampling: `n` independent draws from the categorical over `weights`."
struct MultinomialResampler <: AbstractResampler end
function resample_indices(rng::AbstractRNG, ::MultinomialResampler, weights, n::Integer)
    return rand(rng, Distributions.Categorical(weights), n)
end

# Stratified and systematic resampling are the same walk up the cumulative weights, differing only
# in where each stratum's offset comes from, so `offset(k)` supplies it. Both schemes draw their
# uniforms in the same order as a hand-written loop would -- note `rand(rng, n)` would *not* be
# equivalent, since Julia fills arrays through a SIMD path that yields a different stream.
function inverse_cdf_indices(weights, n::Integer, offset)
    v = n * weights[1]
    indices = Vector{Int}(undef, n)
    s = 1
    for k in 1:n
        u = oftype(v, offset(k))
        # `s < length(weights)` guards the last particle: if `weights` sums to slightly under one
        # (softmax rounding), `v` can fall a hair short of `u` at the final stratum and the
        # unguarded walk would index past the end.
        while s < length(weights) && v < u
            s += 1
            v += n * weights[s]
        end
        indices[k] = s
    end
    return indices
end

"Stratified resampling: one independent uniform per stratum of width `1/n`."
struct StratifiedResampler <: AbstractResampler end
function resample_indices(rng::AbstractRNG, ::StratifiedResampler, weights, n::Integer)
    return inverse_cdf_indices(weights, n, k -> (k - 1) + rand(rng))
end

"Systematic resampling: one shared uniform placed on a regular grid of `n` points."
struct SystematicResampler <: AbstractResampler end
function resample_indices(rng::AbstractRNG, ::SystematicResampler, weights, n::Integer)
    u = rand(rng)
    return inverse_cdf_indices(weights, n, k -> (k - 1) + u)
end

##
## Effective-sample-size gating
##

"""
    ESSThresholdResampler(threshold, scheme = StratifiedResampler())

Resample with `scheme`, but only when the effective sample size drops below
`threshold * nparticles`. This is the default for [`SMC`](@ref) and [`PG`](@ref).
"""
struct ESSThresholdResampler{T<:Real,R<:AbstractResampler} <: AbstractResampler
    threshold::T
    scheme::R
end
function ESSThresholdResampler(threshold::Real)
    return ESSThresholdResampler(threshold, StratifiedResampler())
end

function should_resample(resampler::ESSThresholdResampler, weights)
    return weight_ess(weights) ≤ resampler.threshold * length(weights)
end
function resample_indices(
    rng::AbstractRNG, resampler::ESSThresholdResampler, weights, n::Integer
)
    return resample_indices(rng, resampler.scheme, weights, n)
end

#
# Particle sweep
#

# In a conditional sweep the last particle is the reference: it is always retained and reuses the
# retained trajectory's values, while the other `n-1` slots are resampled from all `n` particles
# (so they may descend from the reference).

##
## Weights and diagnostics
##

logweights(particles) = [p.logweight for p in particles]
normalized_weights(particles) = softmax(logweights(particles))
log_normalizing_constant(particles) = logsumexp(logweights(particles))
"""
Effective sample size of a normalised weight vector, `1 / Σ wᵢ²`. Named for the weights to keep it
distinct from `MCMCDiagnosticTools.ess`, which Turing re-exports and which measures a *chain's*
autocorrelation rather than a population's weight degeneracy.
"""
weight_ess(weights) = inv(sum(abs2, weights))

##
## Reweighting
##

# Advance one particle by one observation, folding its incremental weight in; return `true`
# once it has finished (produced nothing). Factored out so the serial and multithreaded loops in
# `reweight!` share one body.
function advance_particle!(p::Particle)
    score = advance!(p)
    score === nothing && return true
    p.logweight += score
    return false
end

# Advance every particle by one observation; return `true` once all have finished. A model
# whose number of observations varies across executions leaves particles out of step.
#
# Each particle advances only its own state (rng, varinfo, task), and its rng was already seeded
# serially in `resample_propagate!`, so the multithreaded loop is race-free and gives results
# identical to the serial one; the shared sampler rng is untouched here.
function reweight!(particles, multithreaded::Bool)
    n = length(particles)
    n_done = if multithreaded
        # A shared counter would race, so record per particle and tally afterwards.
        finished = Vector{Bool}(undef, n)
        Threads.@threads for i in 1:n
            finished[i] = advance_particle!(particles[i])
        end
        count(finished)
    else
        count(advance_particle!, particles)
    end
    n_done == 0 && return false
    n_done == n && return true
    return error(
        "mis-aligned execution traces ($n_done/$n finished): the number of observations must not be random.",
    )
end

##
## Resample and propagate
##

# Resample (if the scheme calls for it) and propagate the survivors, or -- when not resampling --
# refresh each ordinary particle's seed so the next step draws fresh randomness. Returns whether it
# resampled, which tells `sweep!` what the total weight now is without recomputing it.
#
# Whether this is a conditional sweep is read off the particles rather than passed in: the reference
# always occupies the last slot, so `isreference` is the single source of truth and resampling cannot
# disagree with the rest of the sweep about which particle is pinned.
function resample_propagate!(rng::AbstractRNG, particles, resampler)
    weights = normalized_weights(particles)
    if !should_resample(resampler, weights)
        # The reference draws nothing (it reuses retained values), so only the others need a
        # fresh seed for the next step.
        for p in particles
            isreference(p) || refresh!(p.rng)
        end
        return false
    end

    n = length(particles)
    conditional = isreference(last(particles))
    # A conditional sweep draws the `n-1` free ancestors independently from the categorical
    # over the weights, whatever scheme `resampler` names -- see the resampling-schemes
    # section for why the named scheme's conditional version is not simply "pin one draw".
    ancestors = if conditional
        resample_indices(rng, MultinomialResampler(), weights, n - 1)
    else
        resample_indices(rng, resampler, weights, n)
    end
    parents = copy(particles)
    taken = falses(n)
    for (slot, a) in enumerate(ancestors)
        # A parent's first offspring continues in the parent's own object; extra offspring need
        # the costly `deepcopy`, and so does any offspring of the reference, which has to survive
        # the sweep intact. Either way the child is reseeded to continue independently.
        first_offspring = !taken[a] && !isreference(parents[a])
        taken[a] = true
        child = first_offspring ? reseed!(parents[a], rng) : fork(parents[a], rng)
        child.logweight = zero(DynamicPPL.LogProbType)
        particles[slot] = child
    end
    # reference retained, weight reset
    conditional && (particles[n].logweight = zero(DynamicPPL.LogProbType))
    return true
end

##
## One sweep
##

# Run a full particle sweep in place, returning the log-evidence estimate and -- when `ess` is set --
# the per-observation effective sample sizes. Only `SMC` reports those, and `PG` runs thousands of
# sweeps, so computing them unconditionally would be pure waste on the sampler that sweeps most.
function sweep!(
    rng::AbstractRNG, particles, resampler, multithreaded::Bool; ess::Bool=false
)
    logZ = zero(DynamicPPL.LogProbType)
    # The ESS values are computed from the particle weights, so they follow whatever
    # `DynamicPPL.LogProbType` is rather than being pinned to `Float64`.
    ess_per_step = DynamicPPL.LogProbType[]
    # Total log weight entering the step. Resampling zeroes every weight, so it is then exactly
    # `log(n)`; otherwise the weights are untouched and it is still last step's total. Either way
    # there is nothing to recompute -- particles start at weight zero, hence `log(n)` initially.
    logZ0 = log(oftype(logZ, length(particles)))
    while true
        done = reweight!(particles, multithreaded)
        # Each observation contributes the log-ratio of total weight it adds; summed over the
        # sweep these telescope into an estimate of the model's log-evidence log p(y).
        total = log_normalizing_constant(particles)
        logZ += total - logZ0
        logZ0 = total
        done && break
        # Post-reweight ESS for this observation: a degeneracy diagnostic (low ESS means few
        # particles carry the weight). After the break, so the finishing pass -- which adds no
        # observation and leaves the weights unchanged -- contributes no spurious entry.
        ess && push!(ess_per_step, weight_ess(normalized_weights(particles)))
        # Resample between observations, never before the first: there the weights are still all
        # equal, so a multinomial draw -- what every conditional sweep uses -- would duplicate
        # particles before any data. One resample is still spent after the last observation, since
        # a particle reveals that it has finished only by producing nothing, one pass later.
        resampled = resample_propagate!(rng, particles, resampler)
        resampled && (logZ0 = log(oftype(logZ, length(particles))))
    end
    return logZ, ess_per_step
end

#
# Sequential Monte Carlo
#

abstract type ParticleInference <: AbstractSampler end

"""
$(TYPEDEF)

Sequential Monte Carlo sampler.

# Fields

$(TYPEDFIELDS)
"""
struct SMC{R<:AbstractResampler} <: ParticleInference
    "resampling scheme"
    resampler::R
    "reweight the particles across threads within each sweep"
    multithreaded::Bool
    function SMC(resampler::R; multithreaded::Bool=false) where {R<:AbstractResampler}
        return new{R}(resampler, multithreaded)
    end
end

"""
    SMC([resampler = ESSThresholdResampler(0.5)]; multithreaded = false)
    SMC([scheme = StratifiedResampler(), ]threshold; multithreaded = false)

Sequential Monte Carlo sampler. By default stratified resampling is triggered whenever the
effective sample size drops below half the number of particles.

Set `multithreaded = true` to evaluate the particles across threads within each sweep; results are
unchanged (start Julia with multiple threads, e.g. `julia -t auto`, for this to have effect).
Threads are the only option here: a suspended particle is a live Libtask task and cannot be
serialised, so a single sweep cannot be spread across processes. Passing `MCMCThreads()` or
`MCMCDistributed()` to [`sample`](@ref) parallelises whole chains instead, which is a separate axis
and composes with this one.

The resampling scheme types (`StratifiedResampler`, `SystematicResampler`, `MultinomialResampler`, `ESSThresholdResampler`) are
not exported; refer to them as e.g. `Turing.Inference.SystematicResampler`.
"""
SMC(; kwargs...) = SMC(ESSThresholdResampler(0.5); kwargs...)
SMC(threshold::Real; kwargs...) = SMC(ESSThresholdResampler(threshold); kwargs...)
function SMC(scheme::AbstractResampler, threshold::Real; kwargs...)
    return SMC(ESSThresholdResampler(threshold, scheme); kwargs...)
end

# SMC is a single weighted sweep, not a Markov chain: rather than fake an iteration through
# AbstractMCMC's step loop (returning the population one particle at a time), we run the sweep
# and bundle the whole population into the chain in one shot. `discard_initial`/`thinning`
# therefore have nothing to apply to.
function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    sampler::SMC,
    nparticles::Integer;
    check_model=true,
    chain_type=DEFAULT_CHAIN_TYPE,
    discard_initial=0,
    thinning=1,
    initial_params=nothing,
    callback=nothing,
    verbose=false,
    kwargs...,
)
    check_model && Turing._check_model(model, sampler)
    error_if_threadsafe_eval(model)
    if discard_initial > 0 || thinning > 1
        @warn "SMC does not support `discard_initial` or `thinning`; they are ignored."
    end
    if initial_params !== nothing && !(initial_params isa DynamicPPL.InitFromPrior)
        @warn "SMC draws its initial population from the prior; `initial_params` is ignored."
    end
    # Accepted only so it can be reported as ignored: AbstractMCMC's contract is one callback
    # per step, and SMC is a single sweep, so there is no iteration to call back from.
    if callback !== nothing
        @warn "SMC runs one sweep rather than an MCMC loop, so there are no per-iteration callbacks; `callback` is ignored."
    end
    particles = [Particle(model, particle_rng(rng)) for _ in 1:nparticles]
    logZ, ess_per_step = sweep!(
        rng, particles, sampler.resampler, sampler.multithreaded; ess=true
    )
    weights = normalized_weights(particles)
    # The sweep ends on a reweight, so resample once more -- unconditionally, unlike the ESS-gated
    # resampling inside it -- to return an equal-weight sample that needs no weighting downstream.
    ancestors = resample_indices(rng, sampler.resampler, weights, nparticles)
    # `log_normalizing_constant` and `ess_per_step` are sweep-level, so every returned particle carries the
    # same values.
    transitions = map(ancestors) do a
        DynamicPPL.ParamsWithStats(
            particles[a].varinfo, (; log_normalizing_constant=logZ, ess_per_step)
        )
    end
    chain = AbstractMCMC.bundle_samples(
        transitions, model, sampler, nothing, chain_type; kwargs...
    )
    post_sample_hook(chain, sampler; verbose)
    return chain
end

#
# Particle Gibbs / conditional SMC
#

"""
$(TYPEDEF)

Particle Gibbs (conditional SMC) sampler.

# Fields

$(TYPEDFIELDS)
"""
struct PG{R<:AbstractResampler} <: ParticleInference
    "number of particles"
    nparticles::Int
    "resampling scheme"
    resampler::R
    "reweight the particles across threads within each sweep"
    multithreaded::Bool
    function PG(
        nparticles::Int, resampler::R; multithreaded::Bool=false
    ) where {R<:AbstractResampler}
        return new{R}(nparticles, resampler, multithreaded)
    end
end

"""
    PG(n, [resampler = ESSThresholdResampler(0.5)]; multithreaded = false)
    PG(n, [scheme = StratifiedResampler(), ]threshold; multithreaded = false)

Particle Gibbs sampler with `n` particles. By default resampling is triggered whenever the
effective sample size drops below half the number of particles. The selected scheme applies to the
unconditional first sweep only; conditional sweeps draw their ancestors from the categorical over
the weights, for the reason given in the resampling-schemes section of this file.

Set `multithreaded = true` to evaluate the particles across threads within each sweep; results are
unchanged (start Julia with multiple threads, e.g. `julia -t auto`, for this to have effect).
Threads are the only option here: a suspended particle is a live Libtask task and cannot be
serialised, so a single sweep cannot be spread across processes. Passing `MCMCThreads()` or
`MCMCDistributed()` to [`sample`](@ref) parallelises whole chains instead, which is a separate axis
and composes with this one.

!!! warning "`log_normalizing_constant` is biased for PG"
    PG chains carry `log_normalizing_constant`, but its exponential is not an unbiased estimator of
    `p(y)` and must not be used for model comparison. A conditional sweep retains the reference
    whatever its weight. Because the reference is a posterior draw rather than a proposal draw, it
    usually has much higher likelihood than a fresh particle and inflates the mean weight at each
    step. In a linear Gaussian SSM with known `p(y)`, `E[Ẑ]` exceeded `p(y)` by 80% at `n = 16`
    and 16% at `n = 64`; the bias decreased approximately as `1/n` in that experiment but remained
    substantial at these particle counts. Use [`SMC`](@ref) when an unbiased estimator of `p(y)` is
    required. Its likelihood-scale estimate `exp(log_normalizing_constant)` is unbiased under the
    usual particle-filter assumptions.
"""
PG(n::Int; kwargs...) = PG(n, ESSThresholdResampler(0.5); kwargs...)
PG(n::Int, threshold::Real; kwargs...) = PG(n, ESSThresholdResampler(threshold); kwargs...)
function PG(n::Int, scheme::AbstractResampler, threshold::Real; kwargs...)
    return PG(n, ESSThresholdResampler(threshold, scheme); kwargs...)
end

"Conditional SMC, an alias for [`PG`](@ref)."
const CSMC = PG

"""
    PGState(trajectory)

Particle Gibbs sampler state: the retained trajectory's raw values, which the next sweep's
reference particle reuses. Plain data, because sampler state has to survive `save_state=true` and
`MCMCDistributed()`, whereas the [`Particle`](@ref) it is read off owns a live `Libtask.TapedTask`
that cannot be serialised. Nothing else needs carrying over: the reference consumes no randomness of
its own, and every other particle is seeded from the sampler's `rng`.
"""
struct PGState{V<:DynamicPPL.VarNamedTuple}
    trajectory::V
end

# First iteration: an ordinary (unconditional) particle sweep.
function AbstractMCMC.step(
    rng::AbstractRNG, model::DynamicPPL.Model, sampler::PG; discard_sample=false, kwargs...
)
    error_if_threadsafe_eval(model)
    particles = [Particle(model, particle_rng(rng)) for _ in 1:(sampler.nparticles)]
    logZ, _ = sweep!(rng, particles, sampler.resampler, sampler.multithreaded)
    return pg_transition_and_state(rng, particles, logZ, discard_sample)
end

# Subsequent iterations: conditional SMC given the retained trajectory, which the reference
# particle reproduces by reusing the retained values (see the `reference` field).
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    sampler::PG,
    state::PGState;
    discard_sample=false,
    kwargs...,
)
    error_if_threadsafe_eval(model)
    n = sampler.nparticles
    # Passing the retained trajectory makes this the reference (see the `reference` field). It is
    # handed a generator like any other particle, but never reads it: every draw comes from the
    # retained values, and its forks are reseeded from `rng` in `reseed!`.
    reference = Particle(model, particle_rng(rng), state.trajectory)
    # `n - 1` fresh particles, with the reference last -- the slot `resample_propagate!` retains.
    particles = [Particle(model, particle_rng(rng)) for _ in 1:(n - 1)]
    push!(particles, reference)
    logZ, _ = sweep!(rng, particles, sampler.resampler, sampler.multithreaded)
    return pg_transition_and_state(rng, particles, logZ, discard_sample)
end

function pg_transition_and_state(rng, particles, logZ, discard_sample)
    retained = particles[rand(
        rng, Distributions.Categorical(normalized_weights(particles))
    )]
    transition = if discard_sample
        nothing
    else
        DynamicPPL.ParamsWithStats(
            deepcopy(retained.varinfo), (; log_normalizing_constant=logZ)
        )
    end
    return transition, PGState(DynamicPPL.get_raw_values(retained.varinfo))
end

#
# Gibbs interface
#

gibbs_get_raw_values(state::PGState) = state.trajectory

function gibbs_update_state!!(
    ::PG, state::PGState, model::DynamicPPL.Model, global_vals::DynamicPPL.VarNamedTuple
)
    # Re-derive the retained trajectory under the values the other Gibbs components have since
    # updated, keeping only the addresses this model still visits. The `nothing` fallback errors on
    # an address `global_vals` lacks rather than inventing one; `GibbsContext` puts every address
    # the component owns into the global values before we get here.
    init = DynamicPPL.InitFromParams(global_vals, nothing)
    vi = last(DynamicPPL.init!!(model, trajectory_varinfo(), init, DynamicPPL.UnlinkAll()))
    return PGState(DynamicPPL.get_raw_values(vi))
end
