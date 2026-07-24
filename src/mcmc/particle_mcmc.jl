###
### Particle filtering and particle MCMC samplers: SMC, PG / conditional SMC.
###
### Key design.
### A probabilistic model becomes a particle filter by reading each `observe` statement as one
### filtering step. Evaluated under `SMCContext`, every likelihood term calls
### `Libtask.produce`, so a *particle* is a suspended model execution: we `advance!` it to its
### next `observe`, take the produced log-likelihood as its weight, then resample. SMC is one
### such sweep; particle Gibbs (PG/CSMC) runs a *conditional* sweep -- one particle is a fixed
### reference trajectory -- inside an MCMC loop.
###
### The reference reproduces the retained trajectory by *reusing its values*: the sampler state
### carries those values, and the reference re-runs the model with `InitFromParams`, so it stays
### the retained trajectory even when the model is re-conditioned between Gibbs sweeps (e.g. a
### state-space transition prior that depends on a parameter owned by another Gibbs component).
### A particle forked from the reference forgets the remaining values (in `reseed!`), so
### branching samples fresh with no per-particle flag. Each particle's counter-based `TracedRNG`
### supplies splittable, version-stable seeds that decorrelate the fresh draws across particles
### and Julia versions.
###
### Sections below: traced RNG; model evaluation via Libtask; resampling schemes; the particle
### sweep; the SMC sampler; the PG/CSMC sampler; the Gibbs-component interface.
###
### Reference: Andrieu, Doucet & Holenstein, "Particle Markov chain Monte Carlo methods",
### Journal of the Royal Statistical Society: Series B 72(3), 269-342 (2010).
###

using StatsFuns: softmax, logsumexp
import Random123

#
# Traced RNG
#
# A counter-based RNG that records the seed used at each model step, so that a particle's
# trajectory can be replayed exactly: the conditional-SMC reference regenerates itself by
# replaying its recorded seeds. This section comes first because `Particle` names `TracedRNG`
# in its type signature.

"""
    TracedRNG([rng = Random.default_rng()])

A `Random123.Philox2x` generator that remembers the seed (`key`) it used at each model step
in `keys`, indexed by the step counter `count`.

  - [`save_state!`](@ref) records the current seed (ordinary particles);
  - [`load_state!`](@ref) restores `keys[count]`, replaying that step's randomness (the
    reference trajectory).
"""
mutable struct TracedRNG{K<:Unsigned,T<:Random123.AbstractR123} <: Random.AbstractRNG
    count::Int
    rng::T
    keys::Vector{K}
end

function TracedRNG(inner::Random123.AbstractR123{T}) where {T<:Unsigned}
    Random123.set_counter!(inner, 0)
    return TracedRNG(1, inner, T[])
end
function TracedRNG(rng::AbstractRNG=Random.default_rng())
    inner = Random.seed!(Random123.Philox2x(), rand(rng, Random.Sampler(rng, UInt64)))
    return TracedRNG(inner)
end

Random.rng_native_52(trng::TracedRNG) = Random.rng_native_52(trng.rng)
Random.rand(trng::TracedRNG, ::Type{T}) where {T<:Unsigned} = Random.rand(trng.rng, T)

"The current seed of the inner generator."
inner_key(rng::Random123.Philox2x) = rng.key

"Reseed and rewind the inner generator. The model-step counter is left untouched."
function Random.seed!(trng::TracedRNG, key::Integer)
    Random.seed!(trng.rng, key)
    Random123.set_counter!(trng.rng, 0)
    return trng
end

"Record the seed used at the current step."
save_state!(trng::TracedRNG) = push!(trng.keys, inner_key(trng.rng))

"Replay the seed recorded at the current step."
load_state!(trng::TracedRNG) = Random.seed!(trng, trng.keys[trng.count])

"Advance the model-step counter by one."
inc_step!(trng::TracedRNG) = (trng.count += 1; trng)

"Rewind the model-step counter to the first step, so a trajectory replays from the start."
rewind!(trng::TracedRNG) = (trng.count = 1; trng)

# Derive a fresh seed from `key`. Splitting one generator into many by re-seeding is fragile
# in two ways: the derived seeds can yield *correlated* streams (Steele et al., "Fast
# Splittable Pseudorandom Number Generators", OOPSLA 2014), and a stdlib `MersenneTwister`
# derivation is not identical across Julia versions (Julia does not guarantee reproducible
# streams), which made SMC/PG drift between versions even under a StableRNG. Both bit the
# previous AdvancedPS implementation (#2781, AdvancedPS.jl#110). Philox is a counter-based
# generator with a fixed, portable algorithm and strong avalanche, so deriving the seed
# through it is both well-decorrelated from its parent and version-stable.
split_key(key::Integer) = rand(Random.seed!(Random123.Philox2x(), key), typeof(key))

"Reseed from the generator's own current state (used between steps when not resampling)."
refresh!(trng::TracedRNG) = Random.seed!(trng, split_key(inner_key(trng.rng)))

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

Leaf context marking a model evaluation as a particle-filter step: `tilde_assume!!` draws
from the prior using the particle's [`TracedRNG`](@ref), and `tilde_observe!!` scores the
observation and `Libtask.produce`s the increment as the particle's weight.
"""
struct SMCContext <: DynamicPPL.AbstractContext end

# `OnlyAccsVarInfo` needs a parameter eltype; `Any` is fine here since particle MCMC never
# involves AD or tracer types (see the `get_param_eltype` docstring in DynamicPPL).
DynamicPPL.get_param_eltype(::DynamicPPL.AbstractVarInfo, ::SMCContext) = Any

"""
    Particle(model, varinfo, rng::TracedRNG)

A single particle: a suspended `model` execution together with its `varinfo`, its own
replayable `rng`, and an accumulated `logweight`. It also serves directly as the particle
Gibbs sampler state (there is no separate state struct).
"""
mutable struct Particle{RT<:TracedRNG,WT<:Real}
    # Abstract on purpose: the VarInfo type can change during PG-inside-Gibbs. Accesses go
    # through Libtask's (already type-unstable) taped globals, so this costs nothing extra.
    varinfo::DynamicPPL.AbstractVarInfo
    rng::RT
    # `logweight` tracks whatever `DynamicPPL.LogProbType` is, so weights follow suit if it
    # is ever changed.
    logweight::WT
    # The retained trajectory's values, which the CSMC reference reproduces by reusing them
    # (`InitFromParams` in `tilde_assume!!`); empty for ordinary particles. `reseed!` clears it
    # so a particle forked off the reference samples fresh beyond the fork point. Reproducing
    # by *value* (not by replayed RNG seeds) is what keeps the reference the retained trajectory
    # when the model is re-conditioned between Gibbs sweeps. A draw is a deterministic function
    # x = g(u; θ) of the RNG output `u` and the distribution parameters θ (canonically the
    # inverse-CDF, x = F⁻¹(u; θ)). Seed-replay fixes `u` and recomputes x' = g(u; θ'); value-replay
    # reuses x' = x. These agree only when θ' = θ. Re-conditioning updates a value θ depends on
    # (owned by another block), so θ' ≠ θ and the replayed draw moves with the changed distribution
    # -- e.g. x ~ Normal(μ, 1) draws x = μ + Φ⁻¹(u), so after μ → μ' the same `u` gives x + (μ' − μ),
    # not x.
    reference_values::DynamicPPL.VarNamedTuple
    task::Libtask.TapedTask
    # `task` is filled in once the particle exists, because the task must capture the
    # particle as its taped globals (a back-reference). This has to be an inner constructor
    # for that reason: `task` is left undefined here and set immediately after.
    function Particle(
        vi::DynamicPPL.AbstractVarInfo,
        rng::RT,
        reference_values::DynamicPPL.VarNamedTuple=DynamicPPL.VarNamedTuple(),
    ) where {RT<:TracedRNG}
        w = zero(DynamicPPL.LogProbType)
        return new{RT,typeof(w)}(vi, rng, w, reference_values)
    end
end

function Particle(
    model::DynamicPPL.Model,
    varinfo::DynamicPPL.AbstractVarInfo,
    rng::TracedRNG,
    reference_values::DynamicPPL.VarNamedTuple=DynamicPPL.VarNamedTuple(),
)
    model = DynamicPPL.setleafcontext(model, SMCContext())
    args, kwargs = DynamicPPL.make_evaluate_args_and_kwargs(model, varinfo)
    particle = Particle(deepcopy(varinfo), rng, reference_values)
    particle.task = Libtask.TapedTask(particle, model.f, args...; kwargs...)
    return particle
end

"""
    reseed!(particle, rng)

Restart `particle` as a fresh continuation seeded from `rng`: it switches from replaying to
sampling afresh, and `keys` is truncated to the steps already taken so a particle descended
from the reference forgets the reference's future. Mutates and returns `particle`.
"""
function reseed!(particle::Particle, rng::AbstractRNG)
    Random.seed!(particle.rng, rand(rng, UInt64))
    resize!(particle.rng.keys, particle.rng.count - 1)
    # A fork samples fresh from here on, so it must forget the reference's remaining values.
    particle.reference_values = DynamicPPL.VarNamedTuple()
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
    advance!(particle, isref) -> Union{Real,Nothing}

Run the particle to its next `observe`, returning the incremental log-likelihood, or
`nothing` once the model finishes. An ordinary particle records the step's seed; the
reference (`isref = true`) replays its recorded seed instead.
"""
function advance!(particle::Particle, isref::Bool)
    isref ? load_state!(particle.rng) : save_state!(particle.rng)
    inc_step!(particle.rng)
    return Libtask.consume(particle.task)
end

function DynamicPPL.tilde_assume!!(
    ::SMCContext, dist::Distribution, vn::VarName, template, ::DynamicPPL.AbstractVarInfo
)
    particle = Libtask.get_taped_globals(Particle)
    # Reuse the retained value (`InitFromParams`) if this particle is reproducing the CSMC
    # reference trajectory and still carries this variable; otherwise draw from the prior.
    # `reference_values` is empty for ordinary particles and is cleared by `reseed!` on a fork,
    # so a fork of the reference draws fresh past the fork point (see the `Particle` fields).
    strategy = if haskey(particle.reference_values, vn)
        DynamicPPL.InitFromParams(particle.reference_values, nothing)
    else
        DynamicPPL.InitFromPrior()
    end
    ctx = DynamicPPL.InitContext(particle.rng, strategy, DynamicPPL.UnlinkAll())
    x, vi = DynamicPPL.tilde_assume!!(ctx, dist, vn, template, particle.varinfo)
    particle.varinfo = vi
    return x, vi
end

# Reweighting invariant: a particle's per-step score is `produce`d from here and from the
# `accloglikelihood!!` overload (for `@addlogprob!`), *after* the likelihood accumulator is
# updated, and equals the accumulator's increment. Producing after the update -- rather than
# inside `acclogp` -- keeps the produced weight in step with the accumulated log-likelihood
# (no one-step lag) and lets `@addlogprob!` terms reach the accumulator, not just the weight.
function DynamicPPL.tilde_observe!!(
    ::SMCContext,
    dist::Distribution,
    left,
    vn::Union{VarName,Nothing},
    template,
    ::DynamicPPL.AbstractVarInfo,
)
    particle = Libtask.get_taped_globals(Particle)
    before = DynamicPPL.getloglikelihood(particle.varinfo)
    left, vi = DynamicPPL.tilde_observe!!(
        DynamicPPL.DefaultContext(), dist, left, vn, template, particle.varinfo
    )
    particle.varinfo = vi
    Libtask.produce(DynamicPPL.getloglikelihood(vi) - before)
    return left, vi
end

"""
    ProduceLogLikelihoodAccumulator{T} <: LogProbAccumulator{T}

A marker likelihood accumulator: it accumulates exactly like `LogLikelihoodAccumulator`, but
its distinct type flags a varinfo as belonging to a particle, so the produce sites know to
emit. The produce happens in [`tilde_observe!!`](@ref) (observations) and the
`accloglikelihood!!` overload below (`@addlogprob!`, issue #1996) -- in each case from the
increase in accumulated log-likelihood, keeping the accumulator the single source of truth.
"""
struct ProduceLogLikelihoodAccumulator{T<:Real} <: DynamicPPL.LogProbAccumulator{T}
    logp::T
end

DynamicPPL.accumulator_name(::Type{<:ProduceLogLikelihoodAccumulator}) = :LogLikelihood
DynamicPPL.logp(acc::ProduceLogLikelihoodAccumulator) = acc.logp
# `acclogp` is inherited from the generic `LogProbAccumulator` method (plain addition); the
# produce is handled by the produce sites, not here.

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

# `@addlogprob!` bypasses `tilde_observe!!`, so its produce is emitted here instead -- again
# only once the accumulator has been updated. Gated on the producing accumulator, so outside
# particle evaluation this reduces to the default (non-producing) method (issue #1996).
function DynamicPPL.accloglikelihood!!(
    vi::DynamicPPL.OnlyAccsVarInfo, logp; ignore_missing_accumulator=false
)
    acc_name = Val(:LogLikelihood)
    if ignore_missing_accumulator && !DynamicPPL.hasacc(vi, acc_name)
        return vi
    end
    is_particle = DynamicPPL.getacc(vi, acc_name) isa ProduceLogLikelihoodAccumulator
    before = is_particle ? DynamicPPL.getloglikelihood(vi) : zero(DynamicPPL.LogProbType)
    vi = DynamicPPL.map_accumulator!!(acc -> DynamicPPL.acclogp(acc, logp), vi, acc_name)
    if is_particle
        particle = Libtask.get_taped_globals(Particle)
        particle.varinfo = vi
        Libtask.produce(DynamicPPL.getloglikelihood(vi) - before)
    end
    return vi
end

# Tell Libtask which calls may contain a `produce`, so it instruments them. The produce lives
# in `tilde_observe!!` and `accloglikelihood!!`; the rest of each chain is marked so Libtask
# tapes through to reach it. Over-approximating is safe (a wrongly-marked call just gets
# instrumented); missing a real one is not, so we err towards marking.
#
#   observe:      tilde_observe!! accumulates (accumulate_observe!! -> acclogp), then produces
#   @addlogprob!: accloglikelihood!! accumulates (map_accumulator!! -> acclogp), then produces
#                 (the `@addlogprob! (; ...)` NamedTuple form routes through acclogp!! first)
#   Gibbs:        GibbsContext turns a tilde_assume!! into a tilde_observe!!
Libtask.@might_produce(DynamicPPL.tilde_observe!!)
Libtask.@might_produce(DynamicPPL.accumulate_observe!!)
Libtask.@might_produce(DynamicPPL.acclogp)
Libtask.@might_produce(DynamicPPL.tilde_assume!!)
Libtask.@might_produce(DynamicPPL.accloglikelihood!!)
Libtask.@might_produce(DynamicPPL.map_accumulator!!)
Libtask.@might_produce(DynamicPPL.acclogp!!)
# Every model / submodel evaluator takes a `DynamicPPL.Model`, so this covers them all.
# See https://github.com/TuringLang/Libtask.jl/issues/217.
Libtask.might_produce_if_sig_contains(::Type{<:DynamicPPL.Model}) = true

# A particle needs only the produce-aware likelihood accumulator (which drives reweighting)
# and the raw sampled values. The prior/Jacobian terms shown in chain metadata are recomputed
# downstream from the raw values, so accumulating them per particle would be wasted work.
function particle_varinfo()
    vi = DynamicPPL.OnlyAccsVarInfo()
    vi = DynamicPPL.setacc!!(vi, ProduceLogLikelihoodAccumulator())
    vi = DynamicPPL.setacc!!(vi, DynamicPPL.RawValueAccumulator(true))
    return vi
end

#
# Resampling schemes
#
# On theoretical correctness: particle Gibbs (and the SMC evidence estimate) are justified
# for resampling schemes whose offspring counts satisfy `E[Oᵏ] = N·Wᵏ` (Andrieu, Doucet &
# Holenstein, 2010, Assumption 2). Multinomial and stratified resampling meet this and are
# also consistent as `N → ∞`. Systematic resampling has the same expected counts, but its
# single shared uniform makes it order-dependent and it is not consistent in general (Gerber,
# Chopin & Whiteley, 2019), so it falls outside the particle Gibbs invariance proof. We
# therefore default to stratified resampling and offer systematic only as an explicit choice.

abstract type AbstractResampler end

"""Whether to resample given the normalized `weights`. Bare schemes always resample."""
should_resample(::AbstractResampler, weights) = true

"""Draw `n` ancestor indices from `1:length(weights)` with probabilities `weights`."""
function resample_indices end

"Multinomial resampling: `n` independent draws from the categorical over `weights`."
struct MultinomialResampler <: AbstractResampler end
function resample_indices(rng::AbstractRNG, ::MultinomialResampler, weights, n::Integer)
    return rand(rng, Distributions.Categorical(weights), n)
end

"Stratified resampling: one independent uniform per stratum of width `1/n`."
struct StratifiedResampler <: AbstractResampler end
function resample_indices(rng::AbstractRNG, ::StratifiedResampler, weights, n::Integer)
    v = n * weights[1]
    indices = Vector{Int}(undef, n)
    s = 1
    for k in 1:n
        u = oftype(v, (k - 1) + rand(rng))
        # `s < length(weights)` guards the last particle: if `weights` sums to slightly under
        # one (softmax rounding), `v` can fall a hair short of `u` at the final stratum and the
        # unguarded loop would index past the end.
        while s < length(weights) && v < u
            s += 1
            v += n * weights[s]
        end
        indices[k] = s
    end
    return indices
end

"Systematic resampling: one shared uniform placed on a regular grid of `n` points."
struct SystematicResampler <: AbstractResampler end
function resample_indices(rng::AbstractRNG, ::SystematicResampler, weights, n::Integer)
    v = n * weights[1]
    u = oftype(v, rand(rng))
    indices = Vector{Int}(undef, n)
    s = 1
    for k in 1:n
        # See `StratifiedResampler`: `s < length(weights)` keeps the final stratum from
        # indexing past the end when `weights` sums to slightly under one.
        while s < length(weights) && v < u
            s += 1
            v += n * weights[s]
        end
        indices[k] = s
        u += one(u)
    end
    return indices
end

"""
    ESSResampler(threshold, scheme = StratifiedResampler())

Resample with `scheme`, but only when the effective sample size drops below
`threshold * nparticles`. This is the default for [`SMC`](@ref) and [`PG`](@ref).
"""
struct ESSResampler{T<:Real,R<:AbstractResampler} <: AbstractResampler
    threshold::T
    scheme::R
end
ESSResampler(threshold::Real) = ESSResampler(threshold, StratifiedResampler())

function should_resample(resampler::ESSResampler, weights)
    return ess(weights) ≤ resampler.threshold * length(weights)
end
function resample_indices(rng::AbstractRNG, resampler::ESSResampler, weights, n::Integer)
    return resample_indices(rng, resampler.scheme, weights, n)
end

#
# Particle sweep
#
# In a conditional sweep the last particle is the reference: it is always retained and
# replays its recorded randomness, while the other `n-1` slots are resampled from all `n`
# particles (so they may descend from the reference).

logweights(particles) = [p.logweight for p in particles]
normalized_weights(particles) = softmax(logweights(particles))
log_normalizing_constant(particles) = logsumexp(logweights(particles))
"Effective sample size of a normalised weight vector, `1 / Σ wᵢ²`."
ess(weights) = inv(sum(abs2, weights))

# Advance one particle by one observation, folding its incremental weight in; return `true`
# once it has finished (produced nothing). Factored out so the serial and multithreaded loops in
# `reweight!` share one body.
function advance_particle!(p::Particle, isref::Bool)
    score = advance!(p, isref)
    score === nothing && return true
    p.logweight += score
    return false
end

# Advance every particle by one observation; return `true` once all have finished. A model
# whose number of observations varies across executions leaves particles out of step.
#
# `multithreaded` is *within-sweep* parallelism -- spreading this sweep's particle evaluations
# across threads. It is a separate axis from AbstractMCMC's chain-level ensemble
# (`MCMCThreads`/`MCMCDistributed`, which runs whole chains independently); the two compose.
# Only threading is offered here, not distribution: particles resample every step (all-to-all)
# and are live Libtask tasks, so spreading one sweep across processes would be communication-
# bound rather than a speed-up.
#
# Each particle advances only its own state (rng, varinfo, task), and its rng was already
# seeded serially in `resample_propagate!`, so the multithreaded loop is race-free and gives
# results identical to the serial one. Only the model evaluations parallelise; the shared
# sampler rng is untouched here.
function reweight!(particles, conditional::Bool, multithreaded::Bool)
    n = length(particles)
    if multithreaded
        # A shared counter would race, so collect per-particle results and tally afterwards.
        finished = Vector{Bool}(undef, n)
        Threads.@threads for i in 1:n
            finished[i] = advance_particle!(particles[i], conditional && i == n)
        end
        n_done = count(finished)
    else
        n_done = 0
        for i in 1:n
            n_done += advance_particle!(particles[i], conditional && i == n)
        end
    end
    n_done == 0 && return false
    n_done == n && return true
    return error(
        "mis-aligned execution traces ($n_done/$n finished): the number of observations must not be random.",
    )
end

# Resample (if the scheme calls for it) and propagate the survivors, or -- when not
# resampling -- refresh each ordinary particle's seed so the next step draws fresh randomness.
function resample_propagate!(rng::AbstractRNG, particles, resampler, conditional::Bool)
    n = length(particles)
    weights = normalized_weights(particles)
    if should_resample(resampler, weights)
        ancestors = resample_indices(rng, resampler, weights, conditional ? n - 1 : n)
        old = copy(particles)
        seen = falses(n)
        for (slot, a) in enumerate(ancestors)
            # Reuse each surviving parent's object for its first offspring; only extra
            # offspring -- and any offspring of the retained reference -- need the costly
            # `deepcopy`. Either way the child is reseeded to continue independently.
            reuse = !seen[a] && !(conditional && a == n)
            seen[a] = true
            child = reuse ? reseed!(old[a], rng) : fork(old[a], rng)
            child.logweight = zero(DynamicPPL.LogProbType)
            particles[slot] = child
        end
        # reference retained, weight reset
        conditional && (particles[n].logweight = zero(DynamicPPL.LogProbType))
    else
        for (i, p) in enumerate(particles)
            # Refresh every particle's seed except the reference, which keeps replaying.
            if !(conditional && i == n)
                refresh!(p.rng)
            end
        end
    end
    return nothing
end

# Run a full particle sweep in place, returning the log-evidence estimate and the
# per-observation effective sample sizes.
function sweep!(
    rng::AbstractRNG, particles, resampler, multithreaded::Bool; conditional::Bool=false
)
    logZ = zero(DynamicPPL.LogProbType)
    ess_per_step = Float64[]
    while true
        resample_propagate!(rng, particles, resampler, conditional)
        logZ0 = log_normalizing_constant(particles)
        done = reweight!(particles, conditional, multithreaded)
        # Each observation contributes the log-ratio of total weight it adds; summed over the
        # sweep these telescope into an estimate of the model's log-evidence log p(y).
        logZ += log_normalizing_constant(particles) - logZ0
        done && break
        # Post-reweight ESS for this observation: a degeneracy diagnostic (low ESS means few
        # particles carry the weight). After the break, so the finishing pass -- which adds no
        # observation and leaves the weights unchanged -- contributes no spurious entry.
        push!(ess_per_step, ess(normalized_weights(particles)))
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
    SMC([resampler = ESSResampler(0.5)]; multithreaded = false)
    SMC([scheme = StratifiedResampler(), ]threshold; multithreaded = false)

Sequential Monte Carlo sampler. By default stratified resampling is triggered whenever the
effective sample size drops below half the number of particles.

Set `multithreaded = true` to evaluate the particles across threads within each sweep; results are
unchanged (start Julia with multiple threads, e.g. `julia -t auto`, for this to have effect).

The resampling scheme types (`StratifiedResampler`, `SystematicResampler`, `MultinomialResampler`, `ESSResampler`) are
not exported; refer to them as e.g. `Turing.Inference.SystematicResampler`.
"""
SMC(; kwargs...) = SMC(ESSResampler(0.5); kwargs...)
SMC(threshold::Real; kwargs...) = SMC(ESSResampler(threshold); kwargs...)
function SMC(scheme::AbstractResampler, threshold::Real; kwargs...)
    return SMC(ESSResampler(threshold, scheme); kwargs...)
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
    verbose=false,
    kwargs...,
)
    check_model && Turing._check_model(model, sampler)
    error_if_threadsafe_eval(model)
    if discard_initial > 0 || thinning > 1
        @warn "SMC does not support `discard_initial` or `thinning`; they are ignored."
    end
    if initial_params !== nothing
        @warn "SMC draws its initial population from the prior; `initial_params` is ignored."
    end
    particles = [Particle(model, particle_varinfo(), TracedRNG(rng)) for _ in 1:nparticles]
    logZ, ess_per_step = sweep!(rng, particles, sampler.resampler, sampler.multithreaded)
    weights = normalized_weights(particles)
    # One final resampling step, so the returned particles are an equal-weight sample. The
    # sweep ends on a reweight, leaving the population weighted; resampling once here makes the
    # result a standard unweighted chain (so `mean(chain[...])` and friends need no weighting),
    # at the cost of a little resampling variance. Unconditional -- unlike the ESS-gated
    # resampling inside the sweep.
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
    PG(n, [resampler = ESSResampler(0.5)]; multithreaded = false)
    PG(n, [scheme = StratifiedResampler(), ]threshold; multithreaded = false)

Particle Gibbs sampler with `n` particles. By default stratified resampling is triggered
whenever the effective sample size drops below half the number of particles.

Set `multithreaded = true` to evaluate the particles across threads within each sweep; results are
unchanged (start Julia with multiple threads, e.g. `julia -t auto`, for this to have effect).
"""
PG(n::Int; kwargs...) = PG(n, ESSResampler(0.5); kwargs...)
PG(n::Int, threshold::Real; kwargs...) = PG(n, ESSResampler(threshold); kwargs...)
function PG(n::Int, scheme::AbstractResampler, threshold::Real; kwargs...)
    return PG(n, ESSResampler(threshold, scheme); kwargs...)
end

"Conditional SMC, an alias for [`PG`](@ref)."
const CSMC = PG

# PG's sampler state is just the retained `Particle`: it already carries the reference
# trajectory's `varinfo` and `rng` (its `task`/`logweight` are then unused), so there is no
# dedicated state struct.

# First iteration: an ordinary (unconditional) particle sweep.
function AbstractMCMC.step(
    rng::AbstractRNG, model::DynamicPPL.Model, sampler::PG; discard_sample=false, kwargs...
)
    error_if_threadsafe_eval(model)
    particles = [
        Particle(model, particle_varinfo(), TracedRNG(rng)) for _ in 1:(sampler.nparticles)
    ]
    logZ, _ = sweep!(rng, particles, sampler.resampler, sampler.multithreaded)
    return pg_transition_and_state(rng, particles, logZ, discard_sample)
end

# Subsequent iterations: conditional SMC given the retained trajectory, which the reference
# particle regenerates by replaying `state.rng` from the first step.
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    sampler::PG,
    state::Particle;
    discard_sample=false,
    kwargs...,
)
    error_if_threadsafe_eval(model)
    n = sampler.nparticles
    # The reference reproduces the retained trajectory by reusing its values (passed here and
    # consumed by `tilde_assume!!`), so it stays that trajectory even if the model was
    # re-conditioned since the last sweep. Its varinfo starts empty like any other particle.
    reference = Particle(
        model,
        particle_varinfo(),
        rewind!(deepcopy(state.rng)),
        DynamicPPL.get_raw_values(state.varinfo),
    )
    particles = map(1:n) do i
        i < n ? Particle(model, particle_varinfo(), TracedRNG(rng)) : reference
    end
    logZ, _ = sweep!(
        rng, particles, sampler.resampler, sampler.multithreaded; conditional=true
    )
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
    return transition, retained
end

#
# Gibbs interface
#

gibbs_get_raw_values(state::Particle) = DynamicPPL.get_raw_values(state.varinfo)

function gibbs_update_state!!(
    ::PG, state::Particle, model::DynamicPPL.Model, global_vals::DynamicPPL.VarNamedTuple
)
    init = DynamicPPL.InitFromParams(global_vals, nothing)
    # Re-initialise the reference varinfo with the values conditioned by other Gibbs
    # components. Mutating in place is safe: the caller replaces this state with the value we
    # return and never reads the pre-update one again.
    state.varinfo = last(
        DynamicPPL.init!!(model, state.varinfo, init, DynamicPPL.UnlinkAll())
    )
    return state
end
