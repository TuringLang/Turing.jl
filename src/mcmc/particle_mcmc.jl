###
### Particle filtering and particle MCMC samplers: SMC, PG / conditional SMC.
###
### Key design.
### A probabilistic model becomes a particle filter by reading each `observe` statement as one
### filtering step. Evaluated under `ParticleMCMCContext`, every likelihood term calls
### `Libtask.produce`, so a *particle* is a suspended model execution: we `advance!` it to its
### next `observe`, take the produced log-likelihood as its weight, then resample. SMC is one
### such sweep; particle Gibbs (PG/CSMC) runs a *conditional* sweep -- one particle is a fixed
### reference trajectory -- inside an MCMC loop.
###
### The reference is reproduced without storing its values: each particle's `TracedRNG`
### records the seed it drew at every step, and the reference simply *replays* those seeds
### (`load_state!`), regenerating its trajectory exactly. A particle forked from the reference
### is reseeded, which flips it from replaying to sampling fresh, so branching needs no
### per-particle flag. This is what keeps the reference handling small, and it rests on one
### invariant: every step must draw from a fresh seed -- guaranteed by the resample/refresh in
### `resample_propagate!` -- otherwise the recorded seeds collide and replay is wrong.
###
### Sections below: traced RNG; model evaluation via Libtask; resampling schemes; the particle
### sweep; the SMC sampler; the PG/CSMC sampler; the Gibbs-component interface.
###

using StatsFuns: softmax, logsumexp

#
# Traced RNG
#
# A counter-based RNG that records the seed used at each model step, so that a particle's
# trajectory can be replayed exactly: the conditional-SMC reference regenerates itself by
# replaying its recorded seeds. This section comes first because `Particle` and `PGState`
# name `TracedRNG` in their type signatures.

"""
    TracedRNG([rng = Random.default_rng()])

A `Random123.Philox2x` generator that remembers the seed (`key`) it used at each model step
in `keys`, indexed by the step counter `count`.

  - [`save_state!`](@ref) records the current seed (ordinary particles);
  - [`load_state!`](@ref) restores `keys[count]`, replaying that step's randomness (the
    reference trajectory).
"""
mutable struct TracedRNG{K,T<:Random123.AbstractR123} <: Random.AbstractRNG
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
function Random.seed!(trng::TracedRNG, key)
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

"Deterministically derive a fresh seed from `key`."
split_key(key::Integer) = rand(Random.MersenneTwister(key), typeof(key))

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
    ParticleMCMCContext

Leaf context marking a model evaluation as a particle-filter step: `tilde_assume!!` draws
from the prior using the particle's [`TracedRNG`](@ref), and `tilde_observe!!` scores the
observation, which [`ProduceLogLikelihoodAccumulator`](@ref) turns into a `Libtask.produce`.
"""
struct ParticleMCMCContext <: DynamicPPL.AbstractContext end

# `OnlyAccsVarInfo` needs a parameter eltype; `Any` is fine here since particle MCMC never
# involves AD or tracer types (see the `get_param_eltype` docstring in DynamicPPL).
DynamicPPL.get_param_eltype(::DynamicPPL.AbstractVarInfo, ::ParticleMCMCContext) = Any

"""
    Particle(model, varinfo, rng::TracedRNG)

A single particle: a suspended `model` execution together with its `varinfo`, its own
replayable `rng`, and an accumulated `logweight`.
"""
mutable struct Particle
    # Abstract on purpose: the VarInfo type can change during PG-inside-Gibbs. Accesses go
    # through Libtask's (already type-unstable) taped globals, so this costs nothing extra.
    varinfo::DynamicPPL.AbstractVarInfo
    rng::TracedRNG
    logweight::Float64
    task::Libtask.TapedTask
    # `task` is filled in once the particle exists, because the task must capture the
    # particle as its taped globals (a back-reference).
    Particle(vi, rng) = new(vi, rng, 0.0)
end

function Particle(
    model::DynamicPPL.Model, varinfo::DynamicPPL.AbstractVarInfo, rng::TracedRNG
)
    model = DynamicPPL.setleafcontext(model, ParticleMCMCContext())
    args, kwargs = DynamicPPL.make_evaluate_args_and_kwargs(model, varinfo)
    particle = Particle(deepcopy(varinfo), rng)
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
    advance!(particle, isref) -> Union{Float64,Nothing}

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
    ::ParticleMCMCContext,
    dist::Distribution,
    vn::VarName,
    template,
    ::DynamicPPL.AbstractVarInfo,
)
    particle = Libtask.get_taped_globals(Particle)
    ctx = DynamicPPL.InitContext(
        particle.rng, DynamicPPL.InitFromPrior(), DynamicPPL.UnlinkAll()
    )
    x, vi = DynamicPPL.tilde_assume!!(ctx, dist, vn, template, particle.varinfo)
    particle.varinfo = vi
    return x, vi
end

function DynamicPPL.tilde_observe!!(
    ::ParticleMCMCContext,
    dist::Distribution,
    left,
    vn::Union{VarName,Nothing},
    template,
    ::DynamicPPL.AbstractVarInfo,
)
    particle = Libtask.get_taped_globals(Particle)
    left, vi = DynamicPPL.tilde_observe!!(
        DynamicPPL.DefaultContext(), dist, left, vn, template, particle.varinfo
    )
    particle.varinfo = vi
    return left, vi
end

"""
    ProduceLogLikelihoodAccumulator{T} <: LogProbAccumulator{T}

Like `LogLikelihoodAccumulator`, but `Libtask.produce`s each likelihood increment as it is
accumulated. Because `@addlogprob!` also routes through `acclogp`, it too triggers a
`produce`, so manual likelihood terms reweight particles correctly (issue #1996).
"""
struct ProduceLogLikelihoodAccumulator{T<:Real} <: DynamicPPL.LogProbAccumulator{T}
    logp::T
end

DynamicPPL.accumulator_name(::Type{<:ProduceLogLikelihoodAccumulator}) = :LogLikelihood
DynamicPPL.logp(acc::ProduceLogLikelihoodAccumulator) = acc.logp

function DynamicPPL.acclogp(acc::ProduceLogLikelihoodAccumulator, val)
    Libtask.produce(val)  # the only difference from `LogLikelihoodAccumulator`
    return ProduceLogLikelihoodAccumulator(acc.logp + val)
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

# Tell Libtask which calls may contain `produce`, walking up the call stack from `acclogp`.
# Over-approximating is safe (a wrongly-marked call just gets instrumented); missing a real
# one is not, so we err towards marking.
Libtask.@might_produce(DynamicPPL.accloglikelihood!!)
# Merging accumulators (across submodels or Gibbs blocks) can add a
# ProduceLogLikelihoodAccumulator to a plain one, which routes through the producing
# `acclogp` -- so this `+` may itself produce.
function Libtask.might_produce(
    ::Type{
        <:Tuple{
            typeof(Base.:+),
            ProduceLogLikelihoodAccumulator,
            DynamicPPL.LogLikelihoodAccumulator,
        },
    },
)
    return true
end
Libtask.@might_produce(DynamicPPL.accumulate_observe!!)
Libtask.@might_produce(DynamicPPL.tilde_observe!!)
Libtask.@might_produce(DynamicPPL.tilde_assume!!)  # GibbsContext turns assumes into observes
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
struct Multinomial <: AbstractResampler end
function resample_indices(rng::AbstractRNG, ::Multinomial, weights, n::Integer)
    return rand(rng, Distributions.Categorical(weights), n)
end

"Stratified resampling: one independent uniform per stratum of width `1/n`."
struct Stratified <: AbstractResampler end
function resample_indices(rng::AbstractRNG, ::Stratified, weights, n::Integer)
    v = n * weights[1]
    indices = Vector{Int}(undef, n)
    s = 1
    for k in 1:n
        u = oftype(v, (k - 1) + rand(rng))
        while v < u
            s += 1
            v += n * weights[s]
        end
        indices[k] = s
    end
    return indices
end

"Systematic resampling: one shared uniform placed on a regular grid of `n` points."
struct Systematic <: AbstractResampler end
function resample_indices(rng::AbstractRNG, ::Systematic, weights, n::Integer)
    v = n * weights[1]
    u = oftype(v, rand(rng))
    indices = Vector{Int}(undef, n)
    s = 1
    for k in 1:n
        while v < u
            s += 1
            v += n * weights[s]
        end
        indices[k] = s
        u += one(u)
    end
    return indices
end

"""
    ESSResampler(threshold, scheme = Stratified())

Resample with `scheme`, but only when the effective sample size drops below
`threshold * nparticles`. This is the default for [`SMC`](@ref) and [`PG`](@ref).
"""
struct ESSResampler{R<:AbstractResampler} <: AbstractResampler
    threshold::Float64
    scheme::R
end
ESSResampler(threshold::Real) = ESSResampler(Float64(threshold), Stratified())

function should_resample(resampler::ESSResampler, weights)
    ess = inv(sum(abs2, weights))
    return ess ≤ resampler.threshold * length(weights)
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
logevidence(particles) = logsumexp(logweights(particles))

# Advance every particle by one observation; return `true` once all have finished. A model
# whose number of observations varies across executions leaves particles out of step.
function reweight!(particles, conditional::Bool)
    n = length(particles)
    n_done = 0
    for (i, p) in enumerate(particles)
        score = advance!(p, conditional && i == n)
        if score === nothing
            n_done += 1
        else
            p.logweight += score
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
            child.logweight = 0.0
            particles[slot] = child
        end
        conditional && (particles[n].logweight = 0.0)  # reference retained, weight reset
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

# Run a full particle sweep in place, returning the log-evidence estimate.
function sweep!(rng::AbstractRNG, particles, resampler; conditional::Bool=false)
    logZ = 0.0
    while true
        resample_propagate!(rng, particles, resampler, conditional)
        logZ0 = logevidence(particles)
        done = reweight!(particles, conditional)
        # Each observation contributes the log-ratio of total weight it adds; summed over the
        # sweep these telescope into an estimate of the model's log-evidence log p(y).
        logZ += logevidence(particles) - logZ0
        done && break
    end
    return logZ
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
end

"""
    SMC([resampler = ESSResampler(0.5)])
    SMC([scheme = Stratified(), ]threshold)

Sequential Monte Carlo sampler. By default stratified resampling is triggered whenever the
effective sample size drops below half the number of particles.

The resampling scheme types (`Stratified`, `Systematic`, `Multinomial`, `ESSResampler`) are
not exported; refer to them as e.g. `Turing.Inference.Systematic`.
"""
SMC() = SMC(ESSResampler(0.5))
SMC(threshold::Real) = SMC(ESSResampler(threshold))
function SMC(scheme::AbstractResampler, threshold::Real)
    return SMC(ESSResampler(Float64(threshold), scheme))
end

struct SMCState{P,W}
    particles::P
    weights::W
    index::Int
    logevidence::Float64
end

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    sampler::SMC,
    N::Integer;
    check_model=true,
    chain_type=DEFAULT_CHAIN_TYPE,
    initial_params=Turing.Inference.init_strategy(sampler),
    progress=PROGRESS[],
    discard_initial=0,
    thinning=1,
    verbose=false,
    kwargs...,
)
    check_model && Turing._check_model(model, sampler)
    error_if_threadsafe_eval(model)
    # SMC is not a Markov chain, so these AbstractMCMC knobs do not apply. Consume them here
    # rather than forwarding them to `mcmcsample` (which would `BoundsError`, see #1811).
    if discard_initial > 0 || thinning > 1
        @warn "SMC does not support `discard_initial` or `thinning`; they are ignored."
    end
    chain = AbstractMCMC.mcmcsample(
        rng,
        model,
        sampler,
        N;
        chain_type,
        initial_params,
        progress,
        nparticles=N,
        kwargs...,
    )
    post_sample_hook(chain, sampler; verbose)
    return chain
end

# The whole sweep runs on the first step; later steps read off the population one particle at
# a time (SMC returns a weighted sample, not a chain).
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    sampler::SMC;
    nparticles::Int,
    discard_sample=false,
    kwargs...,
)
    error_if_threadsafe_eval(model)
    particles = [Particle(model, particle_varinfo(), TracedRNG(rng)) for _ in 1:nparticles]
    logZ = sweep!(rng, particles, sampler.resampler)
    weights = normalized_weights(particles)

    stats = (; weight=weights[1], logevidence=logZ)
    transition =
        discard_sample ? nothing : DynamicPPL.ParamsWithStats(particles[1].varinfo, stats)
    return transition, SMCState(particles, weights, 2, logZ)
end

function AbstractMCMC.step(
    ::AbstractRNG,
    ::DynamicPPL.Model,
    ::SMC,
    state::SMCState;
    discard_sample=false,
    kwargs...,
)
    i = state.index
    stats = (; weight=state.weights[i], logevidence=state.logevidence)
    transition = if discard_sample
        nothing
    else
        DynamicPPL.ParamsWithStats(deepcopy(state.particles[i].varinfo), stats)
    end
    return transition, SMCState(state.particles, state.weights, i + 1, state.logevidence)
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
end

"""
    PG(n, [resampler = ESSResampler(0.5)])
    PG(n, [scheme = Stratified(), ]threshold)

Particle Gibbs sampler with `n` particles. By default stratified resampling is triggered
whenever the effective sample size drops below half the number of particles.
"""
PG(n::Int) = PG(n, ESSResampler(0.5))
PG(n::Int, threshold::Real) = PG(n, ESSResampler(threshold))
function PG(n::Int, scheme::AbstractResampler, threshold::Real)
    return PG(n, ESSResampler(Float64(threshold), scheme))
end

"Conditional SMC, an alias for [`PG`](@ref)."
const CSMC = PG

struct PGState{V<:DynamicPPL.AbstractVarInfo,R<:TracedRNG}
    varinfo::V
    rng::R
end

# First iteration: an ordinary (unconditional) particle sweep.
function AbstractMCMC.step(
    rng::AbstractRNG, model::DynamicPPL.Model, sampler::PG; discard_sample=false, kwargs...
)
    error_if_threadsafe_eval(model)
    particles = [
        Particle(model, particle_varinfo(), TracedRNG(rng)) for _ in 1:(sampler.nparticles)
    ]
    logZ = sweep!(rng, particles, sampler.resampler)
    return pg_transition_and_state(rng, particles, logZ, discard_sample)
end

# Subsequent iterations: conditional SMC given the retained trajectory, which the reference
# particle regenerates by replaying `state.rng` from the first step.
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
    reference = Particle(model, particle_varinfo(), rewind!(deepcopy(state.rng)))
    particles = map(1:n) do i
        i < n ? Particle(model, particle_varinfo(), TracedRNG(rng)) : reference
    end
    logZ = sweep!(rng, particles, sampler.resampler; conditional=true)
    return pg_transition_and_state(rng, particles, logZ, discard_sample)
end

function pg_transition_and_state(rng, particles, logZ, discard_sample)
    retained = particles[rand(
        rng, Distributions.Categorical(normalized_weights(particles))
    )]
    transition = if discard_sample
        nothing
    else
        DynamicPPL.ParamsWithStats(deepcopy(retained.varinfo), (; logevidence=logZ))
    end
    return transition, PGState(retained.varinfo, retained.rng)
end

#
# Gibbs interface
#

gibbs_get_raw_values(state::PGState) = DynamicPPL.get_raw_values(state.varinfo)

function gibbs_update_state!!(
    ::PG, state::PGState, model::DynamicPPL.Model, global_vals::DynamicPPL.VarNamedTuple
)
    init = DynamicPPL.InitFromParams(global_vals, nothing)
    new_vi = last(DynamicPPL.init!!(model, state.varinfo, init, DynamicPPL.UnlinkAll()))
    return PGState(new_vi, state.rng)
end
