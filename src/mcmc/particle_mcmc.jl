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
# The reference reproduces the retained trajectory by *reusing its values*: the sampler state
# carries those values, and the reference re-runs the model with `InitFromParams`, so it stays the
# retained trajectory even when the model is re-conditioned between Gibbs sweeps (e.g. a
# state-space transition prior that depends on a parameter owned by another Gibbs component). A
# particle forked from the reference forgets the remaining values (in `reseed!`), so branching
# samples fresh with no per-particle flag.
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
    Particle(model, rng, retained::Particle)

A single particle: a suspended `model` execution together with its `varinfo`, its own `rng`, and an
accumulated `logweight`. It also serves directly as the particle Gibbs sampler state (there is no
separate state struct).

Without `retained` the particle draws from the prior. Given the previous sweep's retained particle it
becomes a conditional-SMC reference pinned to that trajectory, erroring if its execution reaches an
address the retained trajectory lacks or finishes without reaching one it has. Taking the whole
particle, rather than its values and addresses separately, is what makes a half-specified reference
unrepresentable; only those two pieces are kept, so the retained particle is not held alive.
"""
mutable struct Particle{RT<:AbstractRNG,WT<:Real}
    # Abstract on purpose: the VarInfo type can change during PG-inside-Gibbs. Accesses go
    # through Libtask's (already type-unstable) taped globals, so this costs nothing extra.
    varinfo::DynamicPPL.AbstractVarInfo
    rng::RT
    # `logweight` tracks whatever `DynamicPPL.LogProbType` is, so weights follow suit if it
    # is ever changed.
    logweight::WT
    # `nothing` unless this particle is a CSMC reference; one field rather than two so the pair
    # cannot get out of step, and so `isreference` has a single thing to test.
    #
    # `values` is the retained trajectory, which the reference reproduces by reusing it
    # (`InitFromParams` in `tilde_assume!!`). Reusing the *value* rather than replaying the RNG draw
    # is what keeps the reference on that trajectory when Gibbs re-conditions the model: a draw is
    # x = g(u; θ) in the RNG output u and the distribution parameters θ, so replaying u after θ → θ'
    # yields g(u; θ') ≠ x -- e.g. x ~ Normal(μ, 1) is μ + Φ⁻¹(u), which shifts by μ' − μ.
    #
    # `varnames` is the set of addresses the retained trajectory assumed, and cannot be recovered
    # from `values`: a slice assume such as `x[1:2] ~ MvNormal(...)` is stored under the keys `x[1]`,
    # `x[2]` but assumed under the single address `x[1:2]`, so comparing against those keys would
    # report a spurious trace change. Without it an address the retained trajectory never had would
    # silently draw from the prior, corrupting the reference.
    reference::Union{
        Nothing,
        @NamedTuple{values::DynamicPPL.VarNamedTuple, varnames::Set{DynamicPPL.VarName}}
    }
    # Addresses assumed by this execution, in the same form as `reference.varnames`. Survives
    # forking, so a particle that becomes the retained state hands the complete set to the next
    # reference; a reference must finish having assumed exactly that set.
    assumed_varnames::Set{DynamicPPL.VarName}
    task::Libtask.TapedTask
    # `task` is filled in once the particle exists, because the task must capture the
    # particle as its taped globals (a back-reference). This has to be an inner constructor
    # for that reason: `task` is left undefined here and set immediately after.
    function Particle(
        vi::DynamicPPL.AbstractVarInfo, rng::RT, retained::Union{Nothing,Particle}=nothing
    ) where {RT<:AbstractRNG}
        w = zero(DynamicPPL.LogProbType)
        reference = if retained === nothing
            nothing
        else
            (;
                values=DynamicPPL.get_raw_values(retained.varinfo),
                varnames=copy(retained.assumed_varnames),
            )
        end
        return new{RT,typeof(w)}(vi, rng, w, reference, Set{DynamicPPL.VarName}())
    end
end

function Particle(
    model::DynamicPPL.Model, rng::AbstractRNG, retained::Union{Nothing,Particle}=nothing
)
    model = DynamicPPL.setleafcontext(model, SMCContext())
    varinfo = particle_varinfo()
    args, kwargs = DynamicPPL.make_evaluate_args_and_kwargs(model, varinfo)
    particle = Particle(varinfo, rng, retained)
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
    # `tilde_assume!!` already rejects any address outside the retained set, so once the
    # reference has run to completion the only discrepancy left to catch is a retained address
    # it never visited (e.g. a branch that stopped being taken after re-conditioning).
    reference = particle.reference
    if score === nothing && reference !== nothing
        dropped = setdiff(reference.varnames, particle.assumed_varnames)
        isempty(dropped) || error(
            "the reference execution trace changed while replaying retained values " *
            "(retained addresses never reached: $(collect(dropped)))",
        )
    end
    return score
end

function DynamicPPL.tilde_assume!!(
    ::SMCContext, dist::Distribution, vn::VarName, template, ::DynamicPPL.AbstractVarInfo
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
        vn in reference.varnames || error(
            "the reference execution trace changed while replaying retained values " *
            "(new address: $vn)",
        )
        DynamicPPL.InitFromParams(reference.values, nothing)
    end
    ctx = DynamicPPL.InitContext(particle.rng, strategy, DynamicPPL.UnlinkAll())
    x, vi = DynamicPPL.tilde_assume!!(ctx, dist, vn, template, particle.varinfo)
    particle.varinfo = vi
    push!(particle.assumed_varnames, vn)
    return x, vi
end

# Routes the observe through the particle's varinfo and stores the result back. The weight is not
# emitted here -- `ProduceLogLikelihoodAccumulator` produces it as it accumulates, below.
function DynamicPPL.tilde_observe!!(
    ::SMCContext,
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

# The produce lives here, in the single place the likelihood is accumulated, so `val` *is* the
# increment: "the produced weight equals the accumulator's increment" becomes structural rather than
# an invariant to keep two call sites in step with. Both routes reach the accumulator through exactly
# one `acclogp` -- an `observe` via `accumulate_observe!!`, and `@addlogprob!` via
# `accloglikelihood!!` -> `map_accumulator!!` -- so each emits exactly one weight, which is what
# gets `@addlogprob!` terms into the weight as well as the accumulator (issue #1996).
#
# Accumulator merging cannot fire a spurious produce: `combine` adds the two `logp`s directly rather
# than going through `acclogp`, so a submodel's varinfo folding into its parent's stays silent.
#
# `produce` suspends the task before the caller assigns the updated varinfo back onto the particle, so
# a *suspended* particle's accumulated total lags this term. Nothing reads it in that state -- the
# sweep reweights from `logweight`, and every varinfo read (`pg_transition_and_state`,
# `gibbs_get_raw_values`, SMC's bundling) happens once the model has run to completion.
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

# Tell Libtask which calls may contain a `produce`, so it instruments them. The produce itself is in
# `acclogp`; everything else here is marked because it sits on a path that reaches it. Over-
# approximating is safe (a wrongly-marked call is merely instrumented); missing a real one is not.
#
#   observe:      tilde_observe!! -> accumulate_observe!! -> acclogp
#   @addlogprob!: accloglikelihood!! -> map_accumulator!! -> acclogp
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
function reweight!(particles, multithreaded::Bool)
    n = length(particles)
    if multithreaded
        # A shared counter would race, so collect per-particle results and tally afterwards.
        finished = Vector{Bool}(undef, n)
        Threads.@threads for i in 1:n
            finished[i] = advance_particle!(particles[i])
        end
        n_done = count(finished)
    else
        n_done = count(advance_particle!, particles)
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
    n = length(particles)
    conditional = isreference(last(particles))
    weights = normalized_weights(particles)
    if should_resample(resampler, weights)
        # A conditional sweep draws the `n-1` free ancestors independently from the categorical
        # over the weights, whatever scheme `resampler` names -- see the resampling-schemes
        # section for why the named scheme's conditional version is not simply "pin one draw".
        ancestors = if conditional
            resample_indices(rng, MultinomialResampler(), weights, n - 1)
        else
            resample_indices(rng, resampler, weights, n)
        end
        old = copy(particles)
        seen = falses(n)
        for (slot, a) in enumerate(ancestors)
            # Reuse each surviving parent's object for its first offspring; only extra
            # offspring -- and any offspring of the retained reference -- need the costly
            # `deepcopy`. Either way the child is reseeded to continue independently.
            reuse = !seen[a] && !isreference(old[a])
            seen[a] = true
            child = reuse ? reseed!(old[a], rng) : fork(old[a], rng)
            child.logweight = zero(DynamicPPL.LogProbType)
            particles[slot] = child
        end
        # reference retained, weight reset
        conditional && (particles[n].logweight = zero(DynamicPPL.LogProbType))
        return true
    else
        # The reference draws nothing (it reuses retained values), so only the others need a
        # fresh seed for the next step.
        for p in particles
            isreference(p) || refresh!(p.rng)
        end
        return false
    end
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
        resampled = resample_propagate!(rng, particles, resampler)
        resampled && (logZ0 = log(oftype(logZ, length(particles))))
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
    PG(n, [resampler = ESSThresholdResampler(0.5)]; multithreaded = false)
    PG(n, [scheme = StratifiedResampler(), ]threshold; multithreaded = false)

Particle Gibbs sampler with `n` particles. By default resampling is triggered whenever the
effective sample size drops below half the number of particles. The selected scheme applies to the
unconditional first sweep only; conditional sweeps draw their ancestors from the categorical over
the weights, for the reason given in the resampling-schemes section of this file.

Set `multithreaded = true` to evaluate the particles across threads within each sweep; results are
unchanged (start Julia with multiple threads, e.g. `julia -t auto`, for this to have effect).

!!! warning "`log_normalizing_constant` is biased for PG"
    PG chains carry `log_normalizing_constant`, but unlike [`SMC`](@ref)'s it does **not** estimate
    `log p(y)` without bias, so it must not be used for model comparison. A conditional sweep
    retains the reference whatever its weight, and the reference is a draw from the posterior rather
    than from the proposal, so it usually carries far more likelihood than a fresh particle and
    inflates the mean weight at every step. Measured against the exact `p(y)` of a linear Gaussian
    SSM, `E[Ẑ]` overshoots by 80% at `n = 16` and 16% at `n = 64`; the bias decays like `1/n` but
    stays large at practical `n`. Use `SMC` for an unbiased estimate.
"""
PG(n::Int; kwargs...) = PG(n, ESSThresholdResampler(0.5); kwargs...)
PG(n::Int, threshold::Real; kwargs...) = PG(n, ESSThresholdResampler(threshold); kwargs...)
function PG(n::Int, scheme::AbstractResampler, threshold::Real; kwargs...)
    return PG(n, ESSThresholdResampler(threshold, scheme); kwargs...)
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
    particles = [Particle(model, particle_rng(rng)) for _ in 1:(sampler.nparticles)]
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
    # Passing `state` makes this the reference, pinned to the retained trajectory by value (see
    # the `reference` field). Its own generator is never read, since every draw is supplied by value --
    # only its forks' are, and `reseed!` gives those fresh seeds. So it carries the retained
    # generator forward rather than taking a fresh one, which keeps the sweep from drawing anything
    # from `rng` on its behalf; the copy just avoids aliasing `state`.
    reference = Particle(model, deepcopy(state.rng), state)
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
