####
#### Combining DynamicPPL and Libtask.
####
using StatsFuns: softmax, logsumexp

include("traced_rng.jl")

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
    TracedModel(rng, model)

A container for stepping through a DynamicPPL model using Libtask to yield the log score at
every iteration of a Sequential Monte Carlo sampler.

Typically traced models internally `Libtask.consume` through a sampler, but iteration
utilities like Libtask extend to this structure for manually stepping through a model.
"""
mutable struct TracedModel{T<:TapedTask}
    const task::T
    varinfo::DynamicPPL.AbstractVarInfo
end

# NOTE: instead of storing `vi` in the taped task, we can exploit task local storage because
# updating the varinfo happens within a single call stack of `Libtask.consume`
function construct_task(
    rng::AbstractRNG, model::DynamicPPL.Model, vi::DynamicPPL.AbstractVarInfo
)
    inner_model = DynamicPPL.setleafcontext(model, SMCContext())
    args, kwargs = DynamicPPL.make_evaluate_args_and_kwargs(inner_model, vi)
    return TapedTask(rng, inner_model.f, args...; kwargs...)
end

# overload consume to store the local varinfo
function Libtask.consume(trace::TracedModel)
    score = Libtask.consume(trace.task)
    set_varinfo!(trace)
    return score
end

# apply the same iteration utilities as a TapedTask
function Base.iterate(trace::TracedModel, ::Nothing=nothing)
    v = Libtask.consume(trace)
    return v === nothing ? nothing : (v, nothing)
end

Base.IteratorSize(::Type{<:TracedModel}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:TracedModel}) = Base.EltypeUnknown()

# these will be useful when constructing new traces from proposed values
get_model(trace::TracedModel) = fetch_type(DynamicPPL.Model, trace.task.fargs)
get_varinfo(trace::TracedModel) = trace.varinfo
get_rng(trace::TracedModel) = trace.task.taped_globals

@generated function fetch_type(::Type{T}, tup::Tuple) where {T}
    indices = Int[]
    for i in 1:fieldcount(tup)
        (fieldtype(tup, i) <: T) && push!(indices, i)
    end

    if isempty(indices)
        return :(error("No element of type ", T, " found in tuple"))
    elseif length(indices) > 1
        return :(error(
            "Multiple elements of type ", T, " found in tuple at positions ", $indices
        ))
    else
        return :(tup[$(indices[1])])
    end
end

# this prevents the task from resetting the varinfo
function set_reference(trace::TracedModel)
    rng = get_rng(trace)
    Random123.set_counter!(rng, 1)
    return TracedModel(rng, get_model(trace))
end

TracedModel(rng::AbstractRNG, model::DynamicPPL.Model) = TracedModel(TracedRNG(rng), model)

function TracedModel(rng::TracedRNG, model::DynamicPPL.Model)
    accs = DynamicPPL.OnlyAccsVarInfo()
    accs = DynamicPPL.setacc!!(accs, ProduceLogLikelihoodAccumulator())
    accs = DynamicPPL.setacc!!(accs, DynamicPPL.RawValueAccumulator(true))
    return TracedModel(construct_task(rng, model, accs), accs)
end

# update trace varinfo with what has been sampled to this point
function set_varinfo!(trace::TracedModel)
    storage = task_local_storage()
    if haskey(storage, :varinfo)
        trace.varinfo = pop!(storage, :varinfo)
    end
end

struct ProduceLogLikelihoodAccumulator{T<:Real} <: DynamicPPL.LogProbAccumulator{T}
    logp::T
end

DynamicPPL.accumulator_name(::Type{<:ProduceLogLikelihoodAccumulator}) = :LogLikelihood
DynamicPPL.logp(acc::ProduceLogLikelihoodAccumulator) = acc.logp

# this is the only difference between LogLikelihoodAccumulator
function DynamicPPL.acclogp(acc::ProduceLogLikelihoodAccumulator, val)
    task_local_storage(:logscore, val)
    newacc = ProduceLogLikelihoodAccumulator(DynamicPPL.logp(acc) + val)
    return newacc
end

function DynamicPPL.accumulate_assume!!(
    acc::ProduceLogLikelihoodAccumulator, val, tval, logjac, vn, dist, template
)
    return acc
end

function DynamicPPL.accumulate_observe!!(
    acc::ProduceLogLikelihoodAccumulator, dist, val, vn, template
)
    return DynamicPPL.acclogp(acc, Distributions.loglikelihood(dist, val))
end

# Relevant call chains:
# tilde_observe!! -> accumulate_observe!! -> acclogp -> produce
Libtask.@might_produce(DynamicPPL.tilde_observe!!)
Libtask.@might_produce(DynamicPPL.accumulate_observe!!)
Libtask.@might_produce(DynamicPPL.acclogp)

# tilde_assume!! in Gibbs -> tilde_observe!! -> ...
Libtask.@might_produce(DynamicPPL.tilde_assume!!)

# @addlogprob!(::Number) -> accloglikelihood!! -> map_accumulator!! -> acclogp -> produce
Libtask.@might_produce(DynamicPPL.accloglikelihood!!)
Libtask.@might_produce(DynamicPPL.map_accumulator!!)

# @addlogprob!(::NamedTuple) -> acclogp!! -> accloglikelihood!! -> ...
Libtask.@might_produce(DynamicPPL.acclogp!!)

# This handles all models and submodel evaluator functions (including those with keyword
# arguments). The key to this is realising that all model evaluator functions take
# DynamicPPL.Model as an argument, so we can just check for that. See
# https://github.com/TuringLang/Libtask.jl/issues/217.
Libtask.might_produce_if_sig_contains(::Type{<:DynamicPPL.Model}) = true

struct SMCContext <: DynamicPPL.AbstractContext end

DynamicPPL.get_param_eltype(::DynamicPPL.AbstractVarInfo, ::SMCContext) = Any

function init_context(rng::AbstractRNG, vi::AbstractVarInfo, vn::VarName)
    values = DynamicPPL.get_raw_values(vi)
    ctx = if ~haskey(values, vn)
        DynamicPPL.InitFromPrior()
    else
        DynamicPPL.InitFromParams(values, nothing)
    end
    return DynamicPPL.InitContext(rng, ctx, DynamicPPL.UnlinkAll())
end

function DynamicPPL.tilde_assume!!(
    ::SMCContext, dist::Distribution, vn::VarName, template::Any, vi::AbstractVarInfo
)
    rng = Libtask.get_taped_globals(AbstractRNG)
    dispatch_ctx = init_context(rng, vi, vn)
    val, vi = DynamicPPL.tilde_assume!!(dispatch_ctx, dist, vn, template, vi)
    task_local_storage(:varinfo, vi)
    return val, vi
end

function DynamicPPL.tilde_observe!!(
    ::SMCContext,
    dist::Distribution,
    val,
    vn::Union{VarName,Nothing},
    template,
    vi::AbstractVarInfo,
)
    val, vi = DynamicPPL.tilde_observe!!(DefaultContext(), dist, val, vn, template, vi)
    task_local_storage(:varinfo, vi)
    Libtask.produce(pop!(task_local_storage(), :logscore))
    return val, vi
end

"""
    @producelogprob!(ex)

Same as `@addlogprob!`, but intended for use with SMC samplers to ensure Libtask can produce
the loglikelihood when iterating through the model
"""
macro producelogprob!(ex)
    # this ensures @addlogprob! accounts for task local storage ala Libtask
    return quote
        val = $(esc(ex))
        vi = $(esc(:(__varinfo__)))
        if val isa Number
            if DynamicPPL.hasacc(vi, Val(:LogLikelihood))
                $(esc(:(__varinfo__))) = DynamicPPL.accloglikelihood!!(
                    $(esc(:(__varinfo__))), val
                )

                # grab the log score and remove it from storage
                loglike = pop!(task_local_storage(), :logscore)
                if !isnothing(loglike)
                    Libtask.produce(loglike)
                end
            end
        elseif val isa NamedTuple
            $(esc(:(__varinfo__))) = DynamicPPL.acclogp!!(
                $(esc(:(__varinfo__))), val; ignore_missing_accumulator=true
            )
        else
            error("logp must be a Number or a NamedTuple.")
        end
    end
end

####
#### Resampling Schemes.
####

abstract type AbstractResampler end

should_resample(::AbstractVector, ::AbstractResampler) = true

function sample_ancestors(
    rng::AbstractRNG, scheme::AbstractResampler, weights::AbstractVector
)
    return sample_ancestors(rng, scheme, weights, length(weights))
end

struct Multinomial <: AbstractResampler end

function sample_ancestors(
    rng::AbstractRNG, ::Multinomial, weights::AbstractVector{<:Real}, N::Integer
)
    return rand(rng, Categorical(weights), N)
end

struct Systematic <: AbstractResampler end

function sample_ancestors(
    rng::AbstractRNG, ::Systematic, weights::AbstractVector{WT}, N::Integer
) where {WT<:Real}
    # pre-calculations
    vs = cumsum(weights)
    vs *= N

    u0 = rand(rng, WT)

    # initialize sampling algorithm
    a = Vector{Int64}(undef, N)
    idx = 1

    @inbounds for i in 1:N
        u = u0 + (i - 1)
        while vs[idx] <= u
            idx += 1
        end
        a[i] = idx
    end

    return a
end

struct ESSResampler{T<:Real,R<:AbstractResampler} <: AbstractResampler
    threshold::T
    scheme::R
end

ESSResampler(threshold::Real) = ESSResampler(threshold, Systematic())

function should_resample(weights::AbstractVector, resampler::ESSResampler)
    ess = inv(sum(abs2, weights))
    return ess ≤ resampler.threshold * length(weights)
end

function sample_ancestors(
    rng::AbstractRNG, resampler::ESSResampler, weights::AbstractVector, N::Integer
)
    return sample_ancestors(rng, resampler.scheme, weights, N)
end

####
#### Particle Containers.
####

"""
    Particle
"""
mutable struct Particle{PT,WT<:Real}
    const value::PT
    logw::WT
end

Particle(value) = Particle(value, zero(DynamicPPL.LogProbType))

function reset_weight!(particle::Particle{PT,WT}) where {PT,WT}
    particle.logw = zero(WT)
    return nothing
end

"""
    ParticleContainer

A custom array object to handle getting and setting of particle values as well as their log-
weights. This allows a plethora of in-place operations and intuitive handling throughout the
SMC process.
"""
const ParticleContainer{PT,WT} = Vector{Particle{PT,WT}}
ParticleContainer(values::AbstractVector) = Particle.(values)

# this is quite overkill, so I might ditch it in future versions
@inline Base.getproperty(pc::ParticleContainer, s::Symbol) = _getproperty(pc, Val(s))
@inline _getproperty(pc::ParticleContainer, ::Val{:values}) = @. getproperty(pc, :value)
@inline _getproperty(pc::ParticleContainer, ::Val{:log_weights}) = @. getproperty(pc, :logw)
@inline _getproperty(pc::ParticleContainer, ::Val{S}) where {S} = getfield(pc, S)

function StatsBase.weights(particles::ParticleContainer)
    return weights(softmax(particles.log_weights))
end

function StatsBase.sample(rng::AbstractRNG, particles::ParticleContainer)
    return sample(rng, particles.values, StatsBase.weights(particles))
end

function resample!(
    rng::AbstractRNG,
    resampler::AbstractResampler,
    particles::ParticleContainer,
    weights::StatsBase.Weights,
)
    idx = sample_ancestors(rng, resampler, weights)
    @. particles = Particle($split!(rng, particles.values[idx]))
end

logevidence(particles::ParticleContainer) = logsumexp(particles.log_weights)

# reseed new child particles according to the sampler RNG
function split!(rng::AbstractRNG, particles::Vector{<:TracedModel})
    children = deepcopy.(particles)
    seeds = rand(rng, Random.Sampler(rng, UInt64), length(particles))
    @. Random.seed!(get_rng(children), seeds)
    return children
end

# wraps consume such that RNG can be traced upon resampling
function advance!(particle::Particle{<:TracedModel}, isref::Bool=false)
    rng = get_rng(particle.value)
    isref ? load_state!(rng) : save_state!(rng)
    inc_counter!(rng)
    return consume(particle.value)
end

# ensures state loading is consistent with the traced RNG
function update_key!(particle::Particle)
    rng = get_rng(particle.value)
    k = split(state(rng.rng))
    Random.seed!(rng, k[1])
end

update_keys!(particles::ParticleContainer) = map(update_key!, particles)

"""
    ReferencedContainer

An object which associates a given reference trajectory with a given particle container. One
can access `container.values` and `container.log_weights` as before, where the final element
of the vector is the reference trajectory.
"""
struct ReferencedContainer{PT,WT}
    particles::ParticleContainer{PT,WT}
    reference::Particle{PT,WT}
end

Base.length(pc::ReferencedContainer) = length(pc.particles) + 1
Base.keys(pc::ReferencedContainer) = LinearIndices(pc.values)

function Base.getindex(pc::ReferencedContainer, i)
    i == length(pc) && return pc.reference
    return pc.particles[i]
end

Base.getproperty(pc::ReferencedContainer, s::Symbol) = _getproperty(pc, Val(s))

function _getproperty(pc::ReferencedContainer{PT}, ::Val{:values}) where {PT}
    values = Vector{PT}(undef, length(pc.particles) + 1)
    values[1:(end - 1)] = @views(pc.particles.values)
    values[end] = pc.reference.value
    return values
end

function _getproperty(pc::ReferencedContainer{PT,WT}, ::Val{:log_weights}) where {PT,WT}
    log_weights = Vector{WT}(undef, length(pc.particles) + 1)
    log_weights[1:(end - 1)] = @views(pc.particles.log_weights)
    log_weights[end] = pc.reference.logw
    return log_weights
end

_getproperty(pc::ReferencedContainer, ::Val{S}) where {S} = getfield(pc, S)

function StatsBase.weights(particles::ReferencedContainer)
    return weights(softmax(particles.log_weights))
end

function StatsBase.sample(rng::AbstractRNG, particles::ReferencedContainer)
    return sample(rng, particles.values, weights(particles))
end

function resample!(
    rng::AbstractRNG,
    resampler::AbstractResampler,
    ref::ReferencedContainer,
    weights::StatsBase.Weights,
)
    idx = sample_ancestors(rng, resampler, weights, length(ref.particles))
    @. ref.particles = Particle($split!(rng, ref.values[idx]))
    reset_weight!(ref.reference)
end

logevidence(particles::ReferencedContainer) = logsumexp(particles.log_weights)

# this preserves the RNG trace for references
update_keys!(ref::ReferencedContainer) = update_keys!(ref.particles)

####
#### Generic Sequential Monte Carlo sampler.
####

"""
    SMC{RT}

A basic Sequential Monte Carlo sampler, resampling according to scheme RT. By default this
is set to always resample using a [`Systematic`](@ref) resampler.

# Examples
```julia
# samples 128 particles each iteration, resampling when ESS drops below 50%
chain = sample(model, SMC(0.5), 10_000)
```

See [`ParticleGibbs`](@ref) for use within Markov Chain Monte Carlo.
"""
struct SMC{RT} <: AbstractSampler
    resampler::RT
end

# by default always resample unless the user provides a threshold
SMC(threshold::Real) = SMC(ESSResampler(threshold))
SMC(rs::AbstractResampler, threshold::Real) = SMC(ESSResampler(threshold, rs))
SMC() = SMC(Systematic())

function initialize(rng::AbstractRNG, model::DynamicPPL.Model, ::SMC, N::Integer)
    return ParticleContainer([TracedModel(rng, model) for _ in 1:N]), false
end

function initialize(
    rng::AbstractRNG, model::DynamicPPL.Model, sampler::SMC, N::Integer, ref
)
    particles, is_done = initialize(rng, model, sampler, N)
    if !isnothing(ref)
        # handle the case where N is 1, thus we only have a reference
        particles = ReferencedContainer(particles[1:(N - 1)], Particle(set_reference(ref)))
    end
    return particles, is_done
end

increment_weight!(particle::Particle, score::Nothing) = true
increment_weight!(particle::Particle, score::Real) = (particle.logw += score; return false)

check_ref(pc::ParticleContainer, N::Integer) = false
check_ref(pc::ReferencedContainer, N::Integer) = (length(pc) == N)

function reweight!(particles, parallel::AbstractMCMC.AbstractMCMCEnsemble)
    logZ0 = logevidence(particles)
    num_done = ensemble_reweight!(particles, parallel)
    return handle_completion(num_done), (logevidence(particles) - logZ0)
end

function ensemble_reweight!(particles, ::AbstractMCMC.MCMCSerial)
    return map(eachindex(particles)) do i
        is_ref = check_ref(particles, i)
        score = advance!(particles[i], is_ref)
        increment_weight!(particles[i], score)
    end
end

function ensemble_reweight!(particles, ::AbstractMCMC.MCMCThreads)
    num_done = Vector{Bool}(undef, length(particles))
    Threads.@threads for i in eachindex(particles)
        is_ref = check_ref(particles, i)
        score = advance!(particles[i], is_ref)
        num_done[i] = increment_weight!(particles[i], score)
    end
    return num_done
end

function ensemble_reweight!(particles, ::AbstractMCMC.MCMCDistributed)
    return pmap(eachindex(particles)) do i
        is_ref = check_ref(particles, i)
        score = advance!(particles[i], is_ref)
        increment_weight!(particles[i], score)
    end
end

# ensure consistency in observation steps per particle
function handle_completion(num_done)
    if all(num_done)
        return true
    elseif any(num_done)
        error("mis-aligned execution traces: $(sum(num_done)) / $(length(num_done))")
    else
        return false
    end
end

# resampling involves ancestor sampling and RNG splitting done two ways
function maybe_resample!(rng::AbstractRNG, particles, resampler::AbstractResampler)
    weights = StatsBase.weights(particles)
    rs_flag = should_resample(weights, resampler)
    if rs_flag
        # RNG traces generate new seeds based on the sampler RNG
        resample!(rng, resampler, particles, weights)
    else
        # RNG traces generate new seeds based on their own keys
        update_keys!(particles)
    end
    return rs_flag
end

# TODO: add custom logging like is done for AbstractMCMC
function smcsample(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    sampler::SMC,
    ensemble::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer;
    ref=nothing,
    kwargs...,
)
    particles, is_done = initialize(rng, model, sampler, N, ref)
    logZ = zero(DynamicPPL.LogProbType)
    while !is_done
        # resample move (although rejuvenation is still unimplemented)
        maybe_resample!(rng, particles, sampler.resampler)

        # we capture the log evidence here for when resampling always occurs
        is_done, logmarginal = reweight!(particles, ensemble)
        logZ += logmarginal
    end
    return particles, logZ
end

# We overload AbstractMCMC.sample because SMC does not yield a Markov chain. Since stepping
# through a sampler is based on the underlying model, we possess a different structure which
# implies two things: (1) parallelizers are quite a bit different and (2) dependence on the
# output for AbstractMCMC.step
function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    sampler::SMC,
    N::Integer;
    ensemble::AbstractMCMC.AbstractMCMCEnsemble=AbstractMCMC.MCMCSerial(),
    check_model=true,
    chain_type::Any=DEFAULT_CHAIN_TYPE,
    verbose=false,
    kwargs...,
)
    check_model && Turing._check_model(model, sampler)
    error_if_threadsafe_eval(model)

    # perform a particle sweep
    particles, logZ = smcsample(rng, model, sampler, ensemble, N; kwargs...)
    stats = (; logevidence=logZ)

    # convert to readable format and bundle samples
    sample = map(x -> DynamicPPL.ParamsWithStats(get_varinfo(x.value), stats), particles)
    chain = AbstractMCMC.bundle_samples(sample, model, sampler, particles, chain_type)
    post_sample_hook(chain, sampler; verbose)
    return chain
end

####
#### Particle Gibbs and Conditional SMC.
####

"""
    ParticleGibbs

An MCMC sampler which wraps a conditional Sequential Monte Carlo step at every iteration of
the sampler for `N` iterations along a Markov chain.

# Examples
```julia
# samples 10,000 particles each iteration, resampling when ESS drops below 50%
chain = sample(model, SMC(0.5), 10_000)
```

See [`SMC`](@ref) for details on the Sequential Monte Carlo kernel.
"""
struct ParticleGibbs{T<:SMC} <: AbstractMCMC.AbstractSampler
    kernel::T
    N::Int
end

# define aliases for conditional sequential monte carlo / particle gibbs
const PG = ParticleGibbs
const CSMC = PG

# convenience constructors
PG(N::Int, threshold::Real) = PG(SMC(threshold), N)
PG(N::Int, rs::AbstractResampler) = PG(SMC(rs), N)
PG(N::Int, rs::AbstractResampler, threshold::Real) = PG(SMC(rs, threshold), N)
PG(N::Int) = PG(N, 0.5)

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    sampler::ParticleGibbs;
    discard_sample=false,
    kwargs...,
)
    # perform a run of the mill SMC sampler for the first run
    error_if_threadsafe_eval(model)
    particles, logZ = smcsample(
        rng, model, sampler.kernel, AbstractMCMC.MCMCSerial(), sampler.N
    )
    state = StatsBase.sample(rng, particles)
    sample = if discard_sample
        nothing
    else
        stats = (; logevidence=logZ)
        DynamicPPL.ParamsWithStats(get_varinfo(state), stats)
    end
    return sample, state
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    sampler::ParticleGibbs,
    state::TracedModel;
    discard_sample=false,
    kwargs...,
)
    # perform conditional SMC given a reference path from now on
    ref = deepcopy(state)
    particles, logZ = smcsample(
        rng, model, sampler.kernel, AbstractMCMC.MCMCSerial(), sampler.N; ref
    )

    # choose a likely state as reference for subsequent runs
    new_ref = StatsBase.sample(rng, particles)
    sample = if discard_sample
        nothing
    else
        stats = (; logevidence=logZ)
        DynamicPPL.ParamsWithStats(get_varinfo(new_ref), stats)
    end
    return sample, new_ref
end

####
#### Gibbs interface
####

function gibbs_get_raw_values(state::TracedModel)
    vi = get_varinfo(state)
    return DynamicPPL.get_raw_values(vi)
end

function gibbs_update_state!!(
    ::ParticleGibbs,
    state::TracedModel,
    model::DynamicPPL.Model,
    global_vals::DynamicPPL.VarNamedTuple,
)
    init_strat = DynamicPPL.InitFromParams(global_vals, nothing)
    accs = last(
        DynamicPPL.init!!(model, get_varinfo(state), init_strat, DynamicPPL.UnlinkAll())
    )
    return TracedModel(construct_task(get_rng(state), model, accs), accs)
end
