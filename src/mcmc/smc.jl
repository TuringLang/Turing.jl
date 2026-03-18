####
#### Combining DynamicPPL and Libtask.
####
using StatsFuns: softmax

mutable struct TracedModel{T<:TapedTask}
    const task::T
    varinfo::AbstractVarInfo
end

function construct_task(rng::AbstractRNG, model::Model, vi::AbstractVarInfo)
    inner_rng = Random.seed!(Random123.Philox2x(), rand(rng, Random.Sampler(rng, UInt64)))
    inner_model = DynamicPPL.setleafcontext(model, SMCContext())
    args, kwargs = DynamicPPL.make_evaluate_args_and_kwargs(inner_model, vi)
    return TapedTask(inner_rng, inner_model.f, args...; kwargs...)
end

# overload consume to store the local varinfo
function Libtask.consume(trace::TracedModel)
    score = Libtask.consume(trace.task)
    set_varinfo!(trace, score)
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
get_model(trace::TracedModel) = trace.task.fargs[2]
get_varinfo(trace::TracedModel) = trace.varinfo
get_rng(trace::TracedModel) = trace.task.taped_globals

function TracedModel(rng::AbstractRNG, model::Model)
    vi = DynamicPPL.setacc!!(VarInfo(model), ProduceLogLikelihoodAccumulator())
    vi = DynamicPPL.empty!!(vi)
    return TracedModel(construct_task(rng, model, vi), vi)
end

# if score is nothing, the varinfo is caught up and there's no need to update
set_varinfo!(::TracedModel, ::Nothing) = nothing
set_varinfo!(trace::TracedModel, ::Real) = (trace.varinfo = task_local_storage(:varinfo))

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

# Generic catch-all to handle submodels and kwargs on models, see
# https://github.com/TuringLang/Libtask.jl/issues/217
Libtask.might_produce_if_sig_contains(::Type{<:DynamicPPL.Model}) = true

struct SMCContext <: DynamicPPL.AbstractContext end

function init_context(rng::AbstractRNG, vi::VarInfo, vn::VarName)
    if ~haskey(vi, vn)
        return DynamicPPL.InitContext(
            rng, DynamicPPL.InitFromPrior(), vi.transform_strategy
        )
    else
        return DynamicPPL.DefaultContext()
    end
end

function DynamicPPL.tilde_assume!!(
    ::SMCContext, dist::Distribution, vn::VarName, template::Any, vi::AbstractVarInfo
)
    rng = Libtask.get_taped_globals(AbstractRNG)
    dispatch_ctx = init_context(rng, vi, vn)
    val, vi = DynamicPPL.tilde_assume!!(dispatch_ctx, dist, vn, template, vi)
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
    Libtask.produce(task_local_storage(:logscore))
    return val, vi
end

####
#### Resampling Schemes.
####

abstract type AbstractResampler end

struct AlwaysResample <: AbstractResampler end

function should_resample(::AbstractVector, ::AlwaysResample)
    return true
end

struct ESSResampler{T<:Real} <: AbstractResampler
    threshold::T
end

function should_resample(weights::AbstractVector, resampler::ESSResampler)
    ess = inv(sum(abs2, weights))
    return ess ≤ resampler.threshold * length(weights)
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

Particle(value) = Particle(value, 0.0)

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
    return sample(rng, particles.values, weights(particles))
end

function resample!(
    rng::AbstractRNG, particles::ParticleContainer, weights::StatsBase.Weights
)
    idx = sample_ancestors(rng, weights.values)
    @. particles = Particle($split!(rng, particles.values[idx]))
end

# TODO: optimize this for the love of god
function split!(rng::AbstractRNG, particles::Vector{<:TracedModel})
    children = deepcopy.(particles)
    seeds = rand(rng, Random.Sampler(rng, UInt64), length(particles))
    @. Random.seed!(get_rng(children), seeds)
    return children
end

advance!(particle::Particle{<:TracedModel}) = consume(particle.value)

"""
    ReferencedContainer

An object which associates a given reference trajectory with a given particle container. One
can access `container.values` and `container.log_weights` as before, where the final element
of the vector is the reference trajectory.
"""
struct ReferencedContainer{PT,WT,RT}
    particles::ParticleContainer{PT,WT}
    reference::Particle{RT,WT}
end

Base.length(pc::ReferencedContainer) = length(pc.particles) + 1
Base.keys(pc::ReferencedContainer) = LinearIndices(pc.values)

Base.iterate(pc::ReferencedContainer) = iterate(pc.particles)

function Base.iterate(pc::ReferencedContainer, i)
    i == length(pc) && return (pc.reference, i + 1)
    return iterate(pc.particles, i)
end

function Base.getindex(pc::ReferencedContainer, i)
    i == length(pc) && return pc.reference
    return pc.particles[i]
end

function Base.collect(pc::ReferencedContainer)
    particles = Vector{eltype(pc.particles)}(undef, length(pc))
    particles[1:(end - 1)] = @views(pc.particles)
    particles[end] = pc.reference
    return particles
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

function resample!(rng::AbstractRNG, ref::ReferencedContainer, weights::StatsBase.Weights)
    idx = sample_ancestors(rng, weights.values, length(ref.particles))
    @. ref.particles = Particle($split!(rng, ref.values[idx]))
    return ref
end

####
#### Generic Sequential Monte Carlo sampler.
####

"""
    SMC{RT,KT}

A basic Sequential Monte Carlo sampler, resampling according to scheme RT and rejuvenating
the sample according to the kernel KT.

By default this is set to always resample and never to rejuvenate, as was done in previous
versions of Turing.

See [`ParticleGibbs`](@ref) for use within Markov Chain Monte Carlo.
"""
struct SMC{RT,KT} <: AbstractSampler
    resampler::RT
    kernel::KT
end

SMC(threshold::Real) = SMC(ESSResampler(threshold), nothing)
SMC() = SMC(AlwaysResample(), nothing)

function initialize(rng::AbstractRNG, model::DynamicPPL.Model, ::SMC, N::Integer)
    return ParticleContainer([TracedModel(rng, model) for _ in 1:N]), false
end

function initialize(
    rng::AbstractRNG, model::DynamicPPL.Model, sampler::SMC, N::Integer, ref
)
    return particles, is_done = if isnothing(ref)
        initialize(rng, model, sampler, N)
    else
        particles, is_done = initialize(rng, model, sampler, N - 1)
        ReferencedContainer(particles, Particle(ref)), is_done
    end
end

# TODO: replace this with a systematic resampler
function sample_ancestors(
    rng::AbstractRNG, weights::Vector{<:Real}, N::Integer=length(weights)
)
    return rand(rng, Categorical(weights), N)
end

increment_weight!(particle::Particle, score::Nothing) = true
increment_weight!(particle::Particle, score::Real) = (particle.logw += score; return false)

function reweight!(particles, ::AbstractMCMC.MCMCSerial)
    num_done = map(particles) do particle
        score = advance!(particle)
        increment_weight!(particle, score)
    end
    return all(num_done)
end

function reweight!(particles, ::AbstractMCMC.MCMCThreads)
    num_done = Vector{Bool}(undef, length(particles))
    Threads.@threads for i in eachindex(particles)
        score = advance!(particles[i])
        num_done[i] = increment_weight!(particles[i], score)
    end
    return all(num_done)
end

function maybe_resample!(rng::AbstractRNG, particles, resampler::AbstractResampler)
    weights = StatsBase.weights(particles)
    rs_flag = should_resample(weights, resampler)
    rs_flag && resample!(rng, particles, weights)
    return rs_flag
end

# leave out rejuvenation for now, we'll cross that bridge when we get there
function rejuvenate!(
    ::AbstractRNG,
    particles::ParticleContainer,
    ::Nothing,
    ::AbstractMCMC.AbstractMCMCEnsemble,
    ::Bool,
    ::Integer;
    kwargs...,
)
    return particles
end

function smcsample(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    sampler::SMC,
    ensemble::AbstractMCMC.AbstractMCMCEnsemble,
    N::Integer;
    ref=nothing,
)
    particles, is_done = initialize(rng, model, sampler, N, ref)
    iter = 0
    while !is_done
        rs_flag = maybe_resample!(rng, particles, sampler.resampler)
        particles = rejuvenate!(rng, particles, sampler.kernel, ensemble, rs_flag, iter)
        is_done = reweight!(particles, ensemble)
        iter += 1
    end
    return particles
end

# this doesn't return an MCMCChain which kinda blows, but whatever
function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    sampler::SMC,
    N::Integer;
    ensemble::AbstractMCMC.AbstractMCMCEnsemble=MCMCSerial(),
    kwargs...,
)
    return smcsample(rng, model, sampler, ensemble, N; kwargs...)
end

####
#### Particle Gibbs and Conditional SMC.
####

"""
    ParticleGibbs

An MCMC sampler which wraps a conditional Sequential Monte Carlo step at every iteration of
the sampler for `N` iterations.

See [`SMC`](@ref) for details on the Sequential Monte Carlo kernel.

# Examples
```julia
# samples 128 particles each iteration, resampling when ESS drops below 50%
chain = sample(model, PG(SMC(0.5), 128), 10_000)
```
"""
struct ParticleGibbs{T<:SMC} <: AbstractMCMC.AbstractSampler
    kernel::T
    N::Int
end

const PG{T} = ParticleGibbs{T}

function AbstractMCMC.step(
    rng::AbstractRNG, model::DynamicPPL.Model, sampler::ParticleGibbs; kwargs...
)
    particles = smcsample(rng, model, sampler.kernel, MCMCSerial(), sampler.N)
    state = sample(rng, particles)
    return get_varinfo(state), state
end

# NOTE: this needs some TLC, not sure how I can better integrate
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    sampler::ParticleGibbs,
    state::TracedModel;
    kwargs...,
)
    particles = smcsample(
        rng, model, sampler.kernel, MCMCSerial(), sampler.N; ref=deepcopy(state)
    )
    state = sample(rng, particles)
    return get_varinfo(state), state
end
