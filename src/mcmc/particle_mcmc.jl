###
### Particle Filtering and Particle MCMC Samplers.
###

### AdvancedPS models and interface

struct TracedModel{S<:AbstractSampler,V<:AbstractVarInfo,M<:Model,E<:Tuple} <:
       AdvancedPS.AbstractGenericModel
    model::M
    sampler::S
    varinfo::V
    evaluator::E
end

function TracedModel(
    model::Model,
    sampler::AbstractSampler,
    varinfo::AbstractVarInfo,
    rng::Random.AbstractRNG,
)
    context = SamplingContext(rng, sampler, DefaultContext())
    args, kwargs = DynamicPPL.make_evaluate_args_and_kwargs(model, varinfo, context)
    if kwargs !== nothing && !isempty(kwargs)
        error(
            "Sampling with `$(sampler.alg)` does not support models with keyword arguments. See issue #2007 for more details.",
        )
    end
    evaluator = (model.f, args...)
    return TracedModel(model, sampler, varinfo, evaluator)
end

function AdvancedPS.advance!(
    trace::AdvancedPS.Trace{<:AdvancedPS.LibtaskModel{<:TracedModel}}, isref::Bool=false
)
    # Make sure we load/reset the rng in the new replaying mechanism
    DynamicPPL.increment_num_produce!(trace.model.f.varinfo)
    isref ? AdvancedPS.load_state!(trace.rng) : AdvancedPS.save_state!(trace.rng)
    score = consume(trace.model.ctask)
    if score === nothing
        return nothing
    else
        return score + DynamicPPL.getlogp(trace.model.f.varinfo)
    end
end

function AdvancedPS.delete_retained!(trace::TracedModel)
    DynamicPPL.set_retained_vns_del!(trace.varinfo)
    return trace
end

function AdvancedPS.reset_model(trace::TracedModel)
    DynamicPPL.reset_num_produce!(trace.varinfo)
    return trace
end

function AdvancedPS.reset_logprob!(trace::TracedModel)
    DynamicPPL.resetlogp!!(trace.model.varinfo)
    return trace
end

function Libtask.TapedTask(taped_globals::Any, model::TracedModel, args...; kwargs...) # RNG ?
    return Libtask.TapedTask(
        taped_globals, model.evaluator[1], model.evaluator[2:end]...; kwargs...
    )
end

abstract type ParticleInference <: InferenceAlgorithm end

####
#### Generic Sequential Monte Carlo sampler.
####

"""
$(TYPEDEF)

Sequential Monte Carlo sampler.

# Fields

$(TYPEDFIELDS)
"""
struct SMC{R} <: ParticleInference
    resampler::R
end

"""
    SMC([resampler = AdvancedPS.ResampleWithESSThreshold()])
    SMC([resampler = AdvancedPS.resample_systematic, ]threshold)

Create a sequential Monte Carlo sampler of type [`SMC`](@ref).

If the algorithm for the resampling step is not specified explicitly, systematic resampling
is performed if the estimated effective sample size per particle drops below 0.5.
"""
SMC() = SMC(AdvancedPS.ResampleWithESSThreshold())

# Convenient constructors with ESS threshold
function SMC(resampler, threshold::Real)
    return SMC(AdvancedPS.ResampleWithESSThreshold(resampler, threshold))
end
function SMC(threshold::Real)
    return SMC(AdvancedPS.resample_systematic, threshold)
end

struct SMCTransition{T,F<:AbstractFloat} <: AbstractTransition
    "The parameters for any given sample."
    θ::T
    "The joint log probability of the sample (NOTE: does not work, always set to zero)."
    lp::F
    "The weight of the particle the sample was retrieved from."
    weight::F
end

function SMCTransition(model::DynamicPPL.Model, vi::AbstractVarInfo, weight)
    theta = getparams(model, vi)

    # This is pretty useless since we reset the log probability continuously in the
    # particle sweep.
    lp = getlogp(vi)

    return SMCTransition(theta, lp, weight)
end

metadata(t::SMCTransition) = (lp=t.lp, weight=t.weight)

DynamicPPL.getlogp(t::SMCTransition) = t.lp

struct SMCState{P,F<:AbstractFloat}
    particles::P
    particleindex::Int
    # The logevidence after aggregating all samples together.
    average_logevidence::F
end

function getlogevidence(samples, sampler::Sampler{<:SMC}, state::SMCState)
    return state.average_logevidence
end

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    sampler::Sampler{<:SMC},
    N::Integer;
    chain_type=DynamicPPL.default_chain_type(sampler),
    resume_from=nothing,
    initial_state=DynamicPPL.loadstate(resume_from),
    progress=PROGRESS[],
    kwargs...,
)
    if resume_from === nothing
        return AbstractMCMC.mcmcsample(
            rng,
            model,
            sampler,
            N;
            chain_type=chain_type,
            progress=progress,
            nparticles=N,
            kwargs...,
        )
    else
        return AbstractMCMC.mcmcsample(
            rng,
            model,
            sampler,
            N;
            chain_type,
            initial_state,
            progress=progress,
            nparticles=N,
            kwargs...,
        )
    end
end

function DynamicPPL.initialstep(
    rng::AbstractRNG,
    model::AbstractModel,
    spl::Sampler{<:SMC},
    vi::AbstractVarInfo;
    nparticles::Int,
    kwargs...,
)
    # Reset the VarInfo.
    reset_num_produce!(vi)
    set_retained_vns_del!(vi)
    resetlogp!!(vi)
    empty!!(vi)

    # Create a new set of particles.
    particles = AdvancedPS.ParticleContainer(
        [AdvancedPS.Trace(model, spl, vi, AdvancedPS.TracedRNG()) for _ in 1:nparticles],
        AdvancedPS.TracedRNG(),
        rng,
    )

    # Perform particle sweep.
    logevidence = AdvancedPS.sweep!(rng, particles, spl.alg.resampler, spl)

    # Extract the first particle and its weight.
    particle = particles.vals[1]
    weight = AdvancedPS.getweight(particles, 1)

    # Compute the first transition and the first state.
    transition = SMCTransition(model, particle.model.f.varinfo, weight)
    state = SMCState(particles, 2, logevidence)

    return transition, state
end

function AbstractMCMC.step(
    ::AbstractRNG, model::AbstractModel, spl::Sampler{<:SMC}, state::SMCState; kwargs...
)
    # Extract the index of the current particle.
    index = state.particleindex

    # Extract the current particle and its weight.
    particles = state.particles
    particle = particles.vals[index]
    weight = AdvancedPS.getweight(particles, index)

    # Compute the transition and the next state.
    transition = SMCTransition(model, particle.model.f.varinfo, weight)
    nextstate = SMCState(state.particles, index + 1, state.average_logevidence)

    return transition, nextstate
end

####
#### Particle Gibbs sampler.
####

"""
$(TYPEDEF)

Particle Gibbs sampler.

# Fields

$(TYPEDFIELDS)
"""
struct PG{R} <: ParticleInference
    """Number of particles."""
    nparticles::Int
    """Resampling algorithm."""
    resampler::R
end

"""
    PG(n, [resampler = AdvancedPS.ResampleWithESSThreshold()])
    PG(n, [resampler = AdvancedPS.resample_systematic, ]threshold)

Create a Particle Gibbs sampler of type [`PG`](@ref) with `n` particles.

If the algorithm for the resampling step is not specified explicitly, systematic resampling
is performed if the estimated effective sample size per particle drops below 0.5.
"""
function PG(nparticles::Int)
    return PG(nparticles, AdvancedPS.ResampleWithESSThreshold())
end

# Convenient constructors with ESS threshold
function PG(nparticles::Int, resampler, threshold::Real)
    return PG(nparticles, AdvancedPS.ResampleWithESSThreshold(resampler, threshold))
end
function PG(nparticles::Int, threshold::Real)
    return PG(nparticles, AdvancedPS.resample_systematic, threshold)
end

"""
    CSMC(...)

Equivalent to [`PG`](@ref).
"""
const CSMC = PG # type alias of PG as Conditional SMC

struct PGTransition{T,F<:AbstractFloat} <: AbstractTransition
    "The parameters for any given sample."
    θ::T
    "The joint log probability of the sample (NOTE: does not work, always set to zero)."
    lp::F
    "The log evidence of the sample."
    logevidence::F
end

struct PGState
    vi::AbstractVarInfo
    rng::Random.AbstractRNG
end

varinfo(state::PGState) = state.vi

function PGTransition(model::DynamicPPL.Model, vi::AbstractVarInfo, logevidence)
    theta = getparams(model, vi)

    # This is pretty useless since we reset the log probability continuously in the
    # particle sweep.
    lp = getlogp(vi)

    return PGTransition(theta, lp, logevidence)
end

metadata(t::PGTransition) = (lp=t.lp, logevidence=t.logevidence)

DynamicPPL.getlogp(t::PGTransition) = t.lp

function getlogevidence(samples, sampler::Sampler{<:PG}, state::PGState)
    return mean(x.logevidence for x in samples)
end

function DynamicPPL.initialstep(
    rng::AbstractRNG,
    model::AbstractModel,
    spl::Sampler{<:PG},
    vi::AbstractVarInfo;
    kwargs...,
)
    # Reset the VarInfo before new sweep
    reset_num_produce!(vi)
    set_retained_vns_del!(vi)
    resetlogp!!(vi)

    # Create a new set of particles
    num_particles = spl.alg.nparticles
    particles = AdvancedPS.ParticleContainer(
        [AdvancedPS.Trace(model, spl, vi, AdvancedPS.TracedRNG()) for _ in 1:num_particles],
        AdvancedPS.TracedRNG(),
        rng,
    )

    # Perform a particle sweep.
    logevidence = AdvancedPS.sweep!(rng, particles, spl.alg.resampler, spl)

    # Pick a particle to be retained.
    Ws = AdvancedPS.getweights(particles)
    indx = AdvancedPS.randcat(rng, Ws)
    reference = particles.vals[indx]

    # Compute the first transition.
    _vi = reference.model.f.varinfo
    transition = PGTransition(model, _vi, logevidence)

    return transition, PGState(_vi, reference.rng)
end

function AbstractMCMC.step(
    rng::AbstractRNG, model::AbstractModel, spl::Sampler{<:PG}, state::PGState; kwargs...
)
    # Reset the VarInfo before new sweep.
    vi = state.vi
    reset_num_produce!(vi)
    resetlogp!!(vi)

    # Create reference particle for which the samples will be retained.
    reference = AdvancedPS.forkr(AdvancedPS.Trace(model, spl, vi, state.rng))

    # For all other particles, do not retain the variables but resample them.
    set_retained_vns_del!(vi)

    # Create a new set of particles.
    num_particles = spl.alg.nparticles
    x = map(1:num_particles) do i
        if i != num_particles
            return AdvancedPS.Trace(model, spl, vi, AdvancedPS.TracedRNG())
        else
            return reference
        end
    end
    particles = AdvancedPS.ParticleContainer(x, AdvancedPS.TracedRNG(), rng)

    # Perform a particle sweep.
    logevidence = AdvancedPS.sweep!(rng, particles, spl.alg.resampler, spl, reference)

    # Pick a particle to be retained.
    Ws = AdvancedPS.getweights(particles)
    indx = AdvancedPS.randcat(rng, Ws)
    newreference = particles.vals[indx]

    # Compute the transition.
    _vi = newreference.model.f.varinfo
    transition = PGTransition(model, _vi, logevidence)

    return transition, PGState(_vi, newreference.rng)
end

function DynamicPPL.use_threadsafe_eval(
    ::SamplingContext{<:Sampler{<:Union{PG,SMC}}}, ::AbstractVarInfo
)
    return false
end

function trace_local_varinfo_maybe(varinfo)
    try
        trace = Libtask.get_taped_globals(Any).other
        return (trace === nothing ? varinfo : trace.model.f.varinfo)::AbstractVarInfo
    catch e
        # NOTE: this heuristic allows Libtask evaluating a model outside a `Trace`.
        if e == KeyError(:task_variable)
            return varinfo
        else
            rethrow(e)
        end
    end
end

function trace_local_rng_maybe(rng::Random.AbstractRNG)
    try
        return Libtask.get_taped_globals(Any).rng
    catch e
        # NOTE: this heuristic allows Libtask evaluating a model outside a `Trace`.
        if e == KeyError(:task_variable)
            return rng
        else
            rethrow(e)
        end
    end
end

function DynamicPPL.assume(
    rng,
    spl::Sampler{<:Union{PG,SMC}},
    dist::Distribution,
    vn::VarName,
    _vi::AbstractVarInfo,
)
    vi = trace_local_varinfo_maybe(_vi)
    trng = trace_local_rng_maybe(rng)

    if ~haskey(vi, vn)
        r = rand(trng, dist)
        push!!(vi, vn, r, dist)
    elseif is_flagged(vi, vn, "del")
        unset_flag!(vi, vn, "del") # Reference particle parent
        r = rand(trng, dist)
        vi[vn] = DynamicPPL.tovec(r)
        setorder!(vi, vn, get_num_produce(vi))
    else
        r = vi[vn]
    end
    # TODO: Should we make this `zero(promote_type(eltype(dist), eltype(r)))` or something?
    lp = 0
    return r, lp, vi
end

function DynamicPPL.observe(spl::Sampler{<:Union{PG,SMC}}, dist::Distribution, value, vi)
    # NOTE: The `Libtask.produce` is now hit in `acclogp_observe!!`.
    return logpdf(dist, value), trace_local_varinfo_maybe(vi)
end

function DynamicPPL.acclogp!!(
    context::SamplingContext{<:Sampler{<:Union{PG,SMC}}}, varinfo::AbstractVarInfo, logp
)
    varinfo_trace = trace_local_varinfo_maybe(varinfo)
    return DynamicPPL.acclogp!!(DynamicPPL.childcontext(context), varinfo_trace, logp)
end

function DynamicPPL.acclogp_observe!!(
    context::SamplingContext{<:Sampler{<:Union{PG,SMC}}}, varinfo::AbstractVarInfo, logp
)
    Libtask.produce(logp)
    return trace_local_varinfo_maybe(varinfo)
end

# Convenient constructor
function AdvancedPS.Trace(
    model::Model,
    sampler::Sampler{<:Union{SMC,PG}},
    varinfo::AbstractVarInfo,
    rng::AdvancedPS.TracedRNG,
)
    newvarinfo = deepcopy(varinfo)
    DynamicPPL.reset_num_produce!(newvarinfo)

    tmodel = TracedModel(model, sampler, newvarinfo, rng)
    newtrace = AdvancedPS.Trace(tmodel, rng)
    return newtrace
end

# We need to tell Libtask which calls may have `produce` calls within them. In practice most
# of these won't be needed, because of inlining and the fact that `might_produce` is only
# called on `:invoke` expressions rather than `:call`s, but since those are implementation
# details of the compiler, we set a bunch of methods as might_produce = true. We start with
# `acclogp_observe!!` which is what calls `produce` and go up the call stack.
Libtask.might_produce(::Type{<:Tuple{typeof(DynamicPPL.acclogp_observe!!),Vararg}}) = true
Libtask.might_produce(::Type{<:Tuple{typeof(DynamicPPL.tilde_observe!!),Vararg}}) = true
Libtask.might_produce(::Type{<:Tuple{typeof(DynamicPPL.evaluate!!),Vararg}}) = true
function Libtask.might_produce(
    ::Type{<:Tuple{typeof(DynamicPPL.evaluate_threadsafe!!),Vararg}}
)
    return true
end
function Libtask.might_produce(
    ::Type{<:Tuple{typeof(DynamicPPL.evaluate_threadunsafe!!),Vararg}}
)
    return true
end
Libtask.might_produce(::Type{<:Tuple{<:DynamicPPL.Model,Vararg}}) = true
