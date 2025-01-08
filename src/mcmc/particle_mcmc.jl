###
### Particle Filtering and Particle MCMC Samplers.
###

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
    SMC()
    SMC([resampler = AdvancedPS.ResampleWithESSThreshold()])
    SMC([resampler = AdvancedPS.resample_systematic, ]threshold)

Create a sequential Monte Carlo sampler of type [`SMC`](@ref).

If the algorithm for the resampling step is not specified explicitly, systematic resampling
is performed if the estimated effective sample size per particle drops below 0.5.
"""
function SMC(resampler=AdvancedPS.ResampleWithESSThreshold())
    return SMC{typeof(resampler)}(resampler)
end

# Convenient constructors with ESS threshold
function SMC(resampler, threshold::Real)
    return SMC(AdvancedPS.ResampleWithESSThreshold(resampler, threshold))
end
function SMC(threshold::Real)
    return SMC(AdvancedPS.resample_systematic, threshold)
end

drop_space(alg::SMC) = alg
DynamicPPL.getspace(::SMC) = ()

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
    set_retained_vns_del_by_spl!(vi, spl)
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
    PG(n)
    PG(n, [resampler = AdvancedPS.ResampleWithESSThreshold()])
    PG(n, [resampler = AdvancedPS.resample_systematic, ]threshold)

Create a Particle Gibbs sampler of type [`PG`](@ref) with `n` particles.

If the algorithm for the resampling step is not specified explicitly, systematic resampling
is performed if the estimated effective sample size per particle drops below 0.5.
"""
function PG(nparticles::Int, resampler=AdvancedPS.ResampleWithESSThreshold())
    return PG{typeof(resampler)}(nparticles, resampler)
end

# Convenient constructors with ESS threshold
function PG(nparticles::Int, resampler, threshold::Real)
    return PG(nparticles, AdvancedPS.ResampleWithESSThreshold(resampler, threshold))
end
function PG(nparticles::Int, threshold::Real)
    return PG(nparticles, AdvancedPS.resample_systematic, threshold)
end

drop_space(alg::PG) = alg
DynamicPPL.getspace(::PG) = ()

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
    set_retained_vns_del_by_spl!(vi, spl)
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
    set_retained_vns_del_by_spl!(vi, spl)

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
        trace = AdvancedPS.current_trace()
        return trace.model.f.varinfo
    catch e
        # NOTE: this heuristic allows Libtask evaluating a model outside a `Trace`.
        if e == KeyError(:__trace) || current_task().storage isa Nothing
            return varinfo
        else
            rethrow(e)
        end
    end
end

function trace_local_rng_maybe(rng::Random.AbstractRNG)
    try
        trace = AdvancedPS.current_trace()
        return trace.rng
    catch e
        # NOTE: this heuristic allows Libtask evaluating a model outside a `Trace`.
        if e == KeyError(:__trace) || current_task().storage isa Nothing
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
        push!!(vi, vn, r, dist, spl)
    elseif is_flagged(vi, vn, "del")
        unset_flag!(vi, vn, "del") # Reference particle parent
        r = rand(trng, dist)
        vi[vn] = DynamicPPL.tovec(r)
        DynamicPPL.setgid!(vi, spl.selector, vn)
        setorder!(vi, vn, get_num_produce(vi))
    else
        DynamicPPL.updategid!(vi, vn, spl) # Pick data from reference particle
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

    tmodel = Turing.Essential.TracedModel(model, sampler, newvarinfo, rng)
    newtrace = AdvancedPS.Trace(tmodel, rng)
    AdvancedPS.addreference!(newtrace.model.ctask.task, newtrace)
    return newtrace
end
