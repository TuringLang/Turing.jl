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
struct SMC{space, R} <: ParticleInference
    resampler::R
end

"""
    SMC(space...)
    SMC([resampler = AdvancedPS.ResampleWithESSThreshold(), space = ()])
    SMC([resampler = AdvancedPS.resample_systematic, ]threshold[, space = ()])

Create a sequential Monte Carlo sampler of type [`SMC`](@ref) for the variables in `space`.

If the algorithm for the resampling step is not specified explicitly, systematic resampling
is performed if the estimated effective sample size per particle drops below 0.5.
"""
function SMC(resampler = AdvancedPS.ResampleWithESSThreshold(), space::Tuple = ())
    return SMC{space, typeof(resampler)}(resampler)
end

# Convenient constructors with ESS threshold
function SMC(resampler, threshold::Real, space::Tuple = ())
    return SMC(AdvancedPS.ResampleWithESSThreshold(resampler, threshold), space)
end
SMC(threshold::Real, space::Tuple = ()) = SMC(AdvancedPS.resample_systematic, threshold, space)

# If only the space is defined
SMC(space::Symbol...) = SMC(space)
SMC(space::Tuple) = SMC(AdvancedPS.ResampleWithESSThreshold(), space)

struct SMCTransition{T,F<:AbstractFloat}
    "The parameters for any given sample."
    θ::T
    "The joint log probability of the sample (NOTE: does not work, always set to zero)."
    lp::F
    "The weight of the particle the sample was retrieved from."
    weight::F
end

function SMCTransition(vi::AbstractVarInfo, weight)
    theta = tonamedtuple(vi)

    # This is pretty useless since we reset the log probability continuously in the
    # particle sweep.
    lp = getlogp(vi)

    return SMCTransition(theta, lp, weight)
end

metadata(t::SMCTransition) = (lp = t.lp, weight = t.weight)

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
    model::AbstractModel,
    sampler::Sampler{<:SMC},
    N::Integer;
    chain_type=MCMCChains.Chains,
    resume_from=nothing,
    progress=PROGRESS[],
    kwargs...
)
    if resume_from === nothing
        return AbstractMCMC.mcmcsample(rng, model, sampler, N;
                                       chain_type=chain_type,
                                       progress=progress,
                                       nparticles=N,
                                       kwargs...)
    else
        return resume(resume_from, N;
                      chain_type=chain_type, progress=progress, nparticles=N, kwargs...)
    end
end

function DynamicPPL.initialstep(
    rng::AbstractRNG,
    model::AbstractModel,
    spl::Sampler{<:SMC},
    vi::AbstractVarInfo;
    nparticles::Int,
    kwargs...
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
        rng
    )

    # Perform particle sweep.
    logevidence = AdvancedPS.sweep!(rng, particles, spl.alg.resampler)

    # Extract the first particle and its weight.
    particle = particles.vals[1]
    weight = AdvancedPS.getweight(particles, 1)

    # Compute the first transition and the first state.
    transition = SMCTransition(particle.model.f.varinfo, weight)
    state = SMCState(particles, 2, logevidence)

    return transition, state
end

function AbstractMCMC.step(
    ::AbstractRNG,
    model::AbstractModel,
    spl::Sampler{<:SMC},
    state::SMCState;
    kwargs...
)
    # Extract the index of the current particle.
    index = state.particleindex

    # Extract the current particle and its weight.
    particles = state.particles
    particle = particles.vals[index]
    weight = AdvancedPS.getweight(particles, index)

    # Compute the transition and the next state.
    transition = SMCTransition(particle.model.f.varinfo, weight)
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
struct PG{space,R} <: ParticleInference
    """Number of particles."""
    nparticles::Int
    """Resampling algorithm."""
    resampler::R
end

"""
    PG(n, space...)
    PG(n, [resampler = AdvancedPS.ResampleWithESSThreshold(), space = ()])
    PG(n, [resampler = AdvancedPS.resample_systematic, ]threshold[, space = ()])

Create a Particle Gibbs sampler of type [`PG`](@ref) with `n` particles for the variables
in `space`.

If the algorithm for the resampling step is not specified explicitly, systematic resampling
is performed if the estimated effective sample size per particle drops below 0.5.
"""
function PG(
    nparticles::Int,
    resampler = AdvancedPS.ResampleWithESSThreshold(),
    space::Tuple = (),
)
    return PG{space, typeof(resampler)}(nparticles, resampler)
end

# Convenient constructors with ESS threshold
function PG(nparticles::Int, resampler, threshold::Real, space::Tuple = ())
    return PG(nparticles, AdvancedPS.ResampleWithESSThreshold(resampler, threshold), space)
end
function PG(nparticles::Int, threshold::Real, space::Tuple = ())
    return PG(nparticles, AdvancedPS.resample_systematic, threshold, space)
end

# If only the number of particles and the space is defined
PG(nparticles::Int, space::Symbol...) = PG(nparticles, space)
function PG(nparticles::Int, space::Tuple)
    return PG(nparticles, AdvancedPS.ResampleWithESSThreshold(), space)
end

const CSMC = PG # type alias of PG as Conditional SMC

struct PGTransition{T,F<:AbstractFloat}
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

function PGTransition(vi::AbstractVarInfo, logevidence)
    theta = tonamedtuple(vi)

    # This is pretty useless since we reset the log probability continuously in the
    # particle sweep.
    lp = getlogp(vi)

    return PGTransition(theta, lp, logevidence)
end

metadata(t::PGTransition) = (lp = t.lp, logevidence = t.logevidence)

DynamicPPL.getlogp(t::PGTransition) = t.lp

function getlogevidence(samples, sampler::Sampler{<:PG}, state::PGState)
    return mean(x.logevidence for x in samples)
end

function DynamicPPL.initialstep(
    rng::AbstractRNG,
    model::AbstractModel,
    spl::Sampler{<:PG},
    vi::AbstractVarInfo;
    kwargs...
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
        rng
    )

    # Perform a particle sweep.
    logevidence = AdvancedPS.sweep!(rng, particles, spl.alg.resampler)

    # Pick a particle to be retained.
    Ws = AdvancedPS.getweights(particles)
    indx = AdvancedPS.randcat(rng, Ws)
    reference = particles.vals[indx]

    # Compute the first transition.
    _vi = reference.model.f.varinfo
    transition = PGTransition(_vi, logevidence)

    return transition, PGState(_vi, reference.rng)#_vi
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractModel,
    spl::Sampler{<:PG},
    state::PGState;
    #vi::AbstractVarInfo;
    kwargs...
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
    logevidence = AdvancedPS.sweep!(rng, particles, spl.alg.resampler, reference)

    # Pick a particle to be retained.
    Ws = AdvancedPS.getweights(particles)
    indx = AdvancedPS.randcat(rng, Ws)
    newreference = particles.vals[indx]

    # Compute the transition.
    _vi = newreference.model.f.varinfo
    transition = PGTransition(_vi, logevidence)

    return transition, PGState(_vi, newreference.rng)
end

DynamicPPL.use_threadsafe_eval(::SamplingContext{<:Sampler{<:Union{PG,SMC}}}, ::AbstractVarInfo) = false

function DynamicPPL.assume(
    rng,
    spl::Sampler{<:Union{PG,SMC}},
    dist::Distribution,
    vn::VarName,
    __vi__::AbstractVarInfo
)
    local vi
    trace = AdvancedPS.current_trace()
    trng = trace.rng
    try 
        vi = trace.model.f.varinfo
    catch e
        # NOTE: this heuristic allows Libtask evaluating a model outside a `Trace`. 
        if e == KeyError(:__trace) || current_task().storage isa Nothing
            vi = __vi__
        else
            rethrow(e)
        end
    end

    if inspace(vn, spl)
        if ~haskey(vi, vn)
            r = rand(trng, dist)
            push!!(vi, vn, r, dist, spl)
        elseif is_flagged(vi, vn, "del")
            unset_flag!(vi, vn, "del") # Reference particle parent
            r = rand(trng, dist)
            vi[vn] = vectorize(dist, r)
            DynamicPPL.setgid!(vi, spl.selector, vn)
            setorder!(vi, vn, get_num_produce(vi))
        else
            DynamicPPL.updategid!(vi, vn, spl) # Pick data from reference particle
            r = vi[vn]
        end
    else # vn belongs to other sampler <=> conditioning on vn
        if haskey(vi, vn)
            r = vi[vn]
        else
            r = rand(rng, dist)
            push!!(vi, vn, r, dist, DynamicPPL.Selector(:invalid))
        end
        lp = logpdf_with_trans(dist, r, istrans(vi, vn))
        acclogp!!(vi, lp)
    end
    return r, 0, vi
end
function DynamicPPL.observe(spl::Sampler{<:Union{PG,SMC}}, dist::Distribution, value, vi)
    Libtask.produce(logpdf(dist, value))
    return 0, vi
end

# Convenient constructor
function AdvancedPS.Trace(
    model::Model,
    sampler::Sampler{<:Union{SMC,PG}},
    varinfo::AbstractVarInfo,
    rng::AdvancedPS.TracedRNG
)
    newvarinfo = deepcopy(varinfo)
    DynamicPPL.reset_num_produce!(newvarinfo)

    tmodel = Turing.Essential.TracedModel(model, sampler, newvarinfo, rng)
    ttask = Libtask.TapedTask(tmodel, rng)
    wrapedmodel = AdvancedPS.GenericModel(tmodel, ttask)

    newtrace = AdvancedPS.Trace(wrapedmodel, rng)
    return newtrace
end
