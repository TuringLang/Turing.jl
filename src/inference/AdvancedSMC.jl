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
    SMC([resampler = AdvancedPS.resample, ]threshold[, space = ()])

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
SMC(threshold::Real, space::Tuple = ()) = SMC(AdvancedPS.resample, threshold, space)

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
    ::AbstractRNG,
    model::AbstractModel,
    spl::Sampler{<:SMC},
    vi::AbstractVarInfo;
    nparticles::Int,
    kwargs...
)
    # Reset the VarInfo.
    reset_num_produce!(vi)
    set_retained_vns_del_by_spl!(vi, spl)
    resetlogp!(vi)
    empty!(vi)

    # Create a new set of particles.
    particles = AdvancedPS.ParticleContainer(
        [AdvancedPS.Trace(model, spl, vi) for _ in 1:nparticles],
    )

    # Perform particle sweep.
    logevidence = AdvancedPS.sweep!(particles, spl.alg.resampler)

    # Extract the first particle and its weight.
    particle = particles.vals[1]
    weight = AdvancedPS.getweight(particles, 1)

    # Compute the first transition and the first state.
    transition = SMCTransition(particle.f.varinfo, weight)
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
    transition = SMCTransition(particle.f.varinfo, weight)
    nextstate = SMCState(state.particles, index + 1, state.average_logevidence)

    return transition, nextstate
end

####
#### Particle Gibbs sampler.
####

"""
$(TYPEDEF)

Particle Gibbs sampler.

Note that this method is particle-based, and arrays of variables
must be stored in a [`TArray`](@ref) object.

# Fields

$(TYPEDFIELDS)
"""
struct PG{space,R} <: ParticleInference
    """Number of particles."""
    nparticles::Int
    """Resampling algorithm."""
    resampler::R
end

isgibbscomponent(::PG) = true

"""
    PG(n, space...)
    PG(n, [resampler = AdvancedPS.ResampleWithESSThreshold(), space = ()])
    PG(n, [resampler = AdvancedPS.resample, ]threshold[, space = ()])

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
    return PG(nparticles, AdvancedPS.resample, threshold, space)
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

function PGTransition(vi::AbstractVarInfo, logevidence)
    theta = tonamedtuple(vi)

    # This is pretty useless since we reset the log probability continuously in the
    # particle sweep.
    lp = getlogp(vi)

    return PGTransition(theta, lp, logevidence)
end

metadata(t::PGTransition) = (lp = t.lp, logevidence = t.logevidence)

DynamicPPL.getlogp(t::PGTransition) = t.lp

function getlogevidence(samples, sampler::Sampler{<:PG}, vi::AbstractVarInfo)
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
    resetlogp!(vi)

    # Create a new set of particles
    num_particles = spl.alg.nparticles
    particles = AdvancedPS.ParticleContainer(
        [AdvancedPS.Trace(model, spl, vi) for _ in 1:num_particles],
    )

    # Perform a particle sweep.
    logevidence = AdvancedPS.sweep!(particles, spl.alg.resampler)

    # Pick a particle to be retained.
    Ws = AdvancedPS.getweights(particles)
    indx = AdvancedPS.randcat(Ws)
    reference = particles.vals[indx]

    # Compute the first transition.
    _vi = reference.f.varinfo
    transition = PGTransition(_vi, logevidence)

    return transition, _vi
end

function AbstractMCMC.step(
    ::AbstractRNG,
    model::AbstractModel,
    spl::Sampler{<:PG},
    vi::AbstractVarInfo;
    kwargs...
)
    # Reset the VarInfo before new sweep.
    reset_num_produce!(vi)
    set_retained_vns_del_by_spl!(vi, spl)
    resetlogp!(vi)

    # Create a new set of particles.
    num_particles = spl.alg.nparticles
    x = map(1:num_particles) do i
        if i != num_particles
            return AdvancedPS.Trace(model, spl, vi)
        else
        # Create reference particle.
            return AdvancedPS.forkr(AdvancedPS.Trace(model, spl, vi))
        end
    end
    particles = AdvancedPS.ParticleContainer(x)

    # Perform a particle sweep.
    logevidence = AdvancedPS.sweep!(particles, spl.alg.resampler)

    # Pick a particle to be retained.
    Ws = AdvancedPS.getweights(particles)
    indx = AdvancedPS.randcat(Ws)
    newreference = particles.vals[indx]

    # Compute the transition.
    _vi = newreference.f.varinfo
    transition = PGTransition(_vi, logevidence)

    return transition, _vi
end

function DynamicPPL.assume(
    rng,
    spl::Sampler{<:Union{PG,SMC}},
    dist::Distribution,
    vn::VarName,
    ::Any
)
    vi = AdvancedPS.current_trace().f.varinfo
    if inspace(vn, spl)
        if ~haskey(vi, vn)
            r = rand(rng, dist)
            push!(vi, vn, r, dist, spl)
        elseif is_flagged(vi, vn, "del")
            unset_flag!(vi, vn, "del")
            r = rand(rng, dist)
            vi[vn] = vectorize(dist, r)
            setgid!(vi, spl.selector, vn)
            setorder!(vi, vn, get_num_produce(vi))
        else
            updategid!(vi, vn, spl)
            r = vi[vn]
        end
    else # vn belongs to other sampler <=> conditionning on vn
        if haskey(vi, vn)
            r = vi[vn]
        else
            r = rand(rng, dist)
            push!(vi, vn, r, dist, Selector(:invalid))
        end
        lp = logpdf_with_trans(dist, r, istrans(vi, vn))
        acclogp!(vi, lp)
    end
    return r, 0
end

function DynamicPPL.observe(spl::Sampler{<:Union{PG,SMC}}, dist::Distribution, value, vi)
    produce(logpdf(dist, value))
    return 0
end

# Convenient constructor
function AdvancedPS.Trace(
    model::Model,
    sampler::Sampler{<:Union{SMC,PG}},
    varinfo::AbstractVarInfo,
)
    newvarinfo = deepcopy(varinfo)
    DynamicPPL.reset_num_produce!(newvarinfo)
    f = Turing.Core.TracedModel(model, sampler, newvarinfo)
    return AdvancedPS.Trace(f)
end
