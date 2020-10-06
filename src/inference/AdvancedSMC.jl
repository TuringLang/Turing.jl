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
    SMC([resampler = ResampleWithESSThreshold(), space = ()])
    SMC([resampler = resample_systematic, ]threshold[, space = ()])

Create a sequential Monte Carlo sampler of type [`SMC`](@ref) for the variables in `space`.

If the algorithm for the resampling step is not specified explicitly, systematic resampling
is performed if the estimated effective sample size per particle drops below 0.5.
"""
function SMC(resampler = Turing.Core.ResampleWithESSThreshold(), space::Tuple = ())
    return SMC{space, typeof(resampler)}(resampler)
end

# Convenient constructors with ESS threshold
function SMC(resampler, threshold::Real, space::Tuple = ())
    return SMC(Turing.Core.ResampleWithESSThreshold(resampler, threshold), space)
end
SMC(threshold::Real, space::Tuple = ()) = SMC(resample_systematic, threshold, space)

# If only the space is defined
SMC(space::Symbol...) = SMC(space)
SMC(space::Tuple) = SMC(Turing.Core.ResampleWithESSThreshold(), space)

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

function additional_parameters(::Type{<:SMCTransition})
    return [:lp, :weight]
end

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
    T = Trace{typeof(spl),typeof(vi),typeof(model)}
    particles = ParticleContainer(T[Trace(model, spl, vi) for _ in 1:nparticles])

    # Perform particle sweep.
    logevidence = sweep!(particles, spl.alg.resampler)

    # Extract the first particle and its weight.
    particle = particles.vals[1]
    weight = getweight(particles, 1)

    # Compute the first transition and the first state.
    transition = SMCTransition(particle.vi, weight)
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
    weight = getweight(particles, index)

    # Compute the transition and the next state.
    transition = SMCTransition(particle.vi, weight)
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
    PG(n, [resampler = ResampleWithESSThreshold(), space = ()])
    PG(n, [resampler = resample_systematic, ]threshold[, space = ()])

Create a Particle Gibbs sampler of type [`PG`](@ref) with `n` particles for the variables
in `space`.

If the algorithm for the resampling step is not specified explicitly, systematic resampling
is performed if the estimated effective sample size per particle drops below 0.5.
"""
function PG(
    nparticles::Int,
    resampler = Turing.Core.ResampleWithESSThreshold(),
    space::Tuple = (),
)
    return PG{space, typeof(resampler)}(nparticles, resampler)
end

# Convenient constructors with ESS threshold
function PG(nparticles::Int, resampler, threshold::Real, space::Tuple = ())
    return PG(nparticles, Turing.Core.ResampleWithESSThreshold(resampler, threshold), space)
end
function PG(nparticles::Int, threshold::Real, space::Tuple = ())
    return PG(nparticles, resample_systematic, threshold, space)
end

# If only the number of particles and the space is defined
PG(nparticles::Int, space::Symbol...) = PG(nparticles, space)
function PG(nparticles::Int, space::Tuple)
    return PG(nparticles, Turing.Core.ResampleWithESSThreshold(), space)
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

function additional_parameters(::Type{<:PGTransition})
    return [:lp, :logevidence]
end

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
    T = Trace{typeof(spl),typeof(vi),typeof(model)}
    particles = ParticleContainer(T[Trace(model, spl, vi) for _ in 1:num_particles])

    # Perform a particle sweep.
    logevidence = sweep!(particles, spl.alg.resampler)

    # Pick a particle to be retained.
    Ws = getweights(particles)
    indx = randcat(Ws)
    reference = particles.vals[indx]

    # Compute the first transition.
    _vi = reference.vi
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
    T = Trace{typeof(spl),typeof(vi),typeof(model)}
    x = Vector{T}(undef, num_particles)
    @inbounds for i in 1:(num_particles - 1)
        x[i] = Trace(model, spl, vi)
    end
    # Create reference particle.
    @inbounds x[num_particles] = forkr(Trace(model, spl, vi))
    particles = ParticleContainer(x)

    # Perform a particle sweep.
    logevidence = sweep!(particles, spl.alg.resampler)

    # Pick a particle to be retained.
    Ws = getweights(particles)
    indx = randcat(Ws)
    newreference = particles.vals[indx]

    # Compute the transition.
    _vi = newreference.vi
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
    vi = current_trace().vi
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

####
#### Resampling schemes for particle filters
####

# Some references
#  - http://arxiv.org/pdf/1301.4019.pdf
#  - http://people.isy.liu.se/rt/schon/Publications/HolSG2006.pdf
# Code adapted from: http://uk.mathworks.com/matlabcentral/fileexchange/24968-resampling-methods-for-particle-filtering

# Default resampling scheme
function resample(w::AbstractVector{<:Real}, num_particles::Integer=length(w))
    return resample_systematic(w, num_particles)
end

# More stable, faster version of rand(Categorical)
function randcat(p::AbstractVector{<:Real})
    T = eltype(p)
    r = rand(T)
    s = 1
    for j in eachindex(p)
        r -= p[j]
        if r <= zero(T)
            s = j
            break
        end
    end
    return s
end

function resample_multinomial(w::AbstractVector{<:Real}, num_particles::Integer)
    return rand(Distributions.sampler(Categorical(w)), num_particles)
end

function resample_residual(w::AbstractVector{<:Real}, num_particles::Integer)
    # Pre-allocate array for resampled particles
    indices = Vector{Int}(undef, num_particles)

    # deterministic assignment
    residuals = similar(w)
    i = 1
    @inbounds for j in 1:length(w)
        x = num_particles * w[j]
        floor_x = floor(Int, x)
        for k in 1:floor_x
            indices[i] = j
            i += 1
        end
        residuals[j] = x - floor_x
    end
    
    # sampling from residuals
    if i <= num_particles
        residuals ./= sum(residuals)
        rand!(Categorical(residuals), view(indices, i:num_particles))
    end
    
    return indices
end


"""
    resample_stratified(weights, n)

Return a vector of `n` samples `x₁`, ..., `xₙ` from the numbers 1, ..., `length(weights)`,
generated by stratified resampling.

In stratified resampling `n` ordered random numbers `u₁`, ..., `uₙ` are generated, where
``uₖ \\sim U[(k - 1) / n, k / n)``. Based on these numbers the samples `x₁`, ..., `xₙ`
are selected according to the multinomial distribution defined by the normalized `weights`,
i.e., `xᵢ = j` if and only if
``uᵢ \\in [\\sum_{s=1}^{j-1} weights_{s}, \\sum_{s=1}^{j} weights_{s})``.
"""
function resample_stratified(weights::AbstractVector{<:Real}, n::Integer)
    # check input
    m = length(weights)
    m > 0 || error("weight vector is empty")

    # pre-calculations
    @inbounds v = n * weights[1]

    # generate all samples
    samples = Array{Int}(undef, n)
    sample = 1
    @inbounds for i in 1:n
        # sample next `u` (scaled by `n`)
        u = oftype(v, i - 1 + rand())

        # as long as we have not found the next sample
        while v < u
            # increase and check the sample
            sample += 1
            sample > m &&
                error("sample could not be selected (are the weights normalized?)")

            # update the cumulative sum of weights (scaled by `n`)
            v += n * weights[sample]
        end

        # save the next sample
        samples[i] = sample
    end

    return samples
end

"""
    resample_systematic(weights, n)

Return a vector of `n` samples `x₁`, ..., `xₙ` from the numbers 1, ..., `length(weights)`,
generated by systematic resampling.

In systematic resampling a random number ``u \\sim U[0, 1)`` is used to generate `n` ordered
numbers `u₁`, ..., `uₙ` where ``uₖ = (u + k − 1) / n``. Based on these numbers the samples
`x₁`, ..., `xₙ` are selected according to the multinomial distribution defined by the
normalized `weights`, i.e., `xᵢ = j` if and only if
``uᵢ \\in [\\sum_{s=1}^{j-1} weights_{s}, \\sum_{s=1}^{j} weights_{s})``.
"""
function resample_systematic(weights::AbstractVector{<:Real}, n::Integer)
    # check input
    m = length(weights)
    m > 0 || error("weight vector is empty")

    # pre-calculations
    @inbounds v = n * weights[1]
    u = oftype(v, rand())

    # find all samples
    samples = Array{Int}(undef, n)
    sample = 1
    @inbounds for i in 1:n
        # as long as we have not found the next sample
        while v < u
            # increase and check the sample
            sample += 1
            sample > m &&
                error("sample could not be selected (are the weights normalized?)")

            # update the cumulative sum of weights (scaled by `n`)
            v += n * weights[sample]
        end

        # save the next sample
        samples[i] = sample

        # update `u`
        u += one(u)
    end

    return samples
end
