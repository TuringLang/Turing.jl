mutable struct Trace{Tspl<:AbstractSampler, Tvi<:AbstractVarInfo, Tmodel<:Model}
    model::Tmodel
    spl::Tspl
    vi::Tvi
    ctask::CTask

    function Trace{SampleFromPrior}(model::Model, spl::AbstractSampler, vi::AbstractVarInfo)
        return new{SampleFromPrior,typeof(vi),typeof(model)}(model, SampleFromPrior(), vi)
    end
    function Trace{S}(model::Model, spl::S, vi::AbstractVarInfo) where S<:Sampler
        return new{S,typeof(vi),typeof(model)}(model, spl, vi)
    end
end

function Base.copy(trace::Trace)
    vi = deepcopy(trace.vi)
    res = Trace{typeof(trace.spl)}(trace.model, trace.spl, vi)
    res.ctask = copy(trace.ctask)
    return res
end

# NOTE: this function is called by `forkr`
function Trace(f, m::Model, spl::AbstractSampler, vi::AbstractVarInfo)
    res = Trace{typeof(spl)}(m, spl, deepcopy(vi))
    ctask = CTask() do
        res = f()
        produce(nothing)
        return res
    end
    task = ctask.task
    if task.storage === nothing
        task.storage = IdDict()
    end
    task.storage[:turing_trace] = res # create a backward reference in task_local_storage
    res.ctask = ctask
    return res
end

function Trace(m::Model, spl::AbstractSampler, vi::AbstractVarInfo)
    res = Trace{typeof(spl)}(m, spl, deepcopy(vi))
    reset_num_produce!(res.vi)
    ctask = CTask() do
        res = m(vi, spl)
        produce(nothing)
        return res
    end
    task = ctask.task
    if task.storage === nothing
        task.storage = IdDict()
    end
    task.storage[:turing_trace] = res # create a backward reference in task_local_storage
    res.ctask = ctask
    return res
end

# step to the next observe statement, return log likelihood
Libtask.consume(t::Trace) = (increment_num_produce!(t.vi); consume(t.ctask))

# Task copying version of fork for Trace.
function fork(trace :: Trace, is_ref :: Bool = false)
    newtrace = copy(trace)
    is_ref && set_retained_vns_del_by_spl!(newtrace.vi, newtrace.spl)
    newtrace.ctask.task.storage[:turing_trace] = newtrace
    return newtrace
end

# PG requires keeping all randomness for the reference particle
# Create new task and copy randomness
function forkr(trace::Trace)
    newtrace = Trace(trace.ctask.task.code, trace.model, trace.spl, deepcopy(trace.vi))
    newtrace.spl = trace.spl
    reset_num_produce!(newtrace.vi)
    return newtrace
end

current_trace() = current_task().storage[:turing_trace]

const Particle = Trace

"""
Data structure for particle filters
- effectiveSampleSize(pc :: ParticleContainer)
- normalise!(pc::ParticleContainer)
- consume(pc::ParticleContainer): return incremental likelihood
"""
mutable struct ParticleContainer{T<:Particle}
    "Particles."
    vals::Vector{T}
    "Unnormalized logarithmic weights."
    logWs::Vector{Float64}
end

function ParticleContainer(particles::Vector{<:Particle})
    return ParticleContainer(particles, zeros(length(particles)))
end

Base.collect(pc::ParticleContainer) = pc.vals
Base.length(pc::ParticleContainer) = length(pc.vals)
Base.@propagate_inbounds Base.getindex(pc::ParticleContainer, i::Int) = pc.vals[i]

# registers a new x-particle in the container
function Base.push!(pc::ParticleContainer, p::Particle)
    push!(pc.vals, p)
    push!(pc.logWs, 0.0)
    pc
end

# clones a theta-particle
function Base.copy(pc::ParticleContainer)
    # fork particles
    vals = eltype(pc.vals)[fork(p) for p in pc.vals]

    # copy weights
    logWs = copy(pc.logWs)

    ParticleContainer(vals, logWs)
end

"""
    reset_logweights!(pc::ParticleContainer)

Reset all unnormalized logarithmic weights to zero.
"""
function reset_logweights!(pc::ParticleContainer)
    fill!(pc.logWs, 0.0)
    return pc
end

"""
    increase_logweight!(pc::ParticleContainer, i::Int, x)

Increase the unnormalized logarithmic weight of the `i`th particle with `x`.
"""
function increase_logweight!(pc::ParticleContainer, i, logw)
    pc.logWs[i] += logw
    return pc
end

"""
    getweights(pc::ParticleContainer)

Compute the normalized weights of the particles.
"""
getweights(pc::ParticleContainer) = softmax(pc.logWs)

"""
    getweight(pc::ParticleContainer, i)

Compute the normalized weight of the `i`th particle.
"""
getweight(pc::ParticleContainer, i) = exp(pc.logWs[i] - logZ(pc))

"""
    logZ(pc::ParticleContainer)

Return the logarithm of the normalizing constant of the unnormalized logarithmic weights.
"""
logZ(pc::ParticleContainer) = logsumexp(pc.logWs)

"""
    effectiveSampleSize(pc::ParticleContainer)

Compute the effective sample size ``1 / ∑ wᵢ²``, where ``wᵢ```are the normalized weights.
"""
function effectiveSampleSize(pc::ParticleContainer)
    Ws = getweights(pc)
    return inv(sum(abs2, Ws))
end

"""
    resample_propagate!(pc::ParticleContainer[, randcat = resample_systematic, ref = nothing;
                        weights = getweights(pc)])

Resample and propagate the particles in `pc`.

Function `randcat` is used for sampling ancestor indices from the categorical distribution
of the particle `weights`. For Particle Gibbs sampling, one can provide a reference particle
`ref` that is ensured to survive the resampling step.
"""
function resample_propagate!(
    pc::ParticleContainer,
    randcat = Turing.Inference.resample_systematic,
    ref::Union{Particle, Nothing} = nothing;
    weights = getweights(pc)
)
    # check that weights are not NaN
    @assert !any(isnan, weights)

    # sample ancestor indices
    n = length(pc)
    nresamples = ref === nothing ? n : n - 1
    indx = randcat(weights, nresamples)

    # count number of children for each particle
    num_children = zeros(Int, n)
    @inbounds for i in indx
        num_children[i] += 1
    end

    # fork particles
    particles = collect(pc)
    children = similar(particles)
    j = 0
    @inbounds for i in 1:n
        ni = num_children[i]

        if ni > 0
            # fork first child
            pi = particles[i]
            isref = pi === ref
            p = isref ? fork(pi, isref) : pi
            children[j += 1] = p

            # fork additional children
            for _ in 2:ni
                children[j += 1] = fork(p, isref)
            end
        end
    end

    if ref !== nothing
        # Insert the retained particle. This is based on the replaying trick for efficiency
        # reasons. If we implement PG using task copying, we need to store Nx * T particles!
        @inbounds children[n] = ref
    end

    # replace particles and log weights in the container with new particles and weights
    pc.vals = children
    reset_logweights!(pc)

    pc
end

"""
    reweight!(pc::ParticleContainer)

Check if the final time step is reached, and otherwise reweight the particles by
considering the next observation.
"""
function reweight!(pc::ParticleContainer)
    n = length(pc)

    particles = collect(pc)
    numdone = 0
    for i in 1:n
        p = particles[i]

        # Obtain ``\\log p(yₜ | y₁, …, yₜ₋₁, x₁, …, xₜ, θ₁, …, θₜ)``, or `nothing` if the
        # the execution of the model is finished.
        # Here ``yᵢ`` are observations, ``xᵢ`` variables of the particle filter, and
        # ``θᵢ`` are variables of other samplers.
        score = Libtask.consume(p)

        if score === nothing
            numdone += 1
        else
            # Increase the unnormalized logarithmic weights, accounting for the variables
            # of other samplers.
            increase_logweight!(pc, i, score + getlogp(p.vi))

            # Reset the accumulator of the log probability in the model so that we can
            # accumulate log probabilities of variables of other samplers until the next
            # observation.
            resetlogp!(p.vi)
        end
    end

    # Check if all particles are propagated to the final time point.
    numdone == n && return true

    # The posterior for models with random number of observations is not well-defined.
    if numdone != 0
        error("mis-aligned execution traces: # particles = ", n,
              " # completed trajectories = ", numdone,
              ". Please make sure the number of observations is NOT random.")
    end

    return false
end

"""
    sweep!(pc::ParticleContainer, resampler)

Perform a particle sweep and return an unbiased estimate of the log evidence.

The resampling steps use the given `resampler`.

# Reference

Del Moral, P., Doucet, A., & Jasra, A. (2006). Sequential monte carlo samplers.
Journal of the Royal Statistical Society: Series B (Statistical Methodology), 68(3), 411-436.
"""
function sweep!(pc::ParticleContainer, resampler)
    # Initial step:

    # Resample and propagate particles.
    resample_propagate!(pc, resampler)

    # Compute the current normalizing constant ``Z₀`` of the unnormalized logarithmic
    # weights.
    # Usually it is equal to the number of particles in the beginning but this
    # implementation covers also the unlikely case of a particle container that is
    # initialized with non-zero logarithmic weights.
    logZ0 = logZ(pc)

    # Reweight the particles by including the first observation ``y₁``.
    isdone = reweight!(pc)

    # Compute the normalizing constant ``Z₁`` after reweighting.
    logZ1 = logZ(pc)

    # Compute the estimate of the log evidence ``\\log p(y₁)``.
    logevidence = logZ1 - logZ0

    # For observations ``y₂, …, yₜ``:
    while !isdone
        # Resample and propagate particles.
        resample_propagate!(pc, resampler)

        # Compute the current normalizing constant ``Z₀`` of the unnormalized logarithmic
        # weights.
        logZ0 = logZ(pc)

        # Reweight the particles by including the next observation ``yₜ``.
        isdone = reweight!(pc)

        # Compute the normalizing constant ``Z₁`` after reweighting.
        logZ1 = logZ(pc)

        # Compute the estimate of the log evidence ``\\log p(y₁, …, yₜ)``.
        logevidence += logZ1 - logZ0
    end

    return logevidence
end

struct ResampleWithESSThreshold{R, T<:Real}
    resampler::R
    threshold::T
end

function ResampleWithESSThreshold(resampler = Turing.Inference.resample_systematic)
    ResampleWithESSThreshold(resampler, 0.5)
end

function resample_propagate!(
    pc::ParticleContainer,
    resampler::ResampleWithESSThreshold,
    ref::Union{Particle,Nothing} = nothing;
    weights = getweights(pc)
)
    # Compute the effective sample size ``1 / ∑ wᵢ²`` with normalized weights ``wᵢ``
    ess = inv(sum(abs2, weights))

    if ess ≤ resampler.threshold * length(pc)
        resample_propagate!(pc, resampler.resampler, ref; weights = weights)
    end

    pc
end
