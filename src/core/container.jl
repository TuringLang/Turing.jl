mutable struct Trace{Tspl<:AbstractSampler, Tvi<:AbstractVarInfo, Tmodel<:Model}
    model::Tmodel
    spl::Tspl
    vi::Tvi
    task::Task

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
    res.task = copy(trace.task)
    return res
end

# NOTE: this function is called by `forkr`
function Trace(f::Function, m::Model, spl::AbstractSampler, vi::AbstractVarInfo)
    res = Trace{typeof(spl)}(m, spl, deepcopy(vi));
    # CTask(()->f());
    res.task = CTask( () -> begin res=f(); produce(Val{:done}); res; end )
    if res.task.storage === nothing
        res.task.storage = IdDict()
    end
    res.task.storage[:turing_trace] = res # create a backward reference in task_local_storage
    return res
end
function Trace(m::Model, spl::AbstractSampler, vi::AbstractVarInfo)
    res = Trace{typeof(spl)}(m, spl, deepcopy(vi));
    # CTask(()->f());
    res.vi.num_produce = 0
    res.task = CTask( () -> begin vi_new=m(vi, spl); produce(Val{:done}); vi_new; end )
    if res.task.storage === nothing
        res.task.storage = IdDict()
    end
    res.task.storage[:turing_trace] = res # create a backward reference in task_local_storage
    return res
end

# step to the next observe statement, return log likelihood
Libtask.consume(t::Trace) = (t.vi.num_produce += 1; consume(t.task))

# Task copying version of fork for Trace.
function fork(trace :: Trace, is_ref :: Bool = false)
    newtrace = copy(trace)
    is_ref && set_retained_vns_del_by_spl!(newtrace.vi, newtrace.spl)
    newtrace.task.storage[:turing_trace] = newtrace
    return newtrace
end

# PG requires keeping all randomness for the reference particle
# Create new task and copy randomness
function forkr(trace :: Trace)
    newtrace = Trace(trace.task.code, trace.model, trace.spl, deepcopy(trace.vi))
    newtrace.spl = trace.spl
    newtrace.vi.num_produce = 0
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
mutable struct ParticleContainer{T<:Particle, F}
    model::F
    vals::Vector{T}
    # logarithmic weights (Trace) or incremental log-likelihoods (ParticleContainer)
    logWs::Vector{Float64}
    # log model evidence
    logE::Float64
    # helpful for rejuvenation steps, e.g. in SMC2
    n_consume::Int
end

ParticleContainer(model, particles::Vector{<:Particle}) =
    ParticleContainer(model, particles, zeros(length(particles)), 0.0, 0)

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

    ParticleContainer(pc.model, vals, logWs, pc.logE, pc.n_consume)
end

# run particle filter for one step, return incremental likelihood
function Libtask.consume(pc :: ParticleContainer)
    # normalisation factor: 1/N
    z1 = logZ(pc)
    n = length(pc)

    particles = collect(pc)
    num_done = 0
    for i=1:n
        p = particles[i]
        score = Libtask.consume(p)
        if score isa Real
            score += getlogp(p.vi)
            resetlogp!(p.vi)
            increase_logweight(pc, i, Float64(score))
        elseif score == Val{:done}
            num_done += 1
        else
            error("[consume]: error in running particle filter.")
        end
    end

    if num_done == n
        res = Val{:done}
    elseif num_done != 0
        error("[consume]: mis-aligned execution traces, num_particles= $(n), num_done=$(num_done).")
    else
        # update incremental likelihoods
        z2 = logZ(pc)
        res = increase_logevidence(pc, z2 - z1)
        pc.n_consume += 1
        # res = increase_loglikelihood(pc, z2 - z1)
    end

    res
end

# compute the normalized weights
getweights(pc::ParticleContainer) = softmax(pc.logWs)

# compute the log-likelihood estimate, ignoring constant term ``- \log num_particles``
logZ(pc::ParticleContainer) = logsumexp(pc.logWs)

# compute the effective sample size ``1 / ∑ wᵢ²``, where ``wᵢ```are the normalized weights
function effectiveSampleSize(pc :: ParticleContainer)
    Ws = getweights(pc)
    return inv(sum(abs2, Ws))
end

increase_logweight(pc :: ParticleContainer, t :: Int, logw :: Float64) =
    (pc.logWs[t]  += logw)

increase_logevidence(pc :: ParticleContainer, logw :: Float64) =
    (pc.logE += logw)


function resample!(
    pc :: ParticleContainer,
    randcat :: Function = Turing.Inference.resample_systematic,
    ref :: Union{Particle, Nothing} = nothing;
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
    pc.logWs = zeros(n)

    pc
end

struct ResampleWithESSThreshold{R, T<:Real}
    resampler::R
    threshold::T
end

function ResampleWithESSThreshold()
    ResampleWithESSThreshold(Turing.Inference.resample_systematic)
end
ResampleWithESSThreshold(resampler) = ResampleWithESSThreshold(resampler, 0.5)

function resample!(
    pc::ParticleContainer,
    resampler::ResampleWithESSThreshold,
    ref::Union{Particle,Nothing} = nothing;
    weights = getweights(pc)
)
    # Compute the effective sample size ``1 / ∑ wᵢ²`` with normalized weights ``wᵢ``
    ess = inv(sum(abs2, weights))

    if ess ≤ resampler.threshold * length(pc)
        resample!(pc, resampler.resampler, ref; weights = weights)
    end

    pc
end
