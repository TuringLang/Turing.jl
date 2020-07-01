###
### Particle Filtering and Particle MCMC Samplers.
###

#######################
# Particle Transition #
#######################

"""
    ParticleTransition{T, F<:AbstractFloat}

Fields:
- `θ`: The parameters for any given sample.
- `lp`: The log pdf for the sample's parameters.
- `le`: The log evidence retrieved from the particle.
- `weight`: The weight of the particle the sample was retrieved from.
"""
struct ParticleTransition{T, F<:AbstractFloat}
    θ::T
    lp::F
    le::F
    weight::F
end

function additional_parameters(::Type{<:ParticleTransition})
    return [:lp,:le, :weight]
end

DynamicPPL.getlogp(t::ParticleTransition) = t.lp

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

mutable struct SMCState{V<:VarInfo, F<:AbstractFloat} <: AbstractSamplerState
    vi                   ::   V
    # The logevidence after aggregating all samples together.
    average_logevidence  ::   F
    particles            ::   ParticleContainer
end

function SMCState(model::Model)
    vi = VarInfo(model)
    particles = ParticleContainer(Trace[])

    return SMCState(vi, 0.0, particles)
end

function Sampler(alg::SMC, model::Model, s::Selector)
    dict = Dict{Symbol, Any}()
    state = SMCState(model)
    return Sampler(alg, dict, s, state)
end

function AbstractMCMC.sample_init!(
    ::AbstractRNG,
    model::AbstractModel,
    spl::Sampler{<:SMC},
    N::Integer;
    kwargs...
)
    # set the parameters to a starting value
    initialize_parameters!(spl; kwargs...)

    # reset the VarInfo
    vi = spl.state.vi
    reset_num_produce!(vi)
    set_retained_vns_del_by_spl!(vi, spl)
    resetlogp!(vi)
    empty!(vi)

    # create a new set of particles
    T = Trace{typeof(spl),typeof(vi),typeof(model)}
    particles = T[Trace(model, spl, vi) for _ in 1:N]

    # create a new particle container
    spl.state.particles = pc = ParticleContainer(particles)

    # Perform particle sweep.
    logevidence = sweep!(pc, spl.alg.resampler)
    spl.state.average_logevidence = logevidence

    return
end

function AbstractMCMC.step!(
    ::AbstractRNG,
    model::AbstractModel,
    spl::Sampler{<:SMC},
    ::Integer,
    transition;
    iteration=-1,
    kwargs...
)
    # check that we received a real iteration number
    @assert iteration >= 1 "step! needs to be called with an 'iteration' keyword argument."

    # grab the weight
    pc = spl.state.particles
    weight = getweight(pc, iteration)

    # update the master vi
    particle = pc.vals[iteration]
    params = tonamedtuple(particle.vi)

    # This is pretty useless since we reset the log probability continuously in the
    # particle sweep.
    lp = getlogp(particle.vi)

    return ParticleTransition(params, lp, spl.state.average_logevidence, weight)
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

mutable struct PGState{V<:VarInfo, F<:AbstractFloat} <: AbstractSamplerState
    vi                   ::   V
    # The logevidence after aggregating all samples together.
    average_logevidence  ::   F
end

function PGState(model::Model)
    vi = VarInfo(model)
    return PGState(vi, 0.0)
end

const CSMC = PG # type alias of PG as Conditional SMC

"""
    Sampler(alg::PG, model::Model, s::Selector)

Return a `Sampler` object for the PG algorithm.
"""
function Sampler(alg::PG, model::Model, s::Selector)
    info = Dict{Symbol, Any}()
    state = PGState(model)
    return Sampler(alg, info, s, state)
end

function AbstractMCMC.step!(
    ::AbstractRNG,
    model::AbstractModel,
    spl::Sampler{<:PG},
    ::Integer,
    transition;
    kwargs...
)
    # obtain or create reference particle
    vi = spl.state.vi
    ref_particle = isempty(vi) ? nothing : forkr(Trace(model, spl, vi))

    # reset the VarInfo before new sweep
    reset_num_produce!(vi)
    set_retained_vns_del_by_spl!(vi, spl)
    resetlogp!(vi)

    # create a new set of particles
    num_particles = spl.alg.nparticles
    T = Trace{typeof(spl),typeof(vi),typeof(model)}
    if ref_particle === nothing
        particles = T[Trace(model, spl, vi) for _ in 1:num_particles]
    else
        particles = Vector{T}(undef, num_particles)
        @inbounds for i in 1:(num_particles - 1)
            particles[i] = Trace(model, spl, vi)
        end
        @inbounds particles[num_particles] = ref_particle
    end

    # create a new particle container
    pc = ParticleContainer(particles)

    # Perform a particle sweep.
    logevidence = sweep!(pc, spl.alg.resampler)

    # pick a particle to be retained.
    Ws = getweights(pc)
    indx = randcat(Ws)

    # extract the VarInfo from the retained particle.
    params = tonamedtuple(vi)
    spl.state.vi = pc.vals[indx].vi

    # This is pretty useless since we reset the log probability continuously in the
    # particle sweep.
    lp = getlogp(spl.state.vi)

    # update the master vi.
    return ParticleTransition(params, lp, logevidence, 1.0)
end

function AbstractMCMC.sample_end!(
    ::AbstractRNG,
    ::AbstractModel,
    spl::Sampler{<:ParticleInference},
    N::Integer,
    ts::Vector{<:ParticleTransition};
    resume_from = nothing,
    kwargs...
)
    # Exponentiate the average log evidence.
    # loge = exp(mean([t.le for t in ts]))
    loge = mean(t.le for t in ts)

    # If we already had a chain, grab the logevidence.
    if resume_from isa MCMCChains.Chains
        # pushfirst!(samples, resume_from.info[:samples]...)
        pre_loge = resume_from.logevidence
        # Calculate new log-evidence
        pre_n = length(resume_from)
        loge = (pre_loge * pre_n + loge * N) / (pre_n + N)
    elseif resume_from !== nothing
        error("keyword argument `resume_from` has to be `nothing` or a `MCMCChains.Chains` object")
    end

    # Store the logevidence.
    spl.state.average_logevidence = loge
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
