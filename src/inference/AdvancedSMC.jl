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

####
#### Generic Sequential Monte Carlo sampler.
####

"""
    SMC()

Sequential Monte Carlo sampler.

Note that this method is particle-based, and arrays of variables
must be stored in a [`TArray`](@ref) object.

Usage:

```julia
SMC()
```
"""
struct SMC{space, R} <: ParticleInference
    resampler::R
end

function SMC(resampler = Turing.Core.ResampleWithESSThreshold(), space::Tuple = ())
    SMC{space, typeof(resampler)}(resampler)
end
SMC(::Tuple{}) = SMC()
SMC(space::Symbol...) = SMC(space)

mutable struct SMCState{V<:VarInfo, F<:AbstractFloat} <: AbstractSamplerState
    vi                   ::   V
    # The logevidence after aggregating all samples together.
    average_logevidence  ::   F
    particles            ::   ParticleContainer
end

function SMCState(model::Model)
    vi = VarInfo(model)
    particles = ParticleContainer(model, Trace[])

    return SMCState(vi, 0.0, particles)
end

function Sampler(alg::SMC, model::Model, s::Selector)
    dict = Dict{Symbol, Any}()
    state = SMCState(model)
    return Sampler(alg, dict, s, state)
end

function AbstractMCMC.sample_init!(
    ::AbstractRNG,
    model::Model,
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
    spl.state.particles = pc = ParticleContainer(model, particles)

    while consume(pc) !== Val{:done}
        resample!(pc, spl.alg.resampler)
    end
end

function AbstractMCMC.step!(
    ::AbstractRNG,
    model::Model,
    spl::Sampler{<:SMC},
    ::Integer,
    transition;
    iteration=-1,
    kwargs...
)
    # check that we received a real iteration number
    @assert iteration >= 1 "step! needs to be called with an 'iteration' keyword argument."

    # grab the weights
    pc = spl.state.particles
    Ws = getweights(pc)

    # update the master vi
    particle = pc.vals[iteration]
    params = tonamedtuple(particle.vi)
    lp = getlogp(particle.vi)

    return ParticleTransition(params, lp, pc.logE, Ws[iteration])
end

####
#### Particle Gibbs sampler.
####

"""
    PG(n_particles::Int)

Particle Gibbs sampler.

Note that this method is particle-based, and arrays of variables
must be stored in a [`TArray`](@ref) object.

Usage:

```julia
PG(100, 100)
```
"""
struct PG{space} <: ParticleInference
  n_particles           ::    Int         # number of particles used
  resampler             ::    Function    # function to resample
end
function PG(n_particles::Int, resampler::Function, space::Tuple)
    return PG{space}(n_particles, resampler)
end
PG(n1::Int, ::Tuple{}) = PG(n1)
function PG(n1::Int, space::Symbol...)
    PG(n1, resample_systematic, space)
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
    model::Model,
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
    num_particles = spl.alg.n_particles
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
    pc = ParticleContainer(model, particles)

    # run the particle filter
    while consume(pc) !== Val{:done}
        resample!(pc, spl.alg.resampler, ref_particle)
    end

    # pick a particle to be retained.
    Ws = getweights(pc)
    indx = randcat(Ws)

    # extract the VarInfo from the retained particle.
    params = tonamedtuple(vi)
    spl.state.vi = pc.vals[indx].vi
    lp = getlogp(spl.state.vi)

    # update the master vi.
    return ParticleTransition(params, lp, pc.logE, 1.0)
end

function AbstractMCMC.sample_end!(
    ::AbstractRNG,
    ::Model,
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
    if resume_from isa Chains
        # pushfirst!(samples, resume_from.info[:samples]...)
        pre_loge = resume_from.logevidence
        # Calculate new log-evidence
        pre_n = length(resume_from)
        loge = (pre_loge * pre_n + loge * N) / (pre_n + N)
    elseif resume_from !== nothing
        error("keyword argument `resume_from` has to be `nothing` or a `Chains` object")
    end

    # Store the logevidence.
    spl.state.average_logevidence = loge
end

function assume(spl::Sampler{<:Union{PG,SMC}}, dist::Distribution, vn::VarName, ::VarInfo)
    vi = current_trace().vi
    if vn in getspace(spl)
        if ~haskey(vi, vn)
            r = rand(dist)
            push!(vi, vn, r, dist, spl)
        elseif is_flagged(vi, vn, "del")
            unset_flag!(vi, vn, "del")
            r = rand(dist)
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
            r = rand(dist)
            push!(vi, vn, r, dist, Selector(:invalid))
        end
        acclogp!(vi, logpdf_with_trans(dist, r, istrans(vi, vn)))
    end
    return r, 0
end

function observe(spl::Sampler{<:Union{PG,SMC}}, dist::Distribution, value, vi)
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

    M = length(w)

    # "Repetition counts" (plus the random part, later on):
    Ns = floor.(length(w) .* w)

    # The "remainder" or "residual" count:
    R = Int(sum(Ns))

    # The number of particles which will be drawn stocastically:
    M_rdn = num_particles - R

    # The modified weights:
    Ws = (M .* w - floor.(M .* w)) / M_rdn

    # Draw the deterministic part:
    indx1, i = Array{Int}(undef, R), 1
    for j in 1:M
        for k in 1:Ns[j]
            indx1[i] = j
            i += 1
        end
    end

    # And now draw the stocastic (Multinomial) part:
    return append!(indx1, rand(Distributions.sampler(Categorical(w)), M_rdn))
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
