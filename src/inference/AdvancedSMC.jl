###
### Particle Filtering and Particle MCMC Samplers.
###

####################
# Transition Types #
####################

# used by PG, SMC, PMMH
struct ParticleTransition{T} <: AbstractTransition
    θ::T
    lp::Float64
    le::Float64
    weight::Float64
end

abstract type ParticleInference <: InferenceAlgorithm end

transition_type(::Sampler{<:ParticleInference}) = ParticleTransition

function additional_parameters(::Type{<:ParticleTransition})
    return [:lp,:le,:weight]
end

####
#### Generic Sequential Monte Carlo sampler.
####

"""
    SMC(n_particles::Int)

Sequential Monte Carlo sampler.

Note that this method is particle-based, and arrays of variables
must be stored in a [`TArray`](@ref) object.

Usage:

```julia
SMC(1000)
```
"""
mutable struct SMC{space, F} <: ParticleInference
    n_particles           ::  Int
    resampler             ::  F
    resampler_threshold   ::  Float64
end

alg_str(spl::Sampler{SMC}) = "SMC"
function SMC(
        n_particles::Int,
        resampler::F,
        resampler_threshold::Float64,
        space::Tuple) where {F}
    return SMC{space, F}(n_particles, resampler, resampler_threshold)
end
SMC(n) = SMC(n, resample_systematic, 0.5, ())
SMC(n_particles::Int, ::Tuple{}) = SMC(n_particles)
function SMC(n_particles::Int, space::Symbol...)
    SMC(n_particles, resample_systematic, 0.5, space)
end

mutable struct ParticleState <: SamplerState
    logevidence        ::   Vector{Float64}
    vi                 ::   TypedVarInfo
    final_logevidence  ::   Float64
end

ParticleState(model::Model) = ParticleState(Float64[], VarInfo(model), 0.0)

function Sampler(alg::T, model::Model, s::Selector) where T<:SMC
    dict = Dict{Symbol, Any}()
    state = ParticleState(model)
    return Sampler{T,ParticleState}(alg, dict, s, state)
end

function step!(
    ::AbstractRNG, # Note: This function does not use the range argument.
    model::Turing.Model,
    spl::Sampler{<:SMC, ParticleState},
    ::Integer; # Note: This function doesn't use the N argument.
    kwargs...
)
    particles = ParticleContainer{Trace{typeof(spl),
        typeof(spl.state.vi), typeof(model)}}(model)

    spl.state.vi.num_produce = 0;  # Reset num_produce before new sweep\.
    set_retained_vns_del_by_spl!(spl.state.vi, spl)
    resetlogp!(spl.state.vi)

    push!(particles, spl.alg.n_particles, spl, empty!(spl.state.vi))

    while consume(particles) != Val{:done}
      ess = effectiveSampleSize(particles)
      if ess <= spl.alg.resampler_threshold * length(particles)
        resample!(particles,spl.alg.resampler)
      end
    end

    ## pick a particle to be retained.
    Ws, _ = weights(particles)
    indx = randcat(Ws)
    push!(spl.state.logevidence, particles.logE)

    # update the master vi.
    spl.state.vi = particles[indx].vi
    params = getparams(spl.state.vi, spl)
    lp = getlogp(spl.state.vi)

    return transition(params, lp, Ws[indx], particles.logE)
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
mutable struct PG{space, F} <: ParticleInference
  n_particles           ::    Int         # number of particles used
  resampler             ::    F           # function to resample
end
function PG(n_particles::Int, resampler::F, space::Tuple) where F
    return PG{space, F}(n_particles, resampler)
end
PG(n1::Int, ::Tuple{}) = PG(n1)
function PG(n1::Int, space::Symbol...)
    PG(n1, resample_systematic, space)
end

alg_str(spl::Sampler{PG}) = "PG"

const CSMC = PG # type alias of PG as Conditional SMC

"""
    Sampler(alg::PG, model::Model, s::Selector)

Return a `Sampler` object for the PG algorithm.
"""
function Sampler(alg::T, model::Model, s::Selector) where T<:PG
    info = Dict{Symbol, Any}()
    state = ParticleState(model)
    return Sampler{T,ParticleState}(alg, info, s, state)
end

function step!(
    ::AbstractRNG, # Note: This function does not use the range argument for now.
    model::Turing.Model,
    spl::Sampler{<:PG, ParticleState},
    ::Integer; # Note: This function doesn't use the N argument.
    kwargs...
)
    particles = ParticleContainer{Trace{typeof(spl), typeof(spl.state.vi), typeof(model)}}(model)

    spl.state.vi.num_produce = 0;  # Reset num_produce before new sweep\.
    ref_particle = isempty(spl.state.vi) ?
              nothing :
              forkr(Trace(model, spl, spl.state.vi))

    set_retained_vns_del_by_spl!(spl.state.vi, spl)
    resetlogp!(spl.state.vi)

    if ref_particle == nothing
        push!(particles, spl.alg.n_particles, spl, spl.state.vi)
    else
        push!(particles, spl.alg.n_particles-1, spl, spl.state.vi)
        push!(particles, ref_particle)
    end

    while consume(particles) != Val{:done}
        resample!(particles, spl.alg.resampler, ref_particle)
    end

    ## pick a particle to be retained.
    Ws, _ = weights(particles)
    indx = randcat(Ws)
    push!(spl.state.logevidence, particles.logE)

    # Extract the VarInfo from the retained particle.
    params = getparams(spl.state.vi, spl)
    spl.state.vi = particles[indx].vi
    lp = getlogp(spl.state.vi)

    # update the master vi.
    return transition(params, lp, Ws[indx], particles.logE)
end

function sample_end!(
    ::AbstractRNG,
    ::Model,
    spl::Sampler{<:ParticleInference},
    ::Integer,
    ::Vector{ParticleTransition};
    kwargs...
)
    # Set the default for resuming the sampler.
    resume_from = get(kwargs, :resume_from, nothing)

    # Exponentiate the average log evidence.
    loge = exp.(mean(spl.state.logevidence))

    # If we already had a chain, grab it's logevidence.
    if resume_from != nothing   # concat samples
        pushfirst!(samples, resume_from.info[:samples]...)
        pre_loge = exp.(resume_from.logevidence)
        # Calculate new log-evidence
        pre_n = length(resume_from.info[:samples])
        loge = (log(pre_loge) * pre_n + log(loge) * n) / (pre_n + n)
    else
        loge = log(loge)
    end

    # Store the logevidence.
    spl.state.final_logevidence = loge
end

function assume(  spl::Sampler{T},
                  dist::Distribution,
                  vn::VarName,
                  _::VarInfo
                ) where T<:Union{PG,SMC}

    vi = current_trace().vi
    if isempty(getspace(spl.alg)) || vn.sym in getspace(spl.alg)
        if ~haskey(vi, vn)
            r = rand(dist)
            push!(vi, vn, r, dist, spl)
        elseif is_flagged(vi, vn, "del")
            unset_flag!(vi, vn, "del")
            r = rand(dist)
            vi[vn] = vectorize(dist, r)
            setgid!(vi, spl.selector, vn)
            setorder!(vi, vn, vi.num_produce)
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
    return r, zero(Real)
end

function assume(  spl::Sampler{A},
                  dists::Vector{D},
                  vn::VarName,
                  var::Any,
                  vi::VarInfo
                ) where {A<:Union{PG,SMC},D<:Distribution}
    error("[Turing] PG and SMC doesn't support vectorizing assume statement")
end

function observe(spl::Sampler{T}, dist::Distribution, value, vi) where T<:Union{PG,SMC}
    produce(logpdf(dist, value))
    return zero(Real)
end

function observe( spl::Sampler{A},
                  ds::Vector{D},
                  value::Any,
                  vi::VarInfo
                ) where {A<:Union{PG,SMC},D<:Distribution}
    error("[Turing] PG and SMC doesn't support vectorizing observe statement")
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
function randcat(p::AbstractVector{T}) where T<:Real
    r, s = rand(T), 1
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

function resample_stratified(w::AbstractVector{<:Real}, num_particles::Integer)

    Q, N = cumsum(w), num_particles

    T = Array{Float64}(undef, N + 1)
    for i=1:N,
        T[i] = rand() / N + (i - 1) / N
    end
    T[N+1] = 1

    indx, i, j = Array{Int}(undef, N), 1, 1
    while i <= N
        if T[i] < Q[j]
            indx[i] = j
            i += 1
        else
            j += 1
        end
    end
    return indx
end

function resample_systematic(w::AbstractVector{<:Real}, num_particles::Integer)

    Q, N = cumsum(w), num_particles

    T = collect(range(0, stop = maximum(Q)-1/N, length = N)) .+ rand()/N
    push!(T, 1)

    indx, i, j = Array{Int}(undef, N), 1, 1
    while i <= N
        if T[i] < Q[j]
            indx[i] = j
            i += 1
        else
            j += 1
        end
    end
    return indx
end


#############################
# Common particle functions #
#############################

vnames(vi::VarInfo) = Symbol.(collect(keys(vi)))

"""
    transition(vi::AbstractVarInfo, spl::Sampler{<:Union{SMC, PG}}, weight::Float64)

Returns a basic TransitionType for the particle samplers.
"""
function transition(
        theta::T,
        lp::Float64,
        weight::Float64,
        le::Float64
) where {T}
    return ParticleTransition{T}(theta, lp, weight, le)
end
