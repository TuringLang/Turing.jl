###
### Particle Filtering and Particle MCMC Samplers.
###

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
mutable struct SMC{space, F} <: ParticleInference
    resampler             ::  F
    resampler_threshold   ::  Float64
end

alg_str(spl::Sampler{SMC}) = "SMC"
function SMC(
    resampler::F,
    resampler_threshold::Float64,
    space::Tuple
) where {F}
    return SMC{space, F}(resampler, resampler_threshold)
end
SMC() = SMC(resample_systematic, 0.5, ())
SMC(::Tuple{}) = SMC()
function SMC(space::Symbol...)
    SMC(resample_systematic, 0.5, space)
end

mutable struct SMCState <: AbstractSamplerState
    vi                   ::   TypedVarInfo
    # The logevidence after aggregating all samples together.
    average_logevidence  ::   Float64
    particles            ::   ParticleContainer
end

function SMCState(
    model::M, 
) where {
    M<:Model
}
    vi = VarInfo(model)
    particles = ParticleContainer{Trace}(model)

    return SMCState(vi, 0.0, particles)
end

function Sampler(alg::T, model::Model, s::Selector) where T<:SMC
    dict = Dict{Symbol, Any}()
    state = SMCState(model)
    return Sampler{T,SMCState}(alg, dict, s, state)
end

function sample_init!(
    ::AbstractRNG, # Note: This function does not use the range argument.
    model::Turing.Model,
    spl::Sampler{<:SMC},
    N::Integer; # Note: This function doesn't use the N argument.
    kwargs...
)
    # Update the particle container now that the sampler type
    # is defined.
    spl.state.particles = ParticleContainer{Trace{typeof(spl),
        typeof(spl.state.vi), typeof(model)}}(model)

    spl.state.vi.num_produce = 0;  # Reset num_produce before new sweep\.
    set_retained_vns_del_by_spl!(spl.state.vi, spl)
    resetlogp!(spl.state.vi)

    push!(spl.state.particles, N, spl, empty!(spl.state.vi))

    while consume(spl.state.particles) != Val{:done}
        ess = effectiveSampleSize(spl.state.particles)
        if ess <= spl.alg.resampler_threshold * length(spl.state.particles)
            resample!(spl.state.particles, spl.alg.resampler)
        end
    end
end

function step!(
    ::AbstractRNG, 
    model::Turing.Model,
    spl::Sampler{<:SMC},
    ::Integer;
    iteration=-1,
    kwargs...
)
    ## Grab the weights.
    Ws, _ = weights(spl.state.particles)

    # update the master vi.
    particle = spl.state.particles.vals[iteration]
    params = tonamedtuple(particle.vi)
    lp = getlogp(particle.vi)

    return transition(params, lp, spl.state.particles.logE, Ws[iteration])
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

mutable struct PGState <: AbstractSamplerState
    vi                   ::   TypedVarInfo
    # The logevidence after aggregating all samples together.
    average_logevidence  ::   Float64
end

function PGState(model::M) where {M<:Model}
    vi = VarInfo(model)
    return PGState(vi, 0.0)
end

const CSMC = PG # type alias of PG as Conditional SMC

"""
    Sampler(alg::PG, model::Model, s::Selector)

Return a `Sampler` object for the PG algorithm.
"""
function Sampler(alg::T, model::Model, s::Selector) where T<:PG
    info = Dict{Symbol, Any}()
    state = PGState(model)
    return Sampler{T,PGState}(alg, info, s, state)
end

function step!(
    ::AbstractRNG, # Note: This function does not use the range argument for now.
    model::Turing.Model,
    spl::Sampler{<:PG},
    ::Integer; # Note: This function doesn't use the N argument.
    kwargs...
)
    particles = ParticleContainer{Trace{typeof(spl), typeof(spl.state.vi), typeof(model)}}(model)

    spl.state.vi.num_produce = 0;  # Reset num_produce before new sweep.
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

    # Extract the VarInfo from the retained particle.
    params = tonamedtuple(spl.state.vi)
    spl.state.vi = particles[indx].vi
    lp = getlogp(spl.state.vi)

    # update the master vi.
    return transition(params, lp, particles.logE, 1.0)
end

function sample_end!(
    ::AbstractRNG,
    ::Model,
    spl::Sampler{<:ParticleInference},
    N::Integer,
    ts::Vector{ParticleTransition};
    kwargs...
)
    # Set the default for resuming the sampler.
    resume_from = get(kwargs, :resume_from, nothing)

    # Exponentiate the average log evidence.
    loge = exp(mean([t.le for t in ts]))

    # If we already had a chain, grab the logevidence.
    if resume_from != nothing   # concat samples
        @assert resume_from isa Chains "resume_from needs to be a Chains object."
        # pushfirst!(samples, resume_from.info[:samples]...)
        pre_loge = exp.(resume_from.logevidence)
        # Calculate new log-evidence
        pre_n = length(resume_from)
        loge = (log(pre_loge) * pre_n + log(loge) * N) / (pre_n + N)
    else
        loge = log(loge)
    end

    # Store the logevidence.
    spl.state.average_logevidence = loge
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
