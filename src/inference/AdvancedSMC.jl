###
### Particle Filtering and Particle MCMC Samplers.
###

####################
# Transition Types #
####################

# used by PG, SMC, PMMH
struct ParticleTransition{T} <: AbstractTransition
    θ::Vector{T}
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
struct SMC{T, F} <: ParticleInference
    n_particles           ::  Int
    resampler             ::  F
    resampler_threshold   ::  Float64
    space                 ::  Set{T}
end

alg_str(spl::Sampler{SMC}) = "SMC"

SMC(n) = SMC(n, resample_systematic, 0.5, Set())
function SMC(n_particles::Int, space...)
    _space = isa(space, Symbol) ? Set([space]) : Set(space)
    return SMC(n_particles, resample_systematic, 0.5, _space)
end

mutable struct ParticleState <: SamplerState
    logevidence        ::   Vector{Float64}
    vi                 ::   TypedVarInfo
    final_logevidence  ::   Float64
end

ParticleState(model::Model) = ParticleState(Float64[], VarInfo(model), 0.0)

function Sampler(alg::SMC, model::Model, s::Selector)
    dict = Dict{Symbol, Any}()
    state = ParticleState(model)
    return Sampler{SMC,ParticleState}(alg, dict, s, state)
end

function step!(
    ::AbstractRNG, # Note: This function does not use the range argument.
    model::Turing.Model,
    spl::Sampler{SMC, ParticleState},
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

    params = particles[indx].vi[spl]
    vi_draw = particles[indx].vi
    lp = getlogp(vi_draw)

    # update the master vi.
    return transition(vi_draw[spl], lp, Ws[indx], particles.logE)
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
struct PG{T, F} <: ParticleInference
    n_particles           ::    Int         # number of particles used
    resampler             ::    F           # function to resample
    space                 ::    Set{T}      # sampling space, emtpy means all
end

PG(n1::Int) = PG(n1, resample_systematic, Set())
function PG(n1::Int, space...)
    _space = isa(space, Symbol) ? Set([space]) : Set(space)
    alg = PG(n1, resample_systematic, _space)
    return alg
end

alg_str(spl::Sampler{PG}) = "PG"

const CSMC = PG # type alias of PG as Conditional SMC

"""
    Sampler(alg::PG, model::Model, s::Selector)

Return a `Sampler` object for the PG algorithm.
"""
function Sampler(alg::PG, model::Model, s::Selector)
    info = Dict{Symbol, Any}()
    state = ParticleState(model)
    return Sampler{PG,ParticleState}(alg, info, s, state)
end

function step!(
    ::AbstractRNG, # Note: This function does not use the range argument for now.
    model::Turing.Model,
    spl::Sampler{PG, ParticleState},
    ::Integer; # Note: This function doesn't use the N argument.
    kwargs...
)
    particles = ParticleContainer{Trace{typeof(spl),
        typeof(spl.state.vi), typeof(model)}}(model)

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
    vi_draw = particles[indx].vi
    lp = getlogp(vi_draw)

    return transition(vi_draw[spl], lp, Ws[indx], particles.logE)
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
    if isempty(spl.alg.space) || vn.sym in spl.alg.space
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
#### Particle marginal Metropolis-Hastings sampler.
####

"""
    PMMH(smc_alg:::SMC, parameters_algs::Tuple{MH})

Particle independant Metropolis–Hastings and
Particle marginal Metropolis–Hastings samplers.

Note that this method is particle-based, and arrays of variables
must be stored in a [`TArray`](@ref) object.

Usage:

```julia
alg = PMMH(SMC(20, :v1), MH(1,:v2))
alg = PMMH(SMC(20, :v1), MH(1,(:v2, (x) -> Normal(x, 1))))
```

Arguments:

- `smc_alg:::SMC` : An [`SMC`](@ref) algorithm to use.
- `parameters_algs::Tuple{MH}` : An [`MH`](@ref) algorithm, which includes a
sample space specification.
"""
struct PMMH{T, A<:Tuple} <: InferenceAlgorithm
    algs                  ::    A                 # Proposals for state & parameters
    space                 ::    Set{T}            # sampling space, emtpy means all
end

mutable struct PMMHState <: SamplerState
    samplers                :: Array{Sampler} # NOTE: May need parametric types?
    old_prior_prob          :: Float64
    old_likelihood_estimate :: Float64
    vi                      :: VarInfo
end

PMMHState() = PMMHState(Sampler[], 0.0, -Inf, VarInfo())

function PMMH(smc_alg::SMC, parameter_algs...)
    return PMMH(tuple(parameter_algs..., smc_alg), Set())
end

PIMH(smc_alg::SMC) = PMMH(tuple(smc_alg), Set())
alg_str(spl::Sampler{PMMH}) = "PMMH"

function Sampler(alg::PMMH, model::Model, s::Selector)
    info = Dict{Symbol, Any}()
    spl = Sampler(alg, info, s, PMMHState())

    alg_str = "PMMH"
    n_samplers = length(alg.algs)
    samplers = Array{Sampler}(undef, n_samplers)

    space = Set{Symbol}()

    for i in 1:n_samplers
        sub_alg = alg.algs[i]
        if isa(sub_alg, Union{SMC, MH})
            samplers[i] = Sampler(sub_alg, model, Selector(Symbol(typeof(sub_alg))))
        else
            error("[$alg_str] unsupport base sampling algorithm $alg")
        end
        if typeof(sub_alg) == MH && sub_alg.n_iters != 1
            warn("[$alg_str] number of iterations greater than 1 is useless for MH since it is only used for its proposal")
        end
        space = union(space, sub_alg.space)
    end

    # Sanity check for space
    if !isempty(space)
        @assert issubset(Set(get_pvars(model)), space) "[$alg_str] symbols specified to samplers ($space) doesn't cover the model parameters ($(Set(get_pvars(model))))"

        if Set(get_pvars(model)) != space
            warn("[$alg_str] extra parameters specified by samplers don't exist in model: $(setdiff(space, Set(get_pvars(model))))")
        end
    end

    # Set the samplers.
    spl.state.samplers = samplers

    return spl
end

function step!(
    ::AbstractRNG, # Note: This function does not use the range argument for now.
    model::Turing.Model,
    spl::Sampler{PMMH, PMMHState},
    ::Integer; # Note: This function doesn't use the N argument.
    kwargs...
)
    violating_support = false
    proposal_ratio = 0.0
    new_prior_prob = 0.0
    new_likelihood_estimate = 0.0
    old_θ = copy(vi[spl])

    Turing.DEBUG && @debug "Propose new parameters from proposals..."
    for local_spl in spl.state.samplers[1:end-1]
        Turing.DEBUG && @debug "$(typeof(local_spl)) proposing $(local_spl.alg.space)..."
        propose(model, local_spl, spl.state.vi)
        if local_spl.state.violating_support
            violating_support=true
            break
        end
        new_prior_prob += local_spl.state.prior_prob
        proposal_ratio += local_spl.state.proposal_ratio
    end

    if !violating_support # do not run SMC if going to refuse anyway
        Turing.DEBUG && @debug "Propose new state with SMC..."
        spl.state.vi, _ = step(model, spl.state.samplers[end], spl.state.vi)
        new_likelihood_estimate = spl.state.samplers[end].state.logevidence[end]

        Turing.DEBUG && @debug "computing accept rate α..."
        is_accept, _ = mh_accept(
          -(spl.state.old_likelihood_estimate + spl.state.old_prior_prob),
          -(new_likelihood_estimate + new_prior_prob),
          proposal_ratio,
        )
    end

    Turing.DEBUG && @debug "decide whether to accept..."
    if !violating_support && is_accept # accepted
        is_accept = true
        spl.state.old_likelihood_estimate = new_likelihood_estimate
        spl.state.old_prior_prob = new_prior_prob
    else                      # rejected
        is_accept = false
        spl.state.vi[spl] = old_θ
    end

    return transition(spl.state.vi, spl)
end

# function step(model, spl::Sampler{<:PMMH}, vi::VarInfo, is_first::Bool)
#     violating_support = false
#     proposal_ratio = 0.0
#     new_prior_prob = 0.0
#     new_likelihood_estimate = 0.0
#     old_θ = copy(vi[spl])
#
#     Turing.DEBUG && @debug "Propose new parameters from proposals..."
#     for local_spl in spl.info[:samplers][1:end-1]
#         Turing.DEBUG && @debug "$(typeof(local_spl)) proposing $(local_spl.alg.space)..."
#         propose(model, local_spl, vi)
#         if local_spl.info[:violating_support] violating_support=true; break end
#         new_prior_prob += local_spl.info[:prior_prob]
#         proposal_ratio += local_spl.info[:proposal_ratio]
#     end
#
#     if !violating_support # do not run SMC if going to refuse anyway
#         Turing.DEBUG && @debug "Propose new state with SMC..."
#         vi, _ = step(model, spl.info[:samplers][end], vi)
#         new_likelihood_estimate = spl.info[:samplers][end].info[:logevidence][end]
#
#         Turing.DEBUG && @debug "computing accept rate α..."
#         is_accept, _ = mh_accept(
#           -(spl.info[:old_likelihood_estimate] + spl.info[:old_prior_prob]),
#           -(new_likelihood_estimate + new_prior_prob),
#           proposal_ratio,
#         )
#     end
#
#     Turing.DEBUG && @debug "decide whether to accept..."
#     if !violating_support && is_accept # accepted
#         is_accept = true
#         spl.info[:old_likelihood_estimate] = new_likelihood_estimate
#         spl.info[:old_prior_prob] = new_prior_prob
#     else                      # rejected
#         is_accept = false
#         vi[spl] = old_θ
#     end
#
#     return vi, is_accept
# end

function sample(  model::Model,
                  alg::PMMH;
                  save_state=false,         # flag for state saving
                  resume_from=nothing,      # chain to continue
                  reuse_spl_n=0             # flag for spl re-using
                )

    spl = Sampler(alg, model)
    if resume_from != nothing
        spl.selector = resume_from.info[:spl].selector
    end
    alg_str = "PMMH"

    # Number of samples to store
    sample_n = spl.alg.n_iters

    # Init samples
    time_total = zero(Float64)
    samples = Array{Sample}(undef, sample_n)
    weight = 1 / sample_n
    for i = 1:sample_n
        samples[i] = Sample(weight, Dict{Symbol, Any}())
    end

    # Init parameters
    vi = if resume_from == nothing
        vi_ = VarInfo(model)
    else
        resume_from.info[:vi]
    end
    n = spl.alg.n_iters

    # PMMH steps
    accept_his = Bool[]
    PROGRESS[] && (spl.info[:progress] = ProgressMeter.Progress(n, 1, "[$alg_str] Sampling...", 0))
    for i = 1:n
      Turing.DEBUG && @debug "$alg_str stepping..."
      time_elapsed = @elapsed vi, is_accept = step(model, spl, vi, i==1)

      if is_accept # accepted => store the new predcits
          samples[i].value = Sample(vi, spl).value
      else         # rejected => store the previous predcits
          samples[i] = samples[i - 1]
      end

      time_total += time_elapsed
      push!(accept_his, is_accept)
      if PROGRESS[]
        haskey(spl.info, :progress) && ProgressMeter.update!(spl.info[:progress], spl.info[:progress].counter + 1)
      end
    end

    println("[$alg_str] Finished with")
    println("  Running time    = $time_total;")
    accept_rate = sum(accept_his) / n  # calculate the accept rate
    println("  Accept rate         = $accept_rate;")

    if resume_from != nothing   # concat samples
      pushfirst!(samples, resume_from.info[:samples]...)
    end
    c = Chain(-Inf, samples)       # wrap the result by Chain

    if save_state               # save state
      c = save(c, spl, model, vi, samples)
    end

    c
end


####
#### IMCMC Sampler.
####

"""
    IPMCMC(n_particles::Int, n_iters::Int, n_nodes::Int, n_csmc_nodes::Int)

Particle Gibbs sampler.

Note that this method is particle-based, and arrays of variables
must be stored in a [`TArray`](@ref) object.

Usage:

```julia
IPMCMC(100, 100, 4, 2)
```

Arguments:

- `n_particles::Int` : Number of particles to use.
- `n_iters::Int` : Number of iterations to employ.
- `n_nodes::Int` : The number of nodes running SMC and CSMC.
- `n_csmc_nodes::Int` : The number of CSMC nodes.
```

A paper on this can be found [here](https://arxiv.org/abs/1602.05128).
"""
mutable struct IPMCMC{T, F} <: InferenceAlgorithm
  n_particles           ::    Int         # number of particles used
  n_iters               ::    Int         # number of iterations
  n_nodes               ::    Int         # number of nodes running SMC and CSMC
  n_csmc_nodes          ::    Int         # number of nodes CSMC
  resampler             ::    F           # function to resample
  space                 ::    Set{T}      # sampling space, emtpy means all
end
IPMCMC(n1::Int, n2::Int) = IPMCMC(n1, n2, 32, 16, resample_systematic, Set())
IPMCMC(n1::Int, n2::Int, n3::Int) = IPMCMC(n1, n2, n3, Int(ceil(n3/2)), resample_systematic, Set())
IPMCMC(n1::Int, n2::Int, n3::Int, n4::Int) = IPMCMC(n1, n2, n3, n4, resample_systematic, Set())
function IPMCMC(n1::Int, n2::Int, n3::Int, n4::Int, space...)
  _space = isa(space, Symbol) ? Set([space]) : Set(space)
  IPMCMC(n1, n2, n3, n4, resample_systematic, _space)
end

function Sampler(alg::IPMCMC, s::Selector)
  info = Dict{Symbol, Any}()
  spl = Sampler(alg, info, s)
  # Create SMC and CSMC nodes
  samplers = Array{Sampler}(undef, alg.n_nodes)
  # Use resampler_threshold=1.0 for SMC since adaptive resampling is invalid in this setting
  default_CSMC = CSMC(alg.n_particles, 1, alg.resampler, alg.space)
  default_SMC = SMC(alg.n_particles, alg.resampler, 1.0, false, alg.space)

  for i in 1:alg.n_csmc_nodes
    samplers[i] = Sampler(default_CSMC, Selector(Symbol(typeof(default_CSMC))))
  end
  for i in (alg.n_csmc_nodes+1):alg.n_nodes
    samplers[i] = Sampler(default_SMC, Selector(Symbol(typeof(default_CSMC))))
  end

  info[:samplers] = samplers

  return spl
end

function step(model, spl::Sampler{<:IPMCMC}, VarInfos::Array{VarInfo}, is_first::Bool)
    # Initialise array for marginal likelihood estimators
    log_zs = zeros(spl.alg.n_nodes)

    # Run SMC & CSMC nodes
    for j in 1:spl.alg.n_nodes
        VarInfos[j].num_produce = 0
        VarInfos[j] = step(model, spl.info[:samplers][j], VarInfos[j])[1]
        log_zs[j] = spl.info[:samplers][j].info[:logevidence][end]
    end

    # Resampling of CSMC nodes indices
    conditonal_nodes_indices = collect(1:spl.alg.n_csmc_nodes)
    unconditonal_nodes_indices = collect(spl.alg.n_csmc_nodes+1:spl.alg.n_nodes)
    for j in 1:spl.alg.n_csmc_nodes
        # Select a new conditional node by simulating cj
        log_ksi = vcat(log_zs[unconditonal_nodes_indices], log_zs[j])
        ksi = exp.(log_ksi .- maximum(log_ksi))
        c_j = wsample(ksi) # sample from Categorical with unormalized weights

        if c_j < length(log_ksi) # if CSMC node selects another index than itself
            conditonal_nodes_indices[j] = unconditonal_nodes_indices[c_j]
            unconditonal_nodes_indices[c_j] = j
        end
    end
    nodes_permutation = vcat(conditonal_nodes_indices, unconditonal_nodes_indices)

    VarInfos[nodes_permutation]
end

function sample(model::Model, alg::IPMCMC)

  spl = Sampler(alg)

  # Number of samples to store
  sample_n = alg.n_iters * alg.n_csmc_nodes

  # Init samples
  time_total = zero(Float64)
  samples = Array{Sample}(undef, sample_n)
  weight = 1 / sample_n
  for i = 1:sample_n
    samples[i] = Sample(weight, Dict{Symbol, Any}())
  end

  # Init parameters
  vi = empty!(VarInfo(model))
  VarInfos = Array{VarInfo}(undef, spl.alg.n_nodes)
  for j in 1:spl.alg.n_nodes
    VarInfos[j] = deepcopy(vi)
  end
  n = spl.alg.n_iters

  # IPMCMC steps
  if PROGRESS[] spl.info[:progress] = ProgressMeter.Progress(n, 1, "[IPMCMC] Sampling...", 0) end
  for i = 1:n
    Turing.DEBUG && @debug "IPMCMC stepping..."
    time_elapsed = @elapsed VarInfos = step(model, spl, VarInfos, i==1)

    # Save each CSMS retained path as a sample
    for j in 1:spl.alg.n_csmc_nodes
      samples[(i-1)*alg.n_csmc_nodes+j].value = Sample(VarInfos[j], spl).value
    end

    time_total += time_elapsed
    if PROGRESS[]
      haskey(spl.info, :progress) && ProgressMeter.update!(spl.info[:progress], spl.info[:progress].counter + 1)
    end
  end

  println("[IPMCMC] Finished with")
  println("  Running time    = $time_total;")

  Chain(0.0, samples) # wrap the result by Chain
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
        theta::Vector{T},
        lp::Float64,
        weight::Float64,
        le::Float64
) where {T<:Real}
    return ParticleTransition{T}(theta, lp, weight, le)
end
