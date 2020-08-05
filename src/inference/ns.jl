"""
------------------------------------
not working yet
still being built
refering ess.jl which is similar
-------------------------------------

Nested sampling algorithm.

An example
#Note-to-self: set a seed here
## write an example here

"""

## what all to setup here:
## 1. A subtype of AbstractSampler, defined as a mutable struct containing state information or sampler parameters... Partially done in  mutable struct NSState ?? check
## 2. A function sample_init! which performs any necessary set-up (default: do not perform any set-up)... Partially done in: function AbstractMCMC.sample_init! ?? check
## 3. A function step! which returns a transition that represents a single draw from the sampler... Partially done with: 2 function AbstractMCMC.step! ?? check
## 4. A function transitions_init which returns a container for the transitions obtained from the sampler (default: return a Vector{T} of length N where T is the type of the transition obtained in the first step and N is the number of requested samples).
## 5. A function transitions_save! which saves transitions to the container (default: save the transition of iteration i at position i in the vector of transitions)
## 6. A function sample_end! which handles any sampler wrap-up (default: do not perform any wrap-up)... Partially done in: function AbstractMCMC.sample_end! ?? check
## 7. A function bundle_samples which accepts the container of transitions and returns a collection of samples (default: return the vector of transitions)... Partially done with: 2 function AbstractMCMC.bundle_samples ?? check

## model:: Nestedmodel
## sampler:: Nested

## comment on each step of the code and also include docstrings for sections of the code

struct NS{space, P, B} <: InferenceAlgorithm  ## check what is T in T, B ??
    ndims::Int    # number of parameters
    nactive::Int    # number of active points
    proposals::P
    bounds::B   
end

proposal(p::NestedSamplers.Proposals) = p
bound(b::NestedSamplers.Bounds) = b

NS() = NS{()}()
NS(space::Symbol) = NS{(space,)}()

isgibbscomponent(::NS) = true # this states that NS alg is allowed as a Gibbs component

mutable struct NSState{V<:VarInfo} <: AbstractSamplerState
       vi::V
end

function Sampler(alg::NS, model::Model, s::Selector)
       # sanity check
       vi = VarInfo(model)
       info = Dict{Symbol, Any}()
       state = NSState(vi)
       info = Dict{Symbol, Any}()
       return Sampler(alg, info, s, state)
end

function AbstractMCMC.sample_init!(    ## model::Model??, use spl instead of s (sampler) ?? check , ref mh.jl
    rng::AbstractRNG,
    model::NestedModel,
    s::Nested{T,B},
    ::Integer;
    debug::Bool = false,
    kwargs...) where {T,B}

    debug && @info "Initializing sampler"
    local us, vs, logl
    ntries = 0
    while true
        us = rand(rng, s.ndims, s.nactive)
        vs = mapslices(model.prior_transform, us, dims=1)
        logl = mapslices(model.loglike, vs, dims=1)
        any(isfinite, logl) && break
        ntries += 1
        ntries > 100 && error("After 100 attempts, could not initialize any live points with finite loglikelihood. Please check your prior transform and loglikelihood method.")
    end
    # force -Inf to be a finite but small number to keep estimators from breaking
    @. logl[logl == -Inf] = -1e300

    # samples in unit space
    s.active_us .= us
    s.active_points .= vs
    s.active_logl .= logl[1, :]

    return nothing
end

##function AbstractMCMC.step!(   ## check ??
##    rng::AbstractRNG,
##    model::Model,
##    spl::Sampler{<:NS{space, P, B}},
##    ::Integer,
##    transition;
##    kwargs...
##) 
##       ## incomplete
##       where {space, P, B}
##    if spl.selector.rerun # Recompute joint in logp
##        model(spl.state.vi)
##    end
    
##    return Transition(spl)
##end

function AbstractMCMC.step!(::AbstractRNG,     ## check ?? 2 such functions to always accept in the first step??
    ::AbstractModel,
    s::Nested,
    ::Integer;
    kwargs...)
    # Find least likely point
    logL, idx = findmin(s.active_logl)
    draw = s.active_points[:, idx]
    log_wt = s.log_vol + logL

    # update sampler
    logz = logaddexp(s.logz, log_wt)
    s.h = (exp(log_wt - logz) * logL +
           exp(s.logz - logz) * (s.h + s.logz) - logz)
    s.logz = logz

    return NestedTransition(draw, logL, log_wt)
end

function AbstractMCMC.step!(rng::AbstractRNG,    ## check ?? why 2 such step!?? (also check if ! in both ??)
    model::NestedModel,
    s::Nested{T,B},
    ::Integer,
    prev::NestedTransition;
    iteration,
    debug::Bool = false,
    kwargs...) where {T,B}

    # Find least likely point
    logL, idx = findmin(s.active_logl)
    draw = s.active_points[:, idx]
    log_wt = s.log_vol + logL

    # update evidence and information
    logz = logaddexp(s.logz, prev.log_wt)
    s.h = (exp(prev.log_wt - logz) * prev.logL +
           exp(s.logz - logz) * (s.h + s.logz) - logz)
    s.logz = logz

    # check if ready for first update
    if !s.has_bounds && s.ncall > s.min_ncall && iteration / s.ncall < s.min_eff
        debug && @info "First update: it=$iteration, ncall=$(s.ncall), eff=$(iteration / s.ncall)"
        s.has_bounds = true
        pointvol = exp(s.log_vol) / s.nactive
        s.active_bound = Bounds.scale!(Bounds.fit(B, s.active_us, pointvol = pointvol), s.enlarge)
        s.since_update = 0
    # if accepted first update, is it time to update again?
    elseif iszero(s.since_update % s.update_interval)
        debug && @info "Updating bounds: it=$iteration, ncall=$(s.ncall), eff=$(iteration / s.ncall)"
        pointvol = exp(s.log_vol) / s.nactive
        s.active_bound = Bounds.scale!(Bounds.fit(B, s.active_us, pointvol = pointvol), s.enlarge)
        s.since_update = 0
    end
    
    # Get a live point to use for evolving with proposal
    if s.has_bounds
        point, bound = rand_live(rng, s.active_bound, s.active_us)
        u, v, logl, ncall = s.proposal(rng, point, logL, bound, model.loglike, model.prior_transform)
    else
        point = rand(rng, T, s.ndims)
        bound = Bounds.NoBounds(T, s.ndims)
        proposal = Proposals.Uniform()
        u, v, logl, ncall = proposal(rng, point, logL, bound, model.loglike, model.prior_transform)
    end

    # Get new point and log like
    s.active_us[:, idx] = u
    s.active_points[:, idx] = v
    s.active_logl[idx] = logl
    s.ndecl = log_wt < prev.log_wt ? s.ndecl + 1 : 0
    s.ncall += ncall
    s.since_update += 1

    # Shrink interval
    s.log_vol -=  1 / s.nactive

    return NestedTransition(draw, logL, log_wt)
end


struct NSModel{M<:Model,S<:Sampler{<:NS},T} <: AbstractMCMC.AbstractModel    ## or NestedModel ?? check
    model::M
    spl::S
    μ::T
end

function NSModel(model::Model, spl::Sampler{<:NS})    ## or NestedModel ?? check
    vi = spl.state.vi
    vns = _getvns(vi, spl)
    μ = mapreduce(vcat, vns[1]) do vn
        dist = getdist(vi, vn)
        vectorize(dist, mean(dist))
    end

    NSModel(model, spl, μ)    ## or NestedModel ?? check
end

##function AbstractMCMC.sample_end!(
##    ::AbstractRNG,
##    ::Model,
##    spl::Sampler{<:NS{space, P, B}},
##    N::Integer,
##    ts::Vector;
##    kwargs...
##)
##    ## incomplete
##end

function AbstractMCMC.sample_end!(::AbstractRNG,    ## check ??
    ::AbstractModel,
    s::Nested,
    ::Integer,
    transitions;
    debug::Bool = false,
    kwargs...)
    # Pop remaining points in ellipsoid
    N = length(transitions)
    s.log_vol = -N / s.nactive - log(s.nactive)
    @inbounds for i in eachindex(s.active_logl)
        # get new point
        draw = s.active_points[:, i]
        logL = s.active_logl[i]
        log_wt = s.log_vol + logL

        # update sampler
        logz = logaddexp(s.logz, log_wt)
        s.h = (exp(log_wt - logz) * logL +
               exp(s.logz - logz) * (s.h + s.logz) - logz)
        s.logz = logz

        prev = NestedTransition(draw, logL, log_wt)
        push!(transitions, prev)
    end

    # h should always be non-negative. Numerical error can arise from pathological corner cases
    if s.h < 0
        s.h ≉ 0 && @warn "Negative h encountered h=$(s.h). This is likely a bug"
        s.h = zero(s.h)
    end

    return nothing
end

function AbstractMCMC.bundle_samples(::AbstractRNG,   ## check?? why 2 ?? is AbstractMCMC.bundle_samples ?? (also check if or not ! in both)
    ::AbstractModel,
    s::Nested,
    ::Integer,
    transitions,
    Chains;
    param_names = missing,
    check_wsum = true,
    kwargs...)

    vals = copy(mapreduce(t->vcat(t.draw, t.log_wt), hcat, transitions)')
    # update weights based on evidence
    @. vals[:, end, 1] = exp(vals[:, end, 1] - s.logz)
    wsum = sum(vals[:, end, 1])
    @. vals[:, end, 1] /= wsum

    if check_wsum
        err = !iszero(s.h) ? 3sqrt(s.h / s.nactive) : 1e-3
        isapprox(wsum, 1, atol = err) || @warn "Weights sum to $wsum instead of 1; possible bug"
    end

    # Parameter names
    if param_names === missing
        param_names = ["Parameter $i" for i in 1:length(vals[1, :]) - 1]
    end
    push!(param_names, "weights")

    return Chains(vals, param_names, Dict(:internals => ["weights"]), evidence = s.logz)
end

function AbstractMCMC.bundle_samples(::AbstractRNG,    ## check?? why 2?? is AbstractMCMC.bundle_samples ??
    ::AbstractModel,
    s::Nested,
    ::Integer,
    transitions,
    A::Type{<:AbstractArray};
    check_wsum = true,
    kwargs...)

    vals = convert(A, mapreduce(t->t.draw, hcat, transitions)')

    if check_wsum
        # get weights
        wsum = mapreduce(t->exp(t.log_wt - s.logz), +, transitions)

        # check with h
        err = s.h ≠ 0 ? 3sqrt(s.h / s.nactive) : 1e-3
        isapprox(wsum, 1, atol = err) || @warn "Weights sum to $wsum instead of 1; possible bug"
    end

    return vals
end

# evaluate log-likelihood
function Distributions.loglikelihood(model::NSModel, f)  ## or NestedModel ?? check
    spl = model.spl
    vi = spl.state.vi
    vi[spl] = f
    model.model(vi, spl)
    getlogp(vi)
end

# tilde operators
function DynamicPPL.tilde(rng, ctx::DefaultContext, sampler::Sampler{<:NS}, right, vn::VarName, inds, vi)
    if inspace(vn, sampler)
        return DynamicPPL.tilde(rng, LikelihoodContext(), SampleFromPrior(), right, vn, inds, vi)
    else
        return DynamicPPL.tilde(rng, ctx, SampleFromPrior(), right, vn, inds, vi)
    end
end

function DynamicPPL.tilde(ctx::DefaultContext, sampler::Sampler{<:NS}, right, left, vi)
    return DynamicPPL.tilde(ctx, SampleFromPrior(), right, left, vi)
end

function DynamicPPL.dot_tilde(rng, ctx::DefaultContext, sampler::Sampler{<:NS}, right, left, vn::VarName, inds, vi)
    if inspace(vn, sampler)
        return DynamicPPL.dot_tilde(rng, LikelihoodContext(), SampleFromPrior(), right, left, vn, inds, vi)
    else
        return DynamicPPL.dot_tilde(rng, ctx, SampleFromPrior(), right, left, vn, inds, vi)
    end
end

function DynamicPPL.dot_tilde(rng, ctx::DefaultContext, sampler::Sampler{<:NS}, right, left, vi)
    return DynamicPPL.dot_tilde(rng, ctx, SampleFromPrior(), right, left, vi)
end
