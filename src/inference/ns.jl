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
## 1. A subtype of AbstractSampler, defined as a mutable struct containing state information or sampler parameters... Partially done in  mutable struct NSState ? check
## 2. A function sample_init! which performs any necessary set-up (default: do not perform any set-up)... Partially done in: function AbstractMCMC.sample_init! ? check
## 3. A function step! which returns a transition that represents a single draw from the sampler... Partially done with: 2 function AbstractMCMC.step! ? check
## 4. A function transitions_init which returns a container for the transitions obtained from the sampler (default: return a Vector{T} of length N where T is the type of the transition obtained in the first step and N is the number of requested samples).
## 5. A function transitions_save! which saves transitions to the container (default: save the transition of iteration i at position i in the vector of transitions)
## 6. A function sample_end! which handles any sampler wrap-up (default: do not perform any wrap-up)... Partially done in: function AbstractMCMC.sample_end! ? check
## 7. A function bundle_samples which accepts the container of transitions and returns a collection of samples (default: return the vector of transitions)... Partially done with: 2 function AbstractMCMC.bundle_samples ? check

## model:: NestedModel
## sampler:: Nested

## How the struct NestedModel is defined:
##struct NestedModel <: AbstractModel
##    loglike::Function
##    prior_transform::Function
##end

##function NestedModel(loglike, priors::AbstractVector{<:UnivariateDistribution})
##    prior_transform(X) = quantile.(priors, X)
##    return NestedModel(loglike, prior_transform)
##end

## revisit the Nested sampler definition here: https://github.com/TuringLang/NestedSamplers.jl/blob/master/src/staticsampler.jl

## comment on each step of the code and also include docstrings for sections of the code
## both NestedModel and Nested are to be utilized in this code

struct NS{space, T, P <: AbstractProposal, B <: AbstractBoundingSpace{T}} <: InferenceAlgorithm  ## refer the comment on line 39
    ndims::Int    # number of parameters    ## check this step
    nactive::Int    # number of active points    ## check this step
    active_us::Matrix{T}
    active_points::Matrix{T}
    active_logl::Vector{T}
    log_vol::Float64
    logz::Float64
    h::Float64
    update_interval::Int
    since_update::Int
    enlarge::Float64
    has_bounds::Bool
    ncall::Int
    min_ncall::Int
    min_eff::Floaat64
    proposal::P
    active_bound::B   
end  

proposal(p::NestedSamplers.Proposals) = p
bound(b::NestedSamplers.Bounds) = b

NS() = NS{()}()
NS(space::Symbol) = NS{(space,)}()

isgibbscomponent(::NS) = true # this states that NS alg is allowed as a Gibbs component

## Define a general `funtion NS` here ?

mutable struct NSState{V<:VarInfo} <: AbstractSamplerState   ## where is this being used? in the function Sampler ?
       vi::V
end

NSState(model::Model) = NSState(VarInfo(model))

function Sampler(
       alg::NS,
       model::Model,
       s::Selector=Selector()
)
       # sanity check
       vi = VarInfo(model)
       
       # set up info dict
       info = Dict{Symbol, Any}()
    
       # set up state struct
       state = NSState(vi)
    
       # generate a sampler
       return Sampler(alg, info, s, state)
end

function AbstractMCMC.sample_init!(    
    rng::AbstractRNG,
    model::Model,   ## check `NestedModel`?
    spl::Sampler{<:NS},    ## `Nested{T,B}` ## s == spl ## or `Sampler{<:NS{space, T, P, B}}`
    N::Integer,
    verbose::Bool=true,
    resume_from=nothing,
    debug::Bool = false;
    kwargs...
) where {T, P, B}

    debug && @info "Initializing sampler"
    local us, vs, logl
    ntries = 0
    while true
        us = rand(rng, spl.ndims, spl.nactive)
        vs = mapslices(model.prior_transform, us, dims=1)
        logl = mapslices(model.loglike, vs, dims=1)
        any(isfinite, logl) && break
        ntries += 1
        ntries > 100 && error("After 100 attempts, could not initialize any live points with finite loglikelihood. Please check your prior transform and loglikelihood method.")
    end
    # force -Inf to be a finite but small number to keep estimators from breaking
    @. logl[logl == -Inf] = -1e300

    # samples in unit space
    spl.active_us .= us
    spl.active_points .= vs
    spl.active_logl .= logl[1, :]

    return nothing
end

function AbstractMCMC.step!(    ## check 2 AbstractMCMC.step! functions ?
    ::AbstractRNG,     
    model::Model,     ## or `AbstractModel`?
    spl::Sampler{<:NS},     ## or `Nested` or `Sampler{<:NS{space, T, P, B}}`
    ::Integer,
    ::Nothing;
    kwargs...
)
    # Find least likely point
    logL, idx = findmin(spl.active_logl)
    draw = spl.active_points[:, idx]
    log_wt = spl.log_vol + logL

    # update sampler
    logz = logaddexp(spl.logz, log_wt)
    spl.h = (exp(log_wt - logz) * logL +
           exp(spl.logz - logz) * (spl.h + spl.logz) - logz)
    spl.logz = logz

    return NestedTransition(draw, logL, log_wt)     ## or `Transition(spl)`
end

function AbstractMCMC.step!(    ## check 2 AbstractMCMC.step! functions ?
    rng::AbstractRNG,    
    model::Model,    ## or `NestedModel`
    spl::Sampler{<:NS},      ## or `Nested{T,B}`
    ::Integer,
    prev::NestedTransition;    ## or `prev::Transition` or only `transition`
    iteration,
    debug::Bool = false,
    kwargs...
) where {T, P, B}

    # Find least likely point
    logL, idx = findmin(spl.active_logl)
    draw = spl.active_points[:, idx]
    log_wt = spl.log_vol + logL

    # update evidence and information
    logz = logaddexp(spl.logz, prev.log_wt)
    spl.h = (exp(prev.log_wt - logz) * prev.logL +
           exp(spl.logz - logz) * (spl.h + spl.logz) - logz)
    spl.logz = logz

    # check if ready for first update
    if !spl.has_bounds && spl.ncall > spl.min_ncall && iteration / spl.ncall < spl.min_eff
        debug && @info "First update: it=$iteration, ncall=$(spl.ncall), eff=$(iteration / spl.ncall)"
        spl.has_bounds = true
        pointvol = exp(spl.log_vol) / spl.nactive
        spl.active_bound = NestedSamplers.Bounds.scale!(NestedSamplers.Bounds.fit(B, spl.active_us, pointvol = pointvol), spl.enlarge)   ## `NestedSamplers.Bounds` or only `Bounds`
        spl.since_update = 0
    # if accepted first update, is it time to update again?
    elseif iszero(spl.since_update % spl.update_interval)
        debug && @info "Updating bounds: it=$iteration, ncall=$(spl.ncall), eff=$(iteration / spl.ncall)"
        pointvol = exp(spl.log_vol) / spl.nactive
        spl.active_bound = NestedSamplers.Bounds.scale!(NestedSamplers.Bounds.fit(B, spl.active_us, pointvol = pointvol), spl.enlarge)
        spl.since_update = 0
    end
    
    # Get a live point to use for evolving with proposal
    if spl.has_bounds
        point, bound = rand_live(rng, spl.active_bound, spl.active_us)
        u, v, logl, ncall = spl.proposal(rng, point, logL, bound, model.loglike, model.prior_transform)
    else
        point = rand(rng, T, s.ndims)
        bound = NestedSamplers.Bounds.NoBounds(T, spl.ndims)
        proposal = NestedSamplers.Proposals.Uniform()
        u, v, logl, ncall = proposal(rng, point, logL, bound, model.loglike, model.prior_transform)
    end

    # Get new point and log like
    spl.active_us[:, idx] = u
    spl.active_points[:, idx] = v
    spl.active_logl[idx] = logl
    spl.ndecl = log_wt < prev.log_wt ? spl.ndecl + 1 : 0
    spl.ncall += ncall
    spl.since_update += 1

    # Shrink interval
    spl.log_vol -=  1 / spl.nactive

    return NestedTransition(draw, logL, log_wt)
end


struct NSModel{M<:Model,S<:Sampler{<:NS},T} <: AbstractMCMC.AbstractModel    ## or NestedModel ?? check
    model::M
    spl::S
    μ::T
end

function NSModel(    ## or NestedModel ? check  ## check where the struct `NSModel` & corresponding function to include
    model::Model, 
    spl::Sampler{<:NS}
)    
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

function AbstractMCMC.sample_end!(
    rng::AbstractRNG,    ## check ??
    model::Model,     ## or `AbstractModel`
    spl::Sampler{<:NS},      ## or `Nested`
    N::Integer,
    transitions;
    debug::Bool = false,
    kwargs...
)
    # Pop remaining points in ellipsoid
    N = length(transitions)
    spl.log_vol = -N / spl.nactive - log(spl.nactive)
    @inbounds for i in eachindex(spl.active_logl)
        # get new point
        draw = spl.active_points[:, i]
        logL = spl.active_logl[i]
        log_wt = spl.log_vol + logL

        # update sampler
        logz = logaddexp(spl.logz, log_wt)
        spl.h = (exp(log_wt - logz) * logL +
               exp(spl.logz - logz) * (spl.h + spl.logz) - logz)
        spl.logz = logz

        prev = NestedTransition(draw, logL, log_wt)
        push!(transitions, prev)
    end

    # h should always be non-negative. Numerical error can arise from pathological corner cases
    if spl.h < 0
        spl.h ≉ 0 && @warn "Negative h encountered h=$(spl.h). This is likely a bug"
        spl.h = zero(spl.h)
    end

    return nothing
end

function AbstractMCMC.bundle_samples(    ## 2 `AbstractMCMC.bundle_samples` functions
    ::AbstractRNG,  
    model::Model,    ## or `AbstractModel`
    spl::Sampler{<:NS},    ## or `Nested`
    ::Integer,
    transitions,
    Chains;
    param_names = missing,
    check_wsum = true,
    kwargs...
)

    vals = copy(mapreduce(t->vcat(t.draw, t.log_wt), hcat, transitions)')
    # update weights based on evidence
    @. vals[:, end, 1] = exp(vals[:, end, 1] - spl.logz)
    wsum = sum(vals[:, end, 1])
    @. vals[:, end, 1] /= wsum

    if check_wsum
        err = !iszero(spl.h) ? 3sqrt(spl.h / spl.nactive) : 1e-3
        isapprox(wsum, 1, atol = err) || @warn "Weights sum to $wsum instead of 1; possible bug"
    end

    # Parameter names
    if param_names === missing
        param_names = ["Parameter $i" for i in 1:length(vals[1, :]) - 1]
    end
    push!(param_names, "weights")

    return Chains(vals, param_names, Dict(:internals => ["weights"]), evidence = spl.logz)
end

function AbstractMCMC.bundle_samples(    ## 2 `AbstractMCMC.bundle_samples` functions
    rng::AbstractRNG,   
    model::Model,    ## or `AbstractModel`
    spl::Sampler{<:NS},     ## or `Nested`
    ::Integer,
    transitions,
    A::Type{<:AbstractArray};
    check_wsum = true,
    kwargs...
)

    vals = convert(A, mapreduce(t->t.draw, hcat, transitions)')

    if check_wsum
        # get weights
        wsum = mapreduce(t->exp(t.log_wt - spl.logz), +, transitions)

        # check with h
        err = spl.h ≠ 0 ? 3sqrt(spl.h / spl.nactive) : 1e-3
        isapprox(wsum, 1, atol = err) || @warn "Weights sum to $wsum instead of 1; possible bug"
    end

    return vals
end

## tilde operators  ## or experiment with assume, dot_assume, observe and dot_observe

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
