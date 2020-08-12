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

struct ESS{space} <: InferenceAlgorithm end

ESS() = ESS{()}()
ESS(space::Symbol) = ESS{(space,)}()

struct NS{space, P, B} <: InferenceAlgorithm  ## refer the comment on line 39
    proposals::P
    bounds::B   
end  

proposal(p::NestedSamplers.Proposals) = p
bound(b::NestedSamplers.Bounds) = b

##NS() = NS{()}()
##NS(space::Symbol) = NS{(space,)}()

function NS(

isgibbscomponent(::NS) = true # this states that NS alg is allowed as a Gibbs component

mutable struct NSState{V<:VarInfo} <: AbstractSamplerState   ## where is this being used? in the function Sampler ?
       vi::V
end

NSState(model::Model) = NSState(VarInfo(model))

struct NSModel <: AbstractMCMC.AbstractModel    ## or NestedModel ?? check  ... this struct & function were placed above sample_end! earlier (better to define all such wraps here at the beginning)
    loglike::Function
    prior_transform::Function
end

function NSModel(loglike, prior_transform)  
    return NestedSamplers.NestedModel(loglike, prior_transform)   
end

function Sampler(
       alg::NS,
       model::NSModel,    ## check this everywhere   ## when NSmodel is defined, then in which function is it used?
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
    model::NSModel,   ## check `NestedModel`?
    spl::Sampler{<:NS},    ## `Nested{T,B}` ## s == spl ## or `Sampler{<:NS{space, T, P, B}}`
    N::Integer,
    verbose::Bool=true,
    resume_from=nothing,
    debug::Bool = false;
    kwargs...
) where {T, P, B}
    # You need to make a `NestedModel` somehow
    # spl.nested is what I imagine you'd call `Nested` if you placed it in `NS` and just wrapped around everything in 
    # NestedSamplers.jl
    AbstractMCMC.sample_init(rng, nested_model, spl.nested, N)
end

function AbstractMCMC.step!(    ## check 2 AbstractMCMC.step! functions ?
    ::AbstractRNG,     
    model::NSModel,     ## or `AbstractModel`?
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
    model::NSModel,    ## or `NestedModel`
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

function AbstractMCMC.sample_end!(
    rng::AbstractRNG,    ## check ??
    model::NSModel,     ## or `AbstractModel`
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
