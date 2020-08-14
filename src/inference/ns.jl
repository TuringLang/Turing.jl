"""
------------------------------------
not working yet
still being built
-------------------------------------

Nested sampling algorithm.

An example
#Note-to-self: set a seed here
## write an example here

"""

## what all to setup here:
## 1. A subtype of AbstractSampler, defined as a mutable struct containing state information or sampler parameters
## 2. A function sample_init! which performs any necessary set-up (default: do not perform any set-up)... Partially done in: function AbstractMCMC.sample_init! ? check
## 3. A function step! which returns a transition that represents a single draw from the sampler... Partially done with: 2 function AbstractMCMC.step! ? check
## 4. A function transitions_init which returns a container for the transitions obtained from the sampler (default: return a Vector{T} of length N where T is the type of the transition obtained in the first step and N is the number of requested samples).
## 5. A function transitions_save! which saves transitions to the container (default: save the transition of iteration i at position i in the vector of transitions)
## 6. A function sample_end! which handles any sampler wrap-up (default: do not perform any wrap-up)... Partially done in: function AbstractMCMC.sample_end! ? check
## 7. A function bundle_samples which accepts the container of transitions and returns a collection of samples (default: return the vector of transitions)

## model:: NestedModel
## sampler:: Nested

## comment on each step of the code and also include docstrings for sections of the code

struct NS{space} <: InferenceAlgorithm 
    ndims::Int
    nactive::Int
end

## should both these constructors be defined? as `ndims` and `nactive`, at the minimum, are required for `Nested` to work
##NS() = NS{()}()    
##NS(space::Symbol) = NS{(space,)}()

function NS(ndims, nactive)
        nested = NestedSamplers.Nested(ndims, nactive)    ##  using the fact that the module `NestedSamplers` exports `Nested`
        ## revisit this to understand how can you incorporate changes in proposal, bounds and other kwargs of `Nested` 
end    

isgibbscomponent(::NS) = true # this states that NS alg is allowed as a Gibbs component

struct NSModel <: AbstractMCMC.AbstractModel    ## or NestedModel ?? check  ... this struct & function were placed above sample_end! earlier (better to define all such wraps here at the beginning)
    loglike::Function
    prior_transform::Function
end

function NSModel(loglike, prior_transform)  
    return NestedSamplers.NestedModel(loglike, prior_transform)   ## using the fact that the module `NestedSamplers` exports `NestedModel`: 
end

function Sampler(
       alg::NS,
       model::NSModel,    ## check this everywhere, `NSModel` or `Model`   
       s::Selector=Selector()
)
       # sanity check
       vi = VarInfo(model)
       
       # set up info dict
       info = Dict{Symbol, Any}()
    
       # set up state struct
       state = SamplerState(vi)
    
       # generate a sampler
       return Sampler(alg, info, s, state)
end

# initialize nested sampler
function AbstractMCMC.sample_init!(    
    rng::AbstractRNG,
    model::NSModel,  
    spl::Sampler{<:NS},    
    N::Integer,
    verbose::Bool=true,
    resume_from=nothing,
    debug::Bool = false;
    kwargs...
)
    AbstractMCMC.sample_init(rng, model, spl.nested, N)    ## traceback `spl.nested` & check if correctly implemented
end

function AbstractMCMC.step!(   
    ::AbstractRNG,     
    model::NSModel,     
    spl::Sampler{<:NS},   
    ::Integer,
    ::Nothing;
    kwargs...
)
    return Transition(spl)    
end

function AbstractMCMC.step!(    
    rng::AbstractRNG,    
    model::NSModel,   
    spl::Sampler{<:NS},     
    ::Integer,
    transition;    
    iteration,
    debug::Bool = false,
    kwargs...
)
    # random variable to be sampled
    vi = spl.state.vi

    ## more steps to include here
    ## define previous sample, previous sampler state, next state, and update sample and loglike
        
    return Transition(spl)
end

## check if `AbstractMCMC.transitions_init` and `AbstractMCMC.transitions_save!` needed here, also if a NSTransition struct is needed?

# finalize nested sampler
function AbstractMCMC.sample_end!(
    rng::AbstractRNG,    
    model::NSModel,    
    spl::Sampler{<:NS},      
    N::Integer,
    transitions;
    debug::Bool = false,
    kwargs...
)  
    AbstractMCMC.sample_end(rng, model, spl.nested, N)    ## traceback `spl.nested` to ensure correct implementation
end

## tilde operators  ## or experiment with assume, dot_assume, observe and dot_observe

function DynamicPPL.tilde(
    rng, 
    ctx::DefaultContext, 
    sampler::Sampler{<:NS}, 
    right, 
    vn::VarName,
    inds, 
    vi
)
    if inspace(vn, sampler)
        return DynamicPPL.tilde(rng, LikelihoodContext(), SampleFromPrior(), right, vn, inds, vi)
    else
        return DynamicPPL.tilde(rng, ctx, SampleFromPrior(), right, vn, inds, vi)
    end
end

function DynamicPPL.tilde(
    ctx::DefaultContext, 
    sampler::Sampler{<:NS},
    right,
    left,
    vi
)
    return DynamicPPL.tilde(ctx, SampleFromPrior(), right, left, vi)
end

function DynamicPPL.dot_tilde(
    rng, 
    ctx::DefaultContext, 
    sampler::Sampler{<:NS}, 
    right, 
    left, 
    vn::VarName, 
    inds, 
    vi
)
    if inspace(vn, sampler)
        return DynamicPPL.dot_tilde(rng, LikelihoodContext(), SampleFromPrior(), right, left, vn, inds, vi)
    else
        return DynamicPPL.dot_tilde(rng, ctx, SampleFromPrior(), right, left, vn, inds, vi)
    end
end

function DynamicPPL.dot_tilde(
    rng, 
    ctx::DefaultContext, 
    sampler::Sampler{<:NS}, 
    right, 
    left, 
    vi
)
    return DynamicPPL.dot_tilde(rng, ctx, SampleFromPrior(), right, left, vi)
end
