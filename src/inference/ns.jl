"""
------------------------------------
still being built
-------------------------------------

Nested sampling algorithm.

An example
#Note-to-self: set a seed here
## write an example here

"""

## model:: NestedModel
## sampler:: Nested

## include docstrings for sections of the code

# InferenceAlgorithm struct

struct NS{space, N<:NestedSamplers.Nested} <: InferenceAlgorithm
    nested::N
end

function NS(
    ndims::Int,
    nactive::Int,
    bound_type,    # To be input as, for instance, NestedSamplers.Bounds.Ellipsoid
    proposal_type,    # To be input as, for instance, NestedSamplers.Proposals.Uniform()
    enlarge=1.25,
    min_eff=0.10
)
    ##bounds = bound_type 
    ##proposal = proposal_type
    update_interval = default_update_interval(proposal_type, ndims) 
    min_ncall = 2 * nactive
    nested = NestedSamplers.Nested(ndims, nactive, bound_type, proposal_type, enlarge, update_interval,
                                   min_ncall, min_eff)
    return NS{(), typeof(nested)}(nested)
end

## isgibbscomponent(::NS) = true    # This states that NS alg is allowed as a Gibbs component

# Sampler

function Sampler(
       alg::NS,
       model::Model,
       s::Selector=Selector()
)
       vi = VarInfo(model)    # Sanity check
       info = Dict{Symbol, Any}()    # Set up info dict
       state = SamplerState(vi)    # Set up state struct
       return Sampler(alg, info, s, state)    # Generate a sampler
end

# Initialize nested sampler

function AbstractMCMC.sample_init!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:NS},
    N::Integer;
    verbose::Bool=true,
    resume_from=nothing,
    kwargs...
)
    set_resume!(spl; resume_from=resume_from, kwargs...)    # Resume the sampler
    initialize_parameters!(spl; verbose=verbose, kwargs...)    # Get initial parameters
    link!(spl.state.vi, spl)    # Link everything before sampling 
end

# First step

function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:NS},
    N::Integer,
    transition::Nothing;
    kwargs...
)
    return Transition(spl)
end

# Subsequent steps

function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:NS},
    ::Integer,
    transition;
    kwargs...
)
    vi = spl.state.vi    # Random variable to be sampled
    earlier_sample = vi[spl]    # Earlier sample
    if spl.selector.tag !== :default    # Recompute log-likelihood in logp
        model(vi, spl)
    end
    earlier_state = SamplerState(earlier_sample, getlogp(vi))    # Earlier sampler state           ## check this step, if both arguments are correct
    nested_model = NestedSamplers.NestedModel(DynamicPPL.loglikelihood(vi), SampleFromPrior())    # Generate a nested model        ## check this step:
    ## 1. NestedModel uses Distributions.quantile to transform the prior, does the usage of SampleFromPrior(), take this into account? or we need
    ## to have some another approach? 
    ## 2. Also the first argument of NestedModel is loglikelihood function. Using DynamicPPL.loglikelihood(vi) in its place. Is this the correct usage?
    subsequent_state = AbstractMCMC.step!(rng, nested_model, spl.alg.nested, 1, earlier_state)    # Subsequent state
    vi[spl] = subsequent_state.sample    # Update sample and log-likelihood
    setlogp!(vi, subsequent_state.loglikelihood)
    return Transition(spl)
end

# Nested model struct     ## not used anywhere so far

struct NSModel{M<:Model,S<:Sampler{<:NS},T} <: AbstractMCMC.AbstractModel
    model::M
    spl::S
    μ::T
end

## Finalize nested sampler

#function AbstractMCMC.sample_end!(
#    rng::AbstractRNG,
#    model::Model,
#    spl::Sampler{<:NS},
#    N::Integer,
#    transitions;
#    kwargs...
#)
#    invlink!(spl.state.vi, spl)    # Invlink everything after sampling  
#end

# Tilde operators

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
