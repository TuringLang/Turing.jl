using DynamicPPL: VarName
using Random: Random
import AbstractMCMC

# These functions provide specialized methods for GibbsConditional that extend the generic implementations in gibbs.jl

"""
    GibbsConditional(sym::Symbol, conditional)

A Gibbs sampler component that samples a variable according to a user-provided
analytical conditional distribution.

The `conditional` function should take a `NamedTuple` of conditioned variables and return
a `Distribution` from which to sample the variable `sym`.

# Examples

```julia
# Define a model
@model function inverse_gdemo(x)
    λ ~ Gamma(2, 3)
    m ~ Normal(0, sqrt(1 / λ))
    for i in 1:length(x)
        x[i] ~ Normal(m, sqrt(1 / λ))
    end
end

# Define analytical conditionals
function cond_λ(c::NamedTuple)
    a = 2.0
    b = 3.0
    m = c.m
    x = c.x
    n = length(x)
    a_new = a + (n + 1) / 2
    b_new = b + sum((x[i] - m)^2 for i in 1:n) / 2 + m^2 / 2
    return Gamma(a_new, 1 / b_new)
end

function cond_m(c::NamedTuple)  
    λ = c.λ
    x = c.x
    n = length(x)
    m_mean = sum(x) / (n + 1)
    m_var = 1 / (λ * (n + 1))
    return Normal(m_mean, sqrt(m_var))
end

# Sample using GibbsConditional
model = inverse_gdemo([1.0, 2.0, 3.0])
chain = sample(model, Gibbs(
    :λ => GibbsConditional(:λ, cond_λ),
    :m => GibbsConditional(:m, cond_m)
), 1000)
```
"""
struct GibbsConditional{C} <: InferenceAlgorithm
    conditional::C

    function GibbsConditional(sym::Symbol, conditional::C) where {C}
        return new{C}(conditional)
    end
end

# Mark GibbsConditional as a valid Gibbs component
isgibbscomponent(::GibbsConditional) = true

"""
    find_global_varinfo(context, fallback_vi)

Traverse the context stack to find global variable information from
GibbsContext, ConditionContext, FixedContext, etc.
"""
function find_global_varinfo(context, fallback_vi)
    # Start with the given context and traverse down
    current_context = context
    
    while current_context !== nothing
        if current_context isa GibbsContext
            # Found GibbsContext, return its global varinfo
            return get_global_varinfo(current_context)
        elseif hasproperty(current_context, :childcontext) && 
               isdefined(DynamicPPL, :childcontext)
            # Move to child context if it exists
            current_context = DynamicPPL.childcontext(current_context)
        else
            # No more child contexts
            break
        end
    end
    
    # If no GibbsContext found, use the fallback
    return fallback_vi
end

"""
    DynamicPPL.initialstep(rng, model, sampler::GibbsConditional, vi)

Initialize the GibbsConditional sampler.
"""
function DynamicPPL.initialstep(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler::DynamicPPL.Sampler{<:GibbsConditional},
    vi::DynamicPPL.AbstractVarInfo;
    kwargs...,
)
    # GibbsConditional doesn't need any special initialization
    # Just return the initial state
    return nothing, vi
end

"""
    AbstractMCMC.step(rng, model, sampler::GibbsConditional, state)

Perform a step of GibbsConditional sampling.
"""
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler::DynamicPPL.Sampler{<:GibbsConditional{S}},
    state::DynamicPPL.AbstractVarInfo;
    kwargs...,
) where {S}
    alg = sampler.alg

    # For GibbsConditional within Gibbs, we need to get all variable values
    # Traverse the context stack to find all conditioned/fixed/Gibbs variables
    global_vi = if hasproperty(model, :context)
        find_global_varinfo(model.context, state)
    else
        state
    end

    # Extract conditioned values as a NamedTuple
    # Include both random variables and observed data
    condvals_vars = DynamicPPL.values_as(DynamicPPL.invlink(global_vi, model), NamedTuple)
    condvals_obs = NamedTuple{keys(model.args)}(model.args)
    condvals = merge(condvals_vars, condvals_obs)

    # Get the conditional distribution
    conddist = alg.conditional(condvals)

    # Sample from the conditional distribution
    updated = rand(rng, conddist)

    # Update the variable in state
    # The Gibbs sampler ensures that state only contains one variable
    # Get the variable name from the keys
    varname = first(keys(state))
    new_vi = DynamicPPL.setindex!!(state, updated, varname)

    return nothing, new_vi
end

"""
    setparams_varinfo!!(model, sampler::GibbsConditional, state, params::AbstractVarInfo)

Update the variable info with new parameters for GibbsConditional.
"""
function setparams_varinfo!!(
    model::DynamicPPL.Model,
    sampler::DynamicPPL.Sampler{<:GibbsConditional},
    state,
    params::DynamicPPL.AbstractVarInfo,
)
    # For GibbsConditional, we just return the params as-is since 
    # the state is nothing and we don't need to update anything
    return params
end

