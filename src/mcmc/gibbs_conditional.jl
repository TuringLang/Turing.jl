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
    λ ~ Gamma(2, inv(3))
    m ~ Normal(0, sqrt(1 / λ))
    for i in 1:length(x)
        x[i] ~ Normal(m, sqrt(1 / λ))
    end
end

# Define analytical conditionals
function cond_λ(c::NamedTuple)
    a = 2.0
    b = inv(3)
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

# Required methods for Gibbs constructor
Base.length(::GibbsConditional) = 1  # Each GibbsConditional handles one variable

"""
    find_global_varinfo(context, fallback_vi)

Traverse the context stack to find global variable information from
GibbsContext, ConditionContext, FixedContext, etc.
"""
function find_global_varinfo(context, fallback_vi)
    # Traverse the entire context stack to find relevant contexts
    current_context = context
    gibbs_context = nothing
    condition_context = nothing
    fixed_context = nothing

    while current_context !== nothing
        # Use NodeTrait for robust context checking
        if DynamicPPL.NodeTrait(current_context) isa DynamicPPL.IsParent
            if current_context isa GibbsContext
                gibbs_context = current_context
            elseif current_context isa DynamicPPL.ConditionContext
                condition_context = current_context
            elseif current_context isa DynamicPPL.FixedContext
                fixed_context = current_context
            end
            # Move to child context
            current_context = DynamicPPL.childcontext(current_context)
        else
            break
        end
    end

    # Return the most relevant context's varinfo
    if gibbs_context !== nothing
        return get_global_varinfo(gibbs_context)
    elseif condition_context !== nothing
        return DynamicPPL.getvarinfo(condition_context)
    elseif fixed_context !== nothing
        return DynamicPPL.getvarinfo(fixed_context)
    else
        return fallback_vi
    end
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
    sampler::DynamicPPL.Sampler{<:GibbsConditional},
    state::DynamicPPL.AbstractVarInfo;
    kwargs...,
)
    alg = sampler.alg

    # For GibbsConditional within Gibbs, we need to get all variable values
    # Model always has a context field, so we can traverse it directly
    global_vi = find_global_varinfo(model.context, state)

    # Extract conditioned values as a NamedTuple
    # Include both random variables and observed data
    condvals_vars = DynamicPPL.values_as(DynamicPPL.invlink(global_vi, model), NamedTuple)
    condvals_obs = NamedTuple{keys(model.args)}(model.args)
    condvals = merge(condvals_vars, condvals_obs)

    # Get the conditional distribution
    conddist = alg.conditional(condvals)

    # Sample from the conditional distribution
    updated = rand(rng, conddist)

    # Update the variable in state using unflatten for simplicity
    # The Gibbs sampler ensures that state only contains one variable
    new_vi = DynamicPPL.unflatten(state, [updated])

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
