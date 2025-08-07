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
struct GibbsConditional{S,C} <: InferenceAlgorithm
    conditional::C

    function GibbsConditional(sym::Symbol, conditional::C) where {C}
        return new{sym,C}(conditional)
    end
end

# Mark GibbsConditional as a valid Gibbs component
isgibbscomponent(::GibbsConditional) = true

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
    # Check if we're in a Gibbs context
    global_vi = if hasproperty(model, :context) && model.context isa GibbsContext
        # We're in a Gibbs context, get the global varinfo
        get_global_varinfo(model.context)
    else
        # We're not in a Gibbs context, use the current state
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
    # We need to get the actual VarName for this variable
    # The symbol S tells us which variable to update
    vn = VarName{S}()

    # Check if the variable needs to be a vector
    new_vi = if haskey(state, vn)
        # Update the existing variable
        DynamicPPL.setindex!!(state, updated, vn)
    else
        # Try to find the variable with indices
        # This handles cases where the variable might have indices
        local updated_vi = state
        found = false
        for key in keys(state)
            if DynamicPPL.getsym(key) == S
                updated_vi = DynamicPPL.setindex!!(state, updated, key)
                found = true
                break
            end
        end
        if !found
            error("Could not find variable $S in VarInfo")
        end
        updated_vi
    end

    # Update log joint probability
    new_vi = last(DynamicPPL.evaluate!!(model, new_vi, DynamicPPL.DefaultContext()))

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

"""
    gibbs_initialstep_recursive(
        rng, model, sampler::GibbsConditional, target_varnames, global_vi, prev_state
    )

Initialize the GibbsConditional sampler.
"""
function gibbs_initialstep_recursive(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler_wrapped::DynamicPPL.Sampler{<:GibbsConditional},
    target_varnames::AbstractVector{<:VarName},
    global_vi::DynamicPPL.AbstractVarInfo,
    prev_state,
)
    # GibbsConditional doesn't need any special initialization
    # Just perform one sampling step
    return gibbs_step_recursive(
        rng, model, sampler_wrapped, target_varnames, global_vi, nothing
    )
end

"""
    gibbs_step_recursive(
        rng, model, sampler::GibbsConditional, target_varnames, global_vi, state
    )

Perform a single step of GibbsConditional sampling.
"""
function gibbs_step_recursive(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler_wrapped::DynamicPPL.Sampler{<:GibbsConditional{S}},
    target_varnames::AbstractVector{<:VarName},
    global_vi::DynamicPPL.AbstractVarInfo,
    state,
) where {S}
    sampler = sampler_wrapped.alg

    # Extract conditioned values as a NamedTuple
    # Include both random variables and observed data
    condvals_vars = DynamicPPL.values_as(DynamicPPL.invlink(global_vi, model), NamedTuple)
    condvals_obs = NamedTuple{keys(model.args)}(model.args)
    condvals = merge(condvals_vars, condvals_obs)

    # Get the conditional distribution
    conddist = sampler.conditional(condvals)

    # Sample from the conditional distribution
    updated = rand(rng, conddist)

    # Update the variable in global_vi
    # We need to get the actual VarName for this variable
    # The symbol S tells us which variable to update
    vn = VarName{S}()

    # Check if the variable needs to be a vector
    if haskey(global_vi, vn)
        # Update the existing variable
        global_vi = DynamicPPL.setindex!!(global_vi, updated, vn)
    else
        # Try to find the variable with indices
        # This handles cases where the variable might have indices
        for key in keys(global_vi)
            if DynamicPPL.getsym(key) == S
                global_vi = DynamicPPL.setindex!!(global_vi, updated, key)
                break
            end
        end
    end

    # Update log joint probability
    global_vi = last(DynamicPPL.evaluate!!(model, global_vi, DynamicPPL.DefaultContext()))

    return nothing, global_vi
end
