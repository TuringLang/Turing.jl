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

    # Safety check: avoid infinite loops with a maximum depth
    max_depth = 20
    depth = 0

    while current_context !== nothing && depth < max_depth
        depth += 1

        try
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
        catch e
            # If there's an error traversing contexts, break and use fallback
            @debug "Error traversing context at depth $depth: $e"
            break
        end
    end

    # Return the most relevant context's varinfo with error handling
    try
        if gibbs_context !== nothing
            return get_global_varinfo(gibbs_context)
        elseif condition_context !== nothing
            # Check if getvarinfo method exists for ConditionContext
            if hasmethod(DynamicPPL.getvarinfo, (typeof(condition_context),))
                return DynamicPPL.getvarinfo(condition_context)
            end
        elseif fixed_context !== nothing
            # Check if getvarinfo method exists for FixedContext
            if hasmethod(DynamicPPL.getvarinfo, (typeof(fixed_context),))
                return DynamicPPL.getvarinfo(fixed_context)
            end
        end
    catch e
        @debug "Error accessing varinfo from context: $e"
    end

    # Fall back to the provided fallback_vi
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
    sampler::DynamicPPL.Sampler{<:GibbsConditional},
    state::DynamicPPL.AbstractVarInfo;
    kwargs...,
)
    alg = sampler.alg

    try
        # For GibbsConditional within Gibbs, we need to get all variable values
        # Model always has a context field, so we can traverse it directly
        global_vi = find_global_varinfo(model.context, state)

        # Extract conditioned values as a NamedTuple
        # Include both random variables and observed data
        # Use a safe approach for invlink to avoid linking conflicts
        invlinked_global_vi = try
            DynamicPPL.invlink(global_vi, model)
        catch e
            @debug "Failed to invlink global_vi, using as-is: $e"
            global_vi
        end

        condvals_vars = DynamicPPL.values_as(invlinked_global_vi, NamedTuple)
        condvals_obs = NamedTuple{keys(model.args)}(model.args)
        condvals = merge(condvals_vars, condvals_obs)

        # Get the conditional distribution
        conddist = alg.conditional(condvals)

        # Sample from the conditional distribution
        updated = rand(rng, conddist)

        # Update the variable in state, handling linking properly
        # The Gibbs sampler ensures that state only contains one variable
        state_is_linked = try
            DynamicPPL.islinked(state, model)
        catch e
            @debug "Error checking if state is linked: $e"
            false
        end

        if state_is_linked
            # If state is linked, we need to unlink, update, then relink
            try
                unlinked_state = DynamicPPL.invlink(state, model)
                updated_state = DynamicPPL.unflatten(unlinked_state, [updated])
                new_vi = DynamicPPL.link(updated_state, model)
            catch e
                @debug "Error in linked state update path: $e, falling back to direct update"
                new_vi = DynamicPPL.unflatten(state, [updated])
            end
        else
            # State is not linked, we can update directly
            new_vi = DynamicPPL.unflatten(state, [updated])
        end

        return nothing, new_vi

    catch e
        # If there's any error in the step, log it and rethrow
        @error "Error in GibbsConditional step: $e"
        rethrow(e)
    end
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
