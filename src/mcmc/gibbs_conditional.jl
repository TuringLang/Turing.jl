using DynamicPPL: VarName
using Random: Random
import AbstractMCMC

"""
    GibbsConditional(conditional)

A Gibbs component sampler that samples variables according to user-provided
analytical conditional distributions.

`conditional` should be a function that takes a `Dict{<:VarName}` of conditioned variables
and their values, and returns one of the following:
- A single `Distribution`, if only one variable is being sampled.
- An `AbstractDict{<:VarName,<:Distribution}` that maps the variables being sampled to their
  `Distribution`s.
- A `NamedTuple` of `Distribution`s, which is like the `AbstractDict` case but can be used
  if all the variable names are single `Symbol`s, and may be more performant.

If a Gibbs component is created with `(:var1, :var2) => GibbsConditional(conditional)`, then
`var1` and `var2` should be in the keys of the return value of `conditional`.

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
function cond_λ(c)
    a = 2.0
    b = inv(3)
    m = c[@varname(m)]
    x = c[@varname(x)]
    n = length(x)
    a_new = a + (n + 1) / 2
    b_new = b + sum((x[i] - m)^2 for i in 1:n) / 2 + m^2 / 2
    return Gamma(a_new, 1 / b_new)
end

function cond_m(c)
    λ = c[@varname(λ)]
    x = c[@varname(x)]
    n = length(x)
    m_mean = sum(x) / (n + 1)
    m_var = 1 / (λ * (n + 1))
    return Normal(m_mean, sqrt(m_var))
end

# Sample using GibbsConditional
model = inverse_gdemo([1.0, 2.0, 3.0])
chain = sample(model, Gibbs(
    :λ => GibbsConditional(cond_λ),
    :m => GibbsConditional(cond_m)
), 1000)
```
"""
struct GibbsConditional{C} <: AbstractSampler
    conditional::C
end

# Mark GibbsConditional as a valid Gibbs component
isgibbscomponent(::GibbsConditional) = true

"""
    build_variable_dict(model::DynamicPPL.Model)

Traverse the context stack of `model` and build a `Dict` of all the variable values that are
set in GibbsContext, ConditionContext, or FixedContext.
"""
function build_variable_dict(model::DynamicPPL.Model)
    context = model.context
    cond_nt = DynamicPPL.conditioned(context)
    fixed_nt = DynamicPPL.fixed(context)
    # TODO(mhauru) Can we avoid invlinking all the time?
    global_vi = DynamicPPL.invlink(get_gibbs_global_varinfo(context), model)
    # TODO(mhauru) Double-check that the ordered of precedence here is correct. Should we
    # in fact error if there is any overlap in the keys?
    return merge(
        DynamicPPL.values_as(global_vi, Dict),
        Dict(
            (DynamicPPL.VarName{sym}() => val for (sym, val) in pairs(cond_nt))...,
            (DynamicPPL.VarName{sym}() => val for (sym, val) in pairs(fixed_nt))...,
            (DynamicPPL.VarName{sym}() => val for (sym, val) in pairs(model.args))...,
        ),
    )
end

function get_gibbs_global_varinfo(context::DynamicPPL.AbstractContext)
    return if context isa GibbsContext
        get_global_varinfo(context)
    elseif DynamicPPL.NodeTrait(context) isa DynamicPPL.IsParent
        get_gibbs_global_varinfo(DynamicPPL.childcontext(context))
    else
        throw(ArgumentError("""No GibbsContext found in context stack. \
                            Are you trying to use GibbsConditional outside of Gibbs?
                            """))
    end
end

"""
    DynamicPPL.initialstep(rng, model, sampler::GibbsConditional, vi)

Initialize the GibbsConditional sampler.
"""
function initialstep(
    ::Random.AbstractRNG,
    model::DynamicPPL.Model,
    ::GibbsConditional,
    vi::DynamicPPL.AbstractVarInfo;
    kwargs...,
)
    state = DynamicPPL.is_transformed(vi) ? DynamicPPL.invlink(vi, model) : vi
    # Since GibbsConditional is only used within Gibbs, it does not need to return a
    # transition.
    return nothing, state
end

"""
    AbstractMCMC.step(rng, model, sampler::GibbsConditional, state)

Perform a step of GibbsConditional sampling.
"""
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    sampler::GibbsConditional,
    state::DynamicPPL.AbstractVarInfo;
    kwargs...,
)
    # Get all the conditioned variable values from the model context. This is assumed to
    # include a GibbsContext as part of the context stack.
    condvals = build_variable_dict(model)

    # Get the conditional distributions
    conddists = sampler.conditional(condvals)

    # We support three different kinds of return values for `sample.conditional`, to make
    # life easier for the user.
    if conddists isa AbstractDict
        for (vn, dist) in conddists
            state = setindex!!(state, rand(rng, dist), vn)
        end
    elseif conddists isa NamedTuple
        for (vn_sym => dist) in pairs(conddists)
            vn = VarName{vn_sym}()
            state = setindex!!(state, rand(rng, dist), vn)
        end
    else
        # Single variable case
        vn = first(keys(state))
        state = setindex!!(state, rand(rng, conddists), vn)
    end

    # Since GibbsConditional is only used within Gibbs, it does not need to return a
    # transition.
    return nothing, state
end

function setparams_varinfo!!(
    model::DynamicPPL.Model,
    sampler::GibbsConditional,
    state,
    params::DynamicPPL.AbstractVarInfo,
)
    return params
end
