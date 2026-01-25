"""
    IS()

Importance sampling algorithm.

Usage:

```julia
IS()
```

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model function gdemo(x)
    s² ~ InverseGamma(2,3)
    m ~ Normal(0,sqrt.(s))
    x[1] ~ Normal(m, sqrt.(s))
    x[2] ~ Normal(m, sqrt.(s))
    return s², m
end

sample(gdemo([1.5, 2]), IS(), 1000)
```
"""
struct IS <: AbstractSampler end

function Turing.Inference.initialstep(
    rng::AbstractRNG, model::Model, spl::IS, vi::AbstractVarInfo; kwargs...
)
    return DynamicPPL.ParamsWithStats(vi, model), nothing
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG, model::Model, spl::IS, ::Nothing; kwargs...
)
    model = DynamicPPL.setleafcontext(model, ISContext(rng))
    _, vi = DynamicPPL.evaluate!!(model, DynamicPPL.VarInfo())
    return DynamicPPL.ParamsWithStats(vi, model), nothing
end

struct ISContext{R<:AbstractRNG} <: DynamicPPL.AbstractContext
    rng::R
end

function DynamicPPL.tilde_assume!!(
    ctx::ISContext, dist::Distribution, vn::VarName, template, vi::AbstractVarInfo
)
    if haskey(vi, vn)
        tval = vi.values[vn]
        val, logjac = with_logabsdet_jacobian(
            DynamicPPL.get_transform(tval), DynamicPPL.get_internal_value(tval)
        )
    else
        val = rand(ctx.rng, dist)
        vi, logjac, tval = DynamicPPL.setindex_with_dist!!(vi, val, dist, vn, template)
    end
    vi = DynamicPPL.accumulate_assume!!(vi, val, tval, logjac, vn, dist, template)
    return val, vi
end
function DynamicPPL.tilde_observe!!(
    ::ISContext, right::Distribution, left, vn::Union{VarName,Nothing}, vi::AbstractVarInfo
)
    return DynamicPPL.tilde_observe!!(DefaultContext(), right, left, vn, vi)
end
