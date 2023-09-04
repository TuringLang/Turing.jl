"""
    GibbsConditional(sym, conditional)

A "pseudo-sampler" to manually provide analytical Gibbs conditionals to `Gibbs`.
`GibbsConditional(:x, cond)` will sample the variable `x` according to the conditional `cond`, which
must therefore be a function from a `NamedTuple` of the conditioned variables to a `Distribution`.


The `NamedTuple` that is passed in contains all random variables from the model in an unspecified
order, taken from the [`VarInfo`](@ref) object over which the model is run. Scalars and vectors are
stored in their respective shapes. The tuple also contains the value of the conditioned variable
itself, which can be useful, but using it creates something that is not a Gibbs sampler anymore (see
[here](https://github.com/TuringLang/Turing.jl/pull/1275#discussion_r434240387)).

# Examples

```julia
α_0 = 2.0
θ_0 = inv(3.0)
x = [1.5, 2.0]
N = length(x)

@model function inverse_gdemo(x)
    λ ~ Gamma(α_0, θ_0)
    σ = sqrt(1 / λ)
    m ~ Normal(0, σ)
    @. x ~ \$(Normal(m, σ))
end

# The conditionals can be formulated in terms of the following statistics:
x_bar = mean(x) # sample mean
s2 = var(x; mean=x_bar, corrected=false) # sample variance
m_n = N * x_bar / (N + 1)

function cond_m(c)
    λ_n = c.λ * (N + 1)
    σ_n = sqrt(1 / λ_n)
    return Normal(m_n, σ_n)
end

function cond_λ(c)
    α_n = α_0 + (N - 1) / 2 + 1
    β_n = s2 * N / 2 + c.m^2 / 2 + inv(θ_0)
    return Gamma(α_n, inv(β_n))
end

m = inverse_gdemo(x)

sample(m, Gibbs(GibbsConditional(:λ, cond_λ), GibbsConditional(:m, cond_m)), 10)
```
"""
struct GibbsConditional{S, C}
    conditional::C

    function GibbsConditional(sym::Symbol, conditional::C) where {C}
        return new{sym, C}(conditional)
    end
end

DynamicPPL.getspace(::GibbsConditional{S}) where {S} = (S,)

function DynamicPPL.initialstep(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:GibbsConditional},
    vi::AbstractVarInfo;
    kwargs...
)
    return nothing, vi
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:GibbsConditional},
    vi::AbstractVarInfo;
    kwargs...
)
    condvals = DynamicPPL.values_as(DynamicPPL.invlink(vi, model), NamedTuple)
    conddist = spl.alg.conditional(condvals)
    updated = rand(rng, conddist)
    # Setindex allows only vectors in this case.
    vi = setindex!!(vi, [updated;], spl)
    # Update log joint probability.
    vi = last(DynamicPPL.evaluate!!(model, vi, DefaultContext()))

    return nothing, vi
end
