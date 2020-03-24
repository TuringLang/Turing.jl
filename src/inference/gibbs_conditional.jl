"""
    GibbsConditional(sym, conditional)

A "pseudo-sampler" to manually provide analytical Gibbs conditionals to `Gibbs`.
`GibbsConditional(:x, cond)` will sample the variable `x` according to the conditional `cond`, which
must therefore be a function from a `NamedTuple` of the conditioned variables to a `Distribution`.

# Examples

```julia
α₀ = 2.0
θ₀ = inv(3.0)

x = [1.5, 2.0]

function gdemo_statistics(x)
    # The conditionals and posterior can be formulated in terms of the following statistics:
    N = length(x) # number of samples
    x̄ = mean(x) # sample mean
    s² = var(x; mean=x̄, corrected=false) # sample variance
    return N, x̄, s²
end

function gdemo_cond_m(c)
    # c = (λ = ...,)
    N, x̄, s² = gdemo_statistics(x)
    mₙ = N * x̄ / (N + 1)
    λₙ = c.λ * (N + 1)
    σₙ = √(1 / λₙ)
    return Normal(mₙ, σₙ)
end

function gdemo_cond_λ(c)
    # c = (m = ...,)
    N, x̄, s² = gdemo_statistics(x)
    αₙ = α₀ + (N - 1) / 2
    βₙ = (s² * N / 2 + c.m^2 / 2 + inv(θ₀))
    return Gamma(αₙ, inv(βₙ))
end

@model gdemo(x) = begin
    λ ~ Gamma(α₀, θ₀)
    m ~ Normal(0, √(1 / λ))
    x .~ Normal(m, √(1 / λ))
end

m = gdemo(x)

sample(m, Gibbs(GibbsConditional(:λ, gdemo_cond_λ), GibbsConditional(:m, gdemo_cond_m)), 10)
```
"""
struct GibbsConditional{S, C}
    conditional::C

    function GibbsConditional(sym::Symbol, conditional::C) where {C}
        return new{sym, C}(conditional)
    end
end

getspace(::GibbsConditional{S}) where {S} = (S,)
alg_str(::GibbsConditional) = "GibbsConditional"
isgibbscomponent(::GibbsConditional) = true


function Sampler(
    alg::GibbsConditional,
    model::Model,
    s::Selector=Selector()
)
    return Sampler(alg, Dict{Symbol, Any}(), s, SamplerState(VarInfo(model)))
end


function gibbs_step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:GibbsConditional{S}},
    N::Integer,
    transition;
    kwargs...
) where {S}
    if spl.selector.rerun # Recompute joint in logp
        runmodel!(model, spl.state.vi)
    end

    condvals = conditioned(tonamedtuple(spl.state.vi), Val{S}())
    conddist = spl.alg.conditional(condvals)
    updated = rand(rng, conddist)
    spl.state.vi[VarName{S}("")] = [updated;]  # setindex allows only vectors in this case...
    
    return transition
end


"""
    conditioned(θ::NamedTuple, ::Val{S})

Extract a `NamedTuple` of the values in `θ` conditioned on `S`; i.e., all names of `θ` except for
`S`, mapping to their respecitve values.

`θ` is assumed to come from `tonamedtuple(vi)`, which returns a `NamedTuple` of the form

```julia
t = (m = ([0.234, -1.23], ["m[1]", "m[2]"]), λ = ([1.233], ["λ"])
```

so this function does both the cleanup of indexing and filtering by name. `conditioned(t, Val{m}())`
and `conditioned(t, Val{λ}())` will therefore return

```julia
(λ = 1.233,)
```

and

```julia
(m = [0.234, -1.23],)
```
"""
@generated function conditioned(θ::NamedTuple{names}, ::Val{S}) where {names, S}
    condvals = [:($n = extractparam(θ.$n)) for n in names if n ≠ S]
    return Expr(:tuple, condvals...)
end


"""Takes care of removing the `tonamedtuple` indexing form."""
extractparam(p::Tuple{Vector{<:Array{<:Real}}, Vector{String}}) = foldl(vcat, p[1])
function extractparam(p::Tuple{Vector{<:Real}, Vector{String}})
    values, strings = p
    if length(values) == length(strings) == 1 && !occursin(r".\[.+\]$", strings[1])
        # if m ~ MVNormal(1, 1), we could have have ([1], ["m[1]"])!
        return values[1]
    else
        return values
    end
end
