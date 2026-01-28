"""
    SGHMC{AD}

Stochastic Gradient Hamiltonian Monte Carlo (SGHMC) sampler.

# Fields
$(TYPEDFIELDS)

# Reference

Tianqi Chen, Emily Fox, & Carlos Guestrin (2014). Stochastic Gradient Hamiltonian Monte
Carlo. In: Proceedings of the 31st International Conference on Machine Learning
(pp. 1683–1691).
"""
struct SGHMC{AD,T<:Real} <: StaticHamiltonian
    learning_rate::T
    momentum_decay::T
    adtype::AD
end

"""
    SGHMC(;
        learning_rate::Real,
        momentum_decay::Real,
        adtype::ADTypes.AbstractADType = AutoForwardDiff(),
    )

Create a Stochastic Gradient Hamiltonian Monte Carlo (SGHMC) sampler.

If the automatic differentiation (AD) backend `adtype` is not provided, ForwardDiff
with automatically determined `chunksize` is used.

# Reference

Tianqi Chen, Emily Fox, & Carlos Guestrin (2014). Stochastic Gradient Hamiltonian Monte
Carlo. In: Proceedings of the 31st International Conference on Machine Learning
(pp. 1683–1691).
"""
function SGHMC(;
    learning_rate::Real,
    momentum_decay::Real,
    adtype::ADTypes.AbstractADType=Turing.DEFAULT_ADTYPE,
)
    _learning_rate, _momentum_decay = promote(learning_rate, momentum_decay)
    return SGHMC(_learning_rate, _momentum_decay, adtype)
end

struct SGHMCState{L,V<:AbstractVector{<:Real},T<:AbstractVector{<:Real}}
    logdensity::L
    params::V
    velocity::T
end

function Turing.Inference.initialstep(
    rng::Random.AbstractRNG,
    model::Model,
    spl::SGHMC,
    vi::AbstractVarInfo;
    discard_sample=false,
    kwargs...,
)
    # Transform the samples to unconstrained space.
    if !DynamicPPL.is_transformed(vi)
        vi = DynamicPPL.link!!(vi, model)
    end

    # Compute initial sample and state.
    ℓ = DynamicPPL.LogDensityFunction(
        model, DynamicPPL.getlogjoint_internal, vi; adtype=spl.adtype
    )
    initial_params = vi[:]
    sample = discard_sample ? nothing : DynamicPPL.ParamsWithStats(initial_params, ℓ)
    state = SGHMCState(ℓ, initial_params, zero(vi[:]))

    return sample, state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::Model,
    spl::SGHMC,
    state::SGHMCState;
    discard_sample=false,
    kwargs...,
)
    # Compute gradient of log density.
    ℓ = state.logdensity
    θ = state.params
    grad = last(LogDensityProblems.logdensity_and_gradient(ℓ, θ))

    # Update latent variables and velocity according to
    # equation (15) of Chen et al. (2014)
    v = state.velocity
    θ .+= v
    η = spl.learning_rate
    α = spl.momentum_decay
    newv = (1 - α) .* v .+ η .* grad .+ sqrt(2 * η * α) .* randn(rng, eltype(v), length(v))

    # Compute next sample and state.
    sample = discard_sample ? nothing : DynamicPPL.ParamsWithStats(θ, ℓ)
    newstate = SGHMCState(ℓ, θ, newv)

    return sample, newstate
end

"""
    SGLD

Stochastic gradient Langevin dynamics (SGLD) sampler.

# Fields
$(TYPEDFIELDS)

# Reference

Max Welling & Yee Whye Teh (2011). Bayesian Learning via Stochastic Gradient Langevin
Dynamics. In: Proceedings of the 28th International Conference on Machine Learning
(pp. 681–688).
"""
struct SGLD{AD,S} <: StaticHamiltonian
    "Step size function."
    stepsize::S
    adtype::AD
end

struct PolynomialStepsize{T<:Real}
    "Constant scale factor of the step size."
    a::T
    "Constant offset of the step size."
    b::T
    "Decay rate of step size in (0.5, 1]."
    γ::T

    function PolynomialStepsize{T}(a::T, b::T, γ::T) where {T}
        0.5 < γ ≤ 1 || error("the decay rate `γ` has to be in (0.5, 1]")
        return new{T}(a, b, γ)
    end
end

"""
    PolynomialStepsize(a[, b=0, γ=0.55])

Create a polynomially decaying stepsize function.

At iteration `t`, the step size is
```math
a (b + t)^{-γ}.
```
"""
function PolynomialStepsize(a::T, b::T, γ::T) where {T<:Real}
    return PolynomialStepsize{T}(a, b, γ)
end
function PolynomialStepsize(a::Real, b::Real=0, γ::Real=0.55)
    return PolynomialStepsize(promote(a, b, γ)...)
end

(f::PolynomialStepsize)(t::Int) = f.a / (t + f.b)^f.γ

"""
    SGLD(;
        stepsize = PolynomialStepsize(0.01),
        adtype::ADTypes.AbstractADType = AutoForwardDiff(),
    )

Stochastic gradient Langevin dynamics (SGLD) sampler.

By default, a polynomially decaying stepsize is used.

If the automatic differentiation (AD) backend `adtype` is not provided, ForwardDiff
with automatically determined `chunksize` is used.

# Reference

Max Welling & Yee Whye Teh (2011). Bayesian Learning via Stochastic Gradient Langevin
Dynamics. In: Proceedings of the 28th International Conference on Machine Learning
(pp. 681–688).

See also: [`PolynomialStepsize`](@ref)
"""
function SGLD(;
    stepsize=PolynomialStepsize(0.01), adtype::ADTypes.AbstractADType=Turing.DEFAULT_ADTYPE
)
    return SGLD(stepsize, adtype)
end

struct SGLDState{L,V<:AbstractVector{<:Real}}
    logdensity::L
    params::V
    step::Int
end

function Turing.Inference.initialstep(
    rng::Random.AbstractRNG,
    model::Model,
    spl::SGLD,
    vi::AbstractVarInfo;
    discard_sample=false,
    kwargs...,
)
    # Transform the samples to unconstrained space.
    if !DynamicPPL.is_transformed(vi)
        vi = DynamicPPL.link!!(vi, model)
    end

    # Create first sample and state.
    ℓ = DynamicPPL.LogDensityFunction(
        model, DynamicPPL.getlogjoint_internal, vi; adtype=spl.adtype
    )
    initial_params = vi[:]
    transition = if discard_sample
        nothing
    else
        stats = (; SGLD_stepsize=zero(spl.stepsize(0)))
        DynamicPPL.ParamsWithStats(initial_params, ℓ, stats)
    end
    state = SGLDState(ℓ, initial_params, 1)

    return transition, state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::Model,
    spl::SGLD,
    state::SGLDState;
    discard_sample=false,
    kwargs...,
)
    # Perform gradient step.
    ℓ = state.logdensity
    θ = state.params
    grad = last(LogDensityProblems.logdensity_and_gradient(ℓ, θ))
    step = state.step
    stepsize = spl.stepsize(step)
    θ .+= (stepsize / 2) .* grad .+ sqrt(stepsize) .* randn(rng, eltype(θ), length(θ))

    # Compute next sample and state.
    transition = if discard_sample
        nothing
    else
        stats = (; SGLD_stepsize=stepsize)
        DynamicPPL.ParamsWithStats(θ, ℓ, stats)
    end
    newstate = SGLDState(ℓ, θ, state.step + 1)

    return transition, newstate
end
