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

struct SGHMCState{L,V<:AbstractVarInfo,T<:AbstractVector{<:Real}}
    logdensity::L
    vi::V
    velocity::T
end

function DynamicPPL.initialstep(
    rng::Random.AbstractRNG,
    model::Model,
    spl::Sampler{<:SGHMC},
    vi::AbstractVarInfo;
    kwargs...,
)
    # Transform the samples to unconstrained space and compute the joint log probability.
    if !DynamicPPL.islinked(vi)
        vi = DynamicPPL.link!!(vi, model)
        vi = last(DynamicPPL.evaluate!!(model, vi))
    end

    # Compute initial sample and state.
    sample = Transition(model, vi)
    ℓ = DynamicPPL.LogDensityFunction(
        model,
        vi,
        DynamicPPL.SamplingContext(spl, DynamicPPL.DefaultContext());
        adtype=spl.alg.adtype,
    )
    state = SGHMCState(ℓ, vi, zero(vi[:]))

    return sample, state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::Model,
    spl::Sampler{<:SGHMC},
    state::SGHMCState;
    kwargs...,
)
    # Compute gradient of log density.
    ℓ = state.logdensity
    vi = state.vi
    θ = vi[:]
    grad = last(LogDensityProblems.logdensity_and_gradient(ℓ, θ))

    # Update latent variables and velocity according to
    # equation (15) of Chen et al. (2014)
    v = state.velocity
    θ .+= v
    η = spl.alg.learning_rate
    α = spl.alg.momentum_decay
    newv = (1 - α) .* v .+ η .* grad .+ sqrt(2 * η * α) .* randn(rng, eltype(v), length(v))

    # Save new variables and recompute log density.
    vi = DynamicPPL.unflatten(vi, θ)
    vi = last(DynamicPPL.evaluate!!(model, vi))

    # Compute next sample and state.
    sample = Transition(model, vi)
    newstate = SGHMCState(ℓ, vi, newv)

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

struct SGLDTransition{T,F<:Real} <: AbstractTransition
    "The parameters for any given sample."
    θ::T
    "The joint log probability of the sample."
    lp::F
    "The stepsize that was used to obtain the sample."
    stepsize::F
end

function SGLDTransition(model::DynamicPPL.Model, vi::AbstractVarInfo, stepsize)
    theta = getparams(model, vi)
    lp = DynamicPPL.getlogjoint_internal(vi)
    return SGLDTransition(theta, lp, stepsize)
end

metadata(t::SGLDTransition) = (lp=t.lp, SGLD_stepsize=t.stepsize)

DynamicPPL.getlogp(t::SGLDTransition) = t.lp

struct SGLDState{L,V<:AbstractVarInfo}
    logdensity::L
    vi::V
    step::Int
end

function DynamicPPL.initialstep(
    rng::Random.AbstractRNG,
    model::Model,
    spl::Sampler{<:SGLD},
    vi::AbstractVarInfo;
    kwargs...,
)
    # Transform the samples to unconstrained space and compute the joint log probability.
    if !DynamicPPL.islinked(vi)
        vi = DynamicPPL.link!!(vi, model)
        vi = last(DynamicPPL.evaluate!!(model, vi))
    end

    # Create first sample and state.
    sample = SGLDTransition(model, vi, zero(spl.alg.stepsize(0)))
    ℓ = DynamicPPL.LogDensityFunction(
        model,
        vi,
        DynamicPPL.SamplingContext(spl, DynamicPPL.DefaultContext());
        adtype=spl.alg.adtype,
    )
    state = SGLDState(ℓ, vi, 1)

    return sample, state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG, model::Model, spl::Sampler{<:SGLD}, state::SGLDState; kwargs...
)
    # Perform gradient step.
    ℓ = state.logdensity
    vi = state.vi
    θ = vi[:]
    grad = last(LogDensityProblems.logdensity_and_gradient(ℓ, θ))
    step = state.step
    stepsize = spl.alg.stepsize(step)
    θ .+= (stepsize / 2) .* grad .+ sqrt(stepsize) .* randn(rng, eltype(θ), length(θ))

    # Save new variables and recompute log density.
    vi = DynamicPPL.unflatten(vi, θ)
    vi = last(DynamicPPL.evaluate!!(model, vi))

    # Compute next sample and state.
    sample = SGLDTransition(model, vi, stepsize)
    newstate = SGLDState(ℓ, vi, state.step + 1)

    return sample, newstate
end
