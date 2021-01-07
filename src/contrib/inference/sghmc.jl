"""
    SGHMC{AD,space}

Stochastic Gradient Hamiltonian Monte Carlo (SGHMC) sampler.e

# Fields
$(TYPEDFIELDS)

# Reference

Tianqi Chen, Emily Fox, & Carlos Guestrin (2014). Stochastic Gradient Hamiltonian Monte
Carlo. In: Proceedings of the 31st International Conference on Machine Learning
(pp. 1683–1691).
"""
struct SGHMC{AD,space,T<:Real} <: StaticHamiltonian{AD}
    learning_rate::T
    momentum_decay::T
end

"""
    SGHMC{AD}(space::Symbol...; learning_rate, momentum_decay)

Create a Stochastic Gradient Hamiltonian Monte Carlo (SGHMC) sampler.

If the automatic differentiation backend `AD` is not provided, the currently activated
AD backend in Turing is used.

# Reference

Tianqi Chen, Emily Fox, & Carlos Guestrin (2014). Stochastic Gradient Hamiltonian Monte
Carlo. In: Proceedings of the 31st International Conference on Machine Learning
(pp. 1683–1691).
"""
SGHMC(args...; kwargs...) = SGHMC{ADBackend()}(args...; kwargs...)
function SGHMC{AD}(
    space::Symbol...;
    learning_rate::Real,
    momentum_decay::Real,
) where {AD}
    _learning_rate, _momentum_decay = promote(learning_rate, momentum_decay)
    return SGHMC{AD,space,typeof(_learning_rate)}(_learning_rate, _momentum_decay)
end

struct SGHMCState{V<:AbstractVarInfo}
    vi::V
    velocity::Vector{Float64}
end

function DynamicPPL.initialstep(
    rng::Random.AbstractRNG,
    model::Model,
    spl::Sampler{<:SGHMC},
    vi::AbstractVarInfo;
    kwargs...,
)
    # Transform the samples to unconstrained space and compute the joint log probability.
    if !DynamicPPL.islinked(vi, spl)
        DynamicPPL.link!(vi, spl)
        model(rng, vi, spl)
    end

    # Compute initial sample and state.
    sample = Transition(vi)
    state = SGHMCState(vi, zero(vi[spl]))

    return sample, state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::Model,
    spl::Sampler{<:SGHMC},
    state::SGHMCState;
    kwargs...
)
    # Compute gradient of log density.
    vi = state.vi
    θ = vi[spl]
    _, grad = gradient_logp(θ, vi, model, spl)

    # Update latent variables and velocity according to
    # equation (15) of Chen et al. (2014)
    v = state.velocity
    θ .+= v
    η = spl.alg.learning_rate
    α = spl.alg.momentum_decay
    newv = (1 - α) .* v .+ η .* grad .+ sqrt(2 * η * α) .* randn(rng, length(v))

    # Save new variables and recompute log density.
    vi[spl] = θ
    model(rng, vi, spl)

    # Compute next sample and state.
    sample = Transition(vi)
    newstate = SGHMCState(vi, newv)

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
struct SGLD{AD,space,S} <: StaticHamiltonian{AD}
    "Step size function."
    stepsize::S
end

struct PolynomialStepsize{T<:Real}
    "Constant scale factor of the step size."
    a::T
    "Constant offset of the step size."
    b::T
    "Decay rate of step size in (0.5, 1]."
    γ::T

    function PolynomialStepsize{T}(a::T, b::T, γ::T) where T
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
function PolynomialStepsize(a::Real, b::Real = 0, γ::Real = 0.55)
    return PolynomialStepsize(promote(a, b, γ)...)
end

(f::PolynomialStepsize)(t::Int) = f.a / (t + f.b)^f.γ

"""
    SGLD{AD}(space::Symbol...; stepsize = PolynomialStepsize(0.01))

Stochastic gradient Langevin dynamics (SGLD) sampler.

By default, a polynomially decaying stepsize is used.

# Reference

Max Welling & Yee Whye Teh (2011). Bayesian Learning via Stochastic Gradient Langevin
Dynamics. In: Proceedings of the 28th International Conference on Machine Learning
(pp. 681–688).

See also: [`PolynomialStepsize`](@ref)
"""
SGLD(args...; kwargs...) = SGLD{ADBackend()}(args...; kwargs...)
function SGLD{AD}(
    space::Symbol...;
    stepsize = PolynomialStepsize(0.01),
) where {AD}
    return SGLD{AD,space,typeof(stepsize)}(stepsize)
end

struct SGLDTransition{T,F<:Real}
    "The parameters for any given sample."
    θ::T
    "The joint log probability of the sample."
    lp::F
    "The stepsize that was used to obtain the sample."
    stepsize::F
end

function SGLDTransition(vi::AbstractVarInfo, stepsize)
    theta = tonamedtuple(vi)
    lp = getlogp(vi)
    return SGLDTransition(theta, lp, stepsize)
end

metadata(t::SGLDTransition) = (lp = t.lp, SGLD_stepsize = t.stepsize)

DynamicPPL.getlogp(t::SGLDTransition) = t.lp

struct SGLDState{V<:AbstractVarInfo}
    vi::V
    step::Int
end

function DynamicPPL.initialstep(
    rng::Random.AbstractRNG,
    model::Model,
    spl::Sampler{<:SGLD},
    vi::AbstractVarInfo;
    kwargs...
)
    # Transform the samples to unconstrained space and compute the joint log probability.
    if !DynamicPPL.islinked(vi, spl)
        DynamicPPL.link!(vi, spl)
        model(rng, vi, spl)
    end

    # Create first sample and state.
    sample = SGLDTransition(vi, zero(spl.alg.stepsize(0)))
    state = SGLDState(vi, 1)

    return sample, state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::Model,
    spl::Sampler{<:SGLD},
    state::SGLDState;
    kwargs...
)
    # Perform gradient step.
    vi = state.vi
    θ = vi[spl]
    _, grad = gradient_logp(θ, vi, model, spl)
    step = state.step
    stepsize = spl.alg.stepsize(step)
    θ .+= (stepsize / 2) .* grad .+ sqrt(stepsize) .* randn(rng, length(θ))

    # Save new variables and recompute log density.
    vi[spl] = θ
    model(rng, vi, spl)

    # Compute next sample and state.
    sample = SGLDTransition(vi, stepsize)
    newstate = SGLDState(vi, state.step + 1)

    return sample, newstate
end
