"""
    SGLD(n_iters::Int, epsilon::Float64)

 Stochastic Gradient Langevin Dynamics sampler.

Usage:

```julia
SGLD(1000, 0.5)
```

Example:

```julia
@model example begin
  ...
end

sample(example, SGLD(1000, 0.5))
```
"""
mutable struct SGLD{T} <: StaticHamiltonian
    n_iters :: Int       # number of samples
    epsilon :: Float64   # constant scale factor of learning rate
    space   :: Set{T}    # sampling space, emtpy means all
    gid     :: Int
end
SGLD(epsilon::Float64, space...) = SGLD(1, epsilon, space..., 0)
SGLD(n_iters, epsilon) = SGLD(n_iters, epsilon, Set(), 0)
function SGLD(n_iters, epsilon, space...)
    _space = isa(space, Symbol) ? Set([space]) : Set(space)
    return SGLD(n_iters, epsilon, _space, 0)
end
SGLD(alg::SGLD, new_gid::Int) = SGLD(alg.n_iters, alg.epsilon, alg.space, new_gid)

function step(model, spl::Sampler{<:SGLD}, vi::VarInfo, is_first::Val{true})
    spl.alg.gid != 0 && link!(vi, spl)

    spl.info[:wum] = NaiveCompAdapter(UnitPreConditioner(), ManualSSAdapter(MSSState(spl.alg.epsilon)))

    # Initialize iteration counter
    spl.info[:t] = 0

    spl.alg.gid != 0 && invlink!(vi, spl)
    return vi, true
end

function step(model, spl::Sampler{<:SGLD}, vi::VarInfo, is_first::Val{false})
    # Update iteration counter
    spl.info[:t] += 1

    @debug "compute current step size..."
    γ = .35
    ϵ_t = spl.alg.epsilon / spl.info[:t]^γ # NOTE: Choose γ=.55 in paper
    mssa = spl.info[:wum].ssa
    mssa.state.ϵ = ϵ_t

    @debug "X-> R..."
    if spl.alg.gid != 0
        link!(vi, spl)
        runmodel!(model, vi, spl)
    end

    @debug "recording old variables..."
    θ = vi[spl]
    _, grad = gradient(θ, vi, model, spl)
    verifygrad(grad)

    @debug "update latent variables..."
    θ .-= ϵ_t .* grad ./ 2 .+ rand.(Normal.(zeros(length(θ)), sqrt(ϵ_t)))

    @debug "always accept..."
    vi[spl] = θ

    @debug "R -> X..."
    spl.alg.gid != 0 && invlink!(vi, spl)

    return vi, true
end
