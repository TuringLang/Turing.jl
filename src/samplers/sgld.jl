"""
    SGLD(n_iters::Int, epsilon::Float64)

 Stochastic Gradient Langevin Dynamics sampler.

Usage:

```julia
SGLD(1000, 0.5)
```

Arguments:

- `n_iters::Int` : Number of samples to pull.
- `epsilon::Float64` : The scaling factor for the learing rate.

Example:

```julia
@model example begin
  ...
end

sample(example, SGLD(1000, 0.5))
```
"""
mutable struct SGLD{AD, T} <: StaticHamiltonian{AD}
    n_iters :: Int       # number of samples
    epsilon :: Float64   # constant scale factor of learning rate
    space   :: Set{T}    # sampling space, emtpy means all
    gid     :: Int
end
SGLD(args...; kwargs...) = SGLD{ADBackend()}(args...; kwargs...)
function SGLD{AD}(epsilon::Float64, space...) where AD 
    _space = isa(space, Symbol) ? Set([space]) : Set(space)    
    SGLD{AD, eltype(_space)}(1, epsilon, _space, 0)
end
function SGLD{AD}(n_iters, epsilon) where AD
    SGLD{AD, Any}(n_iters, epsilon, Set(), 0)
end
function SGLD{AD}(n_iters, epsilon, space...) where AD
    _space = isa(space, Symbol) ? Set([space]) : Set(space)
    return SGLD{AD, eltype(_space)}(n_iters, epsilon, _space, 0)
end
function SGLD{AD1}(alg::SGLD{AD2, T}, new_gid::Int) where {AD1, AD2, T}
    SGLD{AD1, T}(alg.n_iters, alg.epsilon, alg.space, new_gid)
end
function SGLD{AD, T}(alg::SGLD, new_gid::Int) where {AD, T}
    SGLD{AD, T}(alg.n_iters, alg.epsilon, alg.space, new_gid)
end

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
