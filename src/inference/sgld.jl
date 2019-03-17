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
mutable struct SGLD{AD, space} <: StaticHamiltonian{AD, space}
    n_iters :: Int       # number of samples
    epsilon :: Float64   # constant scale factor of learning rate
    gid     :: Int
end
SGLD(args...) = SGLD{ADBackend()}(args...)
function SGLD{AD}(epsilon::Float64, space...) where AD 
    SGLD{AD, space}(1, epsilon, 0)
end
function SGLD{AD}(n_iters, epsilon) where AD
    SGLD{AD, ()}(n_iters, epsilon, 0)
end
function SGLD{AD}(n_iters, epsilon, space...) where AD
    return SGLD{AD, space}(n_iters, epsilon, 0)
end
function SGLD{AD1}(alg::SGLD{AD2, space}, new_gid::Int) where {AD1, AD2, space}
    SGLD{AD1, space}(alg.n_iters, alg.epsilon, new_gid)
end
function SGLD{AD, space}(alg::SGLD, new_gid::Int) where {AD, space}
    SGLD{AD, space}(alg.n_iters, alg.epsilon, new_gid)
end

mutable struct SGLDInfo{Tt, Twum, Tidcs, Tranges}
    t::Tt
    wum::Twum
    cache_updated::UInt8
    idcs::Tidcs
    ranges::Tranges
    progress::ProgressMeter.Progress
    lf_num::Int
    eval_num::Int
end

function init_adapter(alg::SGLD)
    return NaiveCompAdapter(UnitPreConditioner(), ManualSSAdapter(MSSState(alg.epsilon)))
end

function init_spl(model, alg::SGLD; kwargs...)
    wum = init_adapter(alg)
    vi = VarInfo(model)
    idcs = VarReplay._getidcs(vi, Sampler(alg, nothing))
    ranges = VarReplay._getranges(vi, Sampler(alg, nothing), idcs)
    progress = ProgressMeter.Progress(alg.n_iters, 1, "[SGLD] Sampling...", 0)

    vi = VarInfo(model)
    info = SGLDInfo(0, wum, CACHERESET, idcs, ranges, progress, 0, 1)
    spl = Sampler(alg, info)
    return spl, vi
end

function step(model, spl::Sampler{<:SGLD}, vi::AbstractVarInfo, is_first::Val{true})
    spl.alg.gid != 0 && link!(vi, spl)
    spl.info.wum = init_adapter(spl.alg)

    # Initialize iteration counter
    spl.info.t = 0
    spl.alg.gid != 0 && invlink!(vi, spl)
    return vi, true
end

function step(model, spl::Sampler{<:SGLD}, vi::AbstractVarInfo, is_first::Val{false})
    # Update iteration counter
    spl.info.t += 1

    Turing.DEBUG && @debug "compute current step size..."
    γ = .35
    ϵ_t = spl.alg.epsilon / spl.info.t^γ # NOTE: Choose γ=.55 in paper
    mssa = spl.info.wum.ssa
    mssa.state.ϵ = ϵ_t

    Turing.DEBUG && @debug "X-> R..."
    if spl.alg.gid != 0
        link!(vi, spl)
        runmodel!(model, vi, spl)
    end

    Turing.DEBUG && @debug "recording old variables..."
    θ = vi[spl]
    _, grad = gradient_logp(θ, vi, model, spl)
    verifygrad(grad)

    Turing.DEBUG && @debug "update latent variables..."
    θ .+= ϵ_t .* grad ./ 2 .- rand.(Normal.(zeros(length(θ)), sqrt(ϵ_t)))

    Turing.DEBUG && @debug "always accept..."
    vi[spl] = θ

    Turing.DEBUG && @debug "R -> X..."
    spl.alg.gid != 0 && invlink!(vi, spl)

    return vi, true
end
