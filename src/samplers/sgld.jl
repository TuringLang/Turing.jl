"""
    SGLD(n_iters::Int, step_size::Float64)

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
mutable struct SGLD{T} <: Hamiltonian
    n_iters::Int       # number of samples
    step_size::Float64   # constant scale factor of learning rate
    space::Set{T}    # sampling space, emtpy means all
    gid::Int
end
SGLD(step_size::Float64, space...) = SGLD(1, step_size, space..., 0)
SGLD(n_iters, step_size) = SGLD(n_iters, step_size, Set(), 0)
function SGLD(n_iters, step_size, space...)
    _space = isa(space, Symbol) ? Set([space]) : Set(space)
    return SGLD(n_iters, step_size, _space, 0)
end
SGLD(alg::SGLD, new_gid::Int) = SGLD(alg.n_iters, alg.step_size, alg.space, new_gid)

function step(model, spl::Sampler{<:SGLD}, vi::VarInfo, is_first::Bool)
    if is_first
        if ~haskey(spl.info, :wum)
            spl.alg.gid != 0 && link!(vi, spl)

            D = length(vi[spl])
            ve = VarEstimator{Float64}(0, zeros(D), zeros(D))
            wum = WarmUpManager(1, Dict(), ve)
            wum[:ϵ] = [spl.alg.step_size]
            wum[:stds] = ones(D)
            spl.info[:wum] = wum

            # Initialize iteration counter
            spl.info[:t] = 0

            spl.alg.gid != 0 && invlink!(vi, spl)
        end

        push!(spl.info[:accept_his], true)
    else
        # Update iteration counter
        t = deepcopy(spl.info[:t]) + 1
        spl.info[:t] = deepcopy(t)

        @debug "compute current step size..."
        γ = .35
        ϵ_t = spl.alg.step_size / t^γ # NOTE: Choose γ=.55 in paper
        push!(spl.info[:wum][:ϵ], ϵ_t)

        @debug "X-> R..."
        if spl.alg.gid != 0
            link!(vi, spl)
            runmodel(model, vi, spl)
        end

        @debug "recording old variables..."
        θ = vi[spl]
        _, grad = gradient(θ, vi, model, spl)

        if verifygrad(grad)
            @debug "update latent variables..."
            v = zeros(Float64, size(θ))
            for k in 1:size(θ, 1)
                noise = rand(MvNormal(sqrt(ϵ_t) .* ones(length(θ[k, :]))))
                θ[k, :] .-= ϵ_t .* grad[k, :] ./ 2 .+ noise
            end
        end

        @debug "always accept..."
        push!(spl.info[:accept_his], true)
        vi[spl] = θ

        @debug "R -> X..."
        spl.alg.gid != 0 && invlink!(vi, spl)
    end
    return vi
end
