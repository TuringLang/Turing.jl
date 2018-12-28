function step(model, spl::Sampler{<:SGLD}, vi::UntypedVarInfo, is_first::Val{true})
    spl.alg.gid != 0 && link!(vi, spl)

    spl.info[:wum] = NaiveCompAdapter(UnitPreConditioner(), ManualSSAdapter(MSSState(spl.alg.epsilon)))

    # Initialize iteration counter
    spl.info[:t] = 0

    spl.alg.gid != 0 && invlink!(vi, spl)
    return vi, true
end

function step(model, spl::Sampler{<:SGLD}, vi::UntypedVarInfo, is_first::Val{false})
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
