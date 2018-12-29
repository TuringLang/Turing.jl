function step(model, spl::Sampler{<:SGHMC}, vi::AbstractVarInfo, is_first::Val{true})
    spl.alg.gid != 0 && link!(vi, spl)

    # Initialize velocity
    v = zeros(Float64, size(vi[spl]))
    spl.info[:v] = v

    spl.alg.gid != 0 && invlink!(vi, spl)
    return vi, true
end

function step(model, spl::Sampler{<:SGHMC}, vi::AbstractVarInfo, is_first::Val{false})
    # Set parameters
    η, α = spl.alg.learning_rate, spl.alg.momentum_decay

    @debug "X-> R..."
    if spl.alg.gid != 0
        link!(vi, spl)
        runmodel!(model, vi, spl)
    end

    @debug "recording old variables..."
    θ, v = vi[spl], spl.info[:v]
    _, grad = gradient(θ, vi, model, spl)
    verifygrad(grad)

    # Implements the update equations from (15) of Chen et al. (2014).
    @debug "update latent variables and velocity..."
    θ .+= v
    v .= (1 - α) .* v .- η .* grad .+ rand.(Normal.(zeros(length(θ)), sqrt(2 * η * α)))

    @debug "saving new latent variables..."
    vi[spl] = θ

    @debug "R -> X..."
    spl.alg.gid != 0 && invlink!(vi, spl)

    @debug "always accept..."
    return vi, true
end
