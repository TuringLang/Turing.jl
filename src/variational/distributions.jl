import Distributions: _rand!


function jac_inv_transform(dist::Distribution, x::Real)
    ForwardDiff.derivative(x -> invlink(dist, x), x)
end

function jac_inv_transform(dist::Distribution, x::AbstractArray{<:Real})
    ForwardDiff.jacobian(x -> invlink(dist, x), x)
end

function jac_inv_transform(dist::Distribution, x::TrackedArray{<:Real})
    Tracker.jacobian(x -> invlink(dist, x), x)
end

# instead of creating a diagonal matrix, we just do elementwise multiplication
center_diag_gaussian(x, μ, σ) = (x .- μ) ./ σ
center_diag_gaussian_inv(η, μ, σ) = (η .* σ) .+ μ


# Mean-field approximation used by ADVI
struct MeanField{TDists, V} <: VariationalPosterior where {V <: AbstractVector{<: Real}, TDists <: AbstractVector{<: Distribution}}
    μ::V
    ω::V
    dists::TDists
    ranges::Vector{UnitRange{Int}}
end

Base.length(advi::MeanField) = length(advi.μ)

function _rand!(
    rng::AbstractRNG,
    q::MeanField,
    x::AbstractVector{<:Real}
)
    # extract parameters for convenience
    μ, ω = q.μ, q.ω
    num_params = length(q)

    for i = 1:size(q.dists, 1)
        prior = q.dists[i]
        r = q.ranges[i]

        # initials
        μ_i = μ[r]
        ω_i = ω[r]

        # # sample from VI posterior
        η = randn(rng, length(μ_i))
        ζ = center_diag_gaussian_inv(η, μ_i, exp.(ω_i))
        θ = invlink(prior, ζ)

        x[r] = θ
    end

    return x
end
