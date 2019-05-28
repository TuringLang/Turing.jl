import Distributions: _rand!


function jac_inv_transform(dist::Distribution, x::T where T<:Real)
    ForwardDiff.derivative(x -> invlink(dist, x), x)
end

function jac_inv_transform(dist::Distribution, x::Array{T} where T <: Real)
    ForwardDiff.jacobian(x -> invlink(dist, x), x)
end

function center_diag_gaussian(x, μ, σ)
    # instead of creating a diagonal matrix, we just do elementwise multiplication
    (σ .^(-1)) .* (x - μ)
end

function center_diag_gaussian_inv(η, μ, σ)
    (η .* σ) + μ
end

# Mean-field approximation used by ADVI
struct MeanField{T, TDists <: AbstractVector{<: Distribution}} <: VariationalPosterior where T <: Real
    μ::Vector{T}
    ω::Vector{T}
    dists::TDists
    ranges::Vector{UnitRange{Int}}
end

Base.length(advi::MeanField) = length(advi.μ)

_rand!(rng::AbstractRNG, q::MeanField{T, TDists}, x::AbstractVector{T}) where {T<:Real, TDists <: AbstractVector{<: Distribution}} = begin
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
