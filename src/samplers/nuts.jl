"""
    NUTS(n_iters::Int, n_adapts::Int, delta::Float64)

No-U-Turn Sampler (NUTS) sampler.

Usage:

```julia
NUTS(1000, 200, 0.6j_max)
```

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0, sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

sample(gdemo([1.j_max, 2]), NUTS(1000, 200, 0.6j_max))
```
"""
mutable struct NUTS{T} <: AdaptiveHamiltonian
  n_iters   ::  Int       # number of samples
  n_adapts  ::  Int       # number of samples with adaption for epsilon
  delta     ::  Float64   # target accept rate
  space     ::  Set{T}    # sampling space, emtpy means all
  gid       ::  Int       # group ID
end
function NUTS(n_adapts::Int, delta::Float64, space...)
  NUTS(1, n_adapts, delta, isa(space, Symbol) ? Set([space]) : Set(space), 0)
end
function NUTS(n_iters::Int, n_adapts::Int, delta::Float64, space...)
  NUTS(n_iters, n_adapts, delta, isa(space, Symbol) ? Set([space]) : Set(space), 0)
end
function NUTS(n_iters::Int, delta::Float64)
  n_adapts_default = Int(round(n_iters / 2))
  NUTS(n_iters, n_adapts_default > 1000 ? 1000 : n_adapts_default, delta, Set(), 0)
end
NUTS(alg::NUTS, new_gid::Int) = NUTS(alg.n_iters, alg.n_adapts, alg.delta, alg.space, new_gid)

function hmc_step(θ, lj, lj_func, grad_func, ϵ, std, alg::NUTS; rev_func=nothing, log_func=nothing)
  θ_new, α = _nuts_step(θ, ϵ, lj, lj_func, grad_func, std)
  lj_new = lj_func(θ_new)
  is_accept = true
  return θ_new, lj_new, is_accept, α
end

"""
  function _build_tree(θ::T, r::AbstractVector, logu::AbstractFloat, v::Int, j::Int, ϵ::AbstractFloat,
                       H0::AbstractFloat,lj_func::Function, grad_func::Function, std::AbstractVector;
                       Δ_max::AbstractFloat=1000) where {T<:Union{Vector,SubArray}}

Recursively build balanced tree.

Ref: Algorithm 6 on http://www.stat.columbia.edu/~gelman/research/published/nuts.pdf

Arguments:

- `θ`         : model parameter
- `r`         : momentum variable
- `logu`      : slice variable (in log scale)
- `v`         : direction ∈ {-1, 1}
- `j`         : depth of tree
- `ϵ`         : leapfrog step size
- `H0`        : initial H
- `lj_func`   : function for log-joint
- `grad_func` : function for the gradient of log-joint
- `std`      : pre-conditioning matrix
- `Δ_max`     : threshold for exploeration error tolerance
"""
function _build_tree(θ::T, r::AbstractVector, logu::AbstractFloat, v::Int, j::Int, ϵ::AbstractFloat,
                     H0::AbstractFloat,lj_func::Function, grad_func::Function, std::AbstractVector;
                     Δ_max::AbstractFloat=1000.0) where {T<:Union{AbstractVector,SubArray}}
    if j == 0
      # Base case - take one leapfrog step in the direction v.
      θ′, r′, τ_valid = _leapfrog(θ, r, 1, v * ϵ, grad_func)
      # Use old H to save computation
      H′ = τ_valid == 0 ? Inf : _find_H(θ′, r′, lj_func, std)
      n′ = (logu <= -H′) ? 1 : 0
      s′ = (logu < Δ_max + -H′) ? 1 : 0
      α′ = exp(min(0, -H′ - (-H0)))

      return θ′, r′, θ′, r′, θ′, n′, s′, α′, 1
    else
      # Recursion - build the left and right subtrees.
      θm, rm, θp, rp, θ′, n′, s′, α′, n′α = _build_tree(θ, r, logu, v, j - 1, ϵ, H0, lj_func, grad_func, std)

      if s′ == 1
        if v == -1
          θm, rm, _, _, θ′′, n′′, s′′, α′′, n′′α = _build_tree(θm, rm, logu, v, j - 1, ϵ, H0, lj_func, grad_func, std)
        else
          _, _, θp, rp, θ′′, n′′, s′′, α′′, n′′α = _build_tree(θp, rp, logu, v, j - 1, ϵ, H0, lj_func, grad_func, std)
        end
        if rand() < n′′ / (n′ + n′′)
          θ′ = θ′′
        end
        α′ = α′ + α′′
        n′α = n′α + n′′α
        s′ = s′′ * (dot(θp - θm, rm) >= 0 ? 1 : 0) * (dot(θp - θm, rp) >= 0 ? 1 : 0)
        n′ = n′ + n′′
      end

      θm, rm, θp, rp, θ′, n′, s′, α′, n′α
    end
  end


"""
  function _nuts_step(θ::T, r0, ϵ::AbstractFloat, lj_func::Function, grad_func::Function, std::AbstractVector;
                      j_max::Int=j_max) where {T<:Union{AbstractVector,SubArray}}

Perform one NUTS step.

Ref: Algorithm 6 on http://www.stat.columbia.edu/~gelman/research/published/nuts.pdf

Arguments:

- `θ`         : model parameter
- `ϵ`         : leapfrog step size
- `lj`        : initial log-joint prob
- `lj_func`   : function for log-joint
- `grad_func` : function for the gradient of log-joint
- `std`      : pre-conditioning matrix
- `j_max`     : maximum expanding of doubling tree
"""
function _nuts_step(θ::T, ϵ::AbstractFloat, lj::Real, lj_func::Function, grad_func::Function, std::AbstractVector;
                    j_max::Int=5) where {T<:Union{AbstractVector,SubArray}}

  @debug "sampling momentums..."
  θ_dim = length(θ)
  r0 = _sample_momentum(θ_dim, std)

  H0 = _find_H(θ, r0, lj, std)
  logu = log(rand()) + -H0

  θm = θ; θp = θ; rm = r0; rp = r0; j = 0; θ_new = θ; n = 1; s = 1
  local da_stat

  while s == 1 && j <= j_max

    v = rand([-1, 1])
    if v == -1
        θm, rm, _, _, θ′, n′, s′, α, nα = _build_tree(θm, rm, logu, v, j, ϵ, H0, lj_func, grad_func, std)
    else
        _, _, θp, rp, θ′, n′, s′, α, nα = _build_tree(θp, rp, logu, v, j, ϵ, H0, lj_func, grad_func, std)
    end

    if s′ == 1
        if rand() < min(1, n′ / n)
            θ_new = θ′
        end
    end

    n = n + n′
    s = s′ * (dot(θp - θm, rm) >= 0 ? 1 : 0) * (dot(θp - θm, rp) >= 0 ? 1 : 0)
    j = j + 1

    da_stat = α / nα

  end

  return θ_new, da_stat

end
