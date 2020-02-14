const ϵ = 1e-8

"""
    TruncatedADAGrad(η=0.1, τ=1.0, n=100)

Implements a truncated version of AdaGrad in the sense that only the `n` previous gradient norms are used to compute the scaling rather than *all* previous. It has parameter specific learning rates based on how frequently it is updated.

## Parameters
  - η: learning rate
  - τ: constant scale factor
  - n: number of previous gradient norms to use in the scaling.

## References
[ADAGrad](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) optimiser.
Parameters don't need tuning.

[TruncatedADAGrad](https://arxiv.org/abs/1506.03431v2) (Appendix E).
"""
mutable struct TruncatedADAGrad
    eta::Float64
    tau::Float64
    n::Int
    
    iters::IdDict
    acc::IdDict
end

function TruncatedADAGrad(η = 0.1, τ = 1.0, n = 100)
    TruncatedADAGrad(η, τ, n, IdDict(), IdDict())
end

function apply!(o::TruncatedADAGrad, x, Δ)
    T = eltype(Tracker.data(Δ))
    
    η = o.eta
    τ = o.tau

    g² = get!(
        o.acc,
        x,
        [zeros(T, size(x)) for j = 1:o.n]
    )::Array{typeof(Tracker.data(Δ)), 1}
    i = get!(o.iters, x, 1)::Int

    # Example: suppose i = 12 and o.n = 10
    idx = mod(i - 1, o.n) + 1 # => idx = 2

    # set the current
    @inbounds @. g²[idx] = Δ^2 # => g²[2] = Δ^2 where Δ is the (o.n + 2)-th Δ

    # TODO: make more efficient and stable
    s = sum(g²)
    
    # increment
    o.iters[x] += 1
    
    # TODO: increment (but "truncate")
    # o.iters[x] = i > o.n ? o.n + mod(i, o.n) : i + 1

    @. Δ *= η / (τ + sqrt(s) + ϵ)
end

"""
    DecayedADAGrad(η=0.1, pre=1.0, post=0.9)

Implements a decayed version of AdaGrad. It has parameter specific learning rates based on how frequently it is updated.

## Parameters
  - η: learning rate
  - pre: weight of new gradient norm
  - post: weight of histroy of gradient norms

## References
[ADAGrad](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) optimiser.
Parameters don't need tuning.
"""
mutable struct DecayedADAGrad
    eta::Float64
    pre::Float64
    post::Float64

    acc::IdDict
end

DecayedADAGrad(η = 0.1, pre = 1.0, post = 0.9) = DecayedADAGrad(η, pre, post, IdDict())

function apply!(o::DecayedADAGrad, x, Δ)
    T = eltype(Tracker.data(Δ))
    
    η = o.eta
    acc = get!(o.acc, x, fill(T(ϵ), size(x)))::typeof(Tracker.data(x))
    @. acc = o.post * acc + o.pre * Δ^2
    @. Δ *= η / (√acc + ϵ)
end
