const ϵ = 1e-8

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
    η = o.eta
    τ = o.tau

    g² = get!(
        o.acc,
        x,
        [fill(0.0, size(x)) for j = 1:o.n]
    )::Array{typeof(Tracker.data(x)), 1}
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


mutable struct DecayedADAGrad
    eta::Float64
    pre::Float64
    post::Float64

    acc::IdDict
end

DecayedADAGrad(η = 0.1, pre = 1.0, post = 0.9) = DecayedADAGrad(η, pre, post, IdDict())

function apply!(o::DecayedADAGrad, x, Δ)
  η = o.eta
  acc = get!(o.acc, x, fill(ϵ, size(x)))::typeof(Tracker.data(x))
  @. acc = o.post * acc + o.pre * Δ^2
  @. Δ *= η / (√acc + ϵ)
end
