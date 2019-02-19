using Turing
using Test

@model hmcmatrixsup() = begin
  v ~ Wishart(7, [1 0.5; 0.5 1])
  v
end

model_f = hmcmatrixsup()
vs = []
chain = nothing
τ = 3000
for _ in 1:5
    chain = sample(model_f, HMC(τ, 0.1, 3))
    r = reshape(chain[:v], τ, 2, 2)
    push!(vs, reshape(mean(r, dims = [1]), 2, 2))
end

@test maximum(abs, mean(vs) - (7 * [1 0.5; 0.5 1])) <= 0.5
