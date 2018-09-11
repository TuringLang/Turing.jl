using Turing
using Test

@model hmcmatrixsup() = begin
  v ~ Wishart(7, [1 0.5; 0.5 1])
  v
end

model_f = hmcmatrixsup()
vs = []
chain = nothing
for _ in 1:5
  chain = sample(model_f, HMC(3000, 0.1, 3))
  push!(vs, mean(chain[:v]))
end

@test mean(vs) ≈ (7 * [1 0.5; 0.5 1]) atol=0.5
