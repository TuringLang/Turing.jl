using Distributions
using Turing: Chain, Sample
using Base.Test

c = Chain()
#@test string(c) == "Empty Chain, weight 0.0"

d = Dict{Symbol, Any}()
d[:m] = [1,2,3]
sp = Sample(1, d)

c2 = Chain(1, Vector{Sample}([sp]))

string(c2)
samples = c2[:samples]
@test samples[1][:m] == d[:m]

#@test mean(c2, :m, x -> x) == [1.0, 2.0, 3.0]


#  Tests for Mamba Chain

@model mamba_chain_test() = begin
  m ~ Uniform(-1, 1)
  x ~ Wishart(7, [1 0.5; 0.5 1])
  y = Array{Array}(2,2)
  for i in eachindex(y)
    y[i] ~ Wishart(7, [1 0.5; 0.5 1])
  end
  return(m, x, y)
end

chain = sample(mamba_chain_test(), PG(5,300));
describe(chain)
