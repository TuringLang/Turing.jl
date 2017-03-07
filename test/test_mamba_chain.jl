using Turing, Distributions
using Base.Test

@model mamba_chain_test() = begin
  m = rand()
  x = rand(2,2)
  y = Array{Array}(2,2)
  for i in eachindex(y)
    y[i] = rand(2,2)
  end
  return(m, x, y)
end

chain = @sample(mamba_chain_test(), PG(5,300));
sim1 = MambaChains(chain)
describe(sim1)
