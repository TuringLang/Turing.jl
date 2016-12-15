using Turing: link, invlink
using Base.Test
using Distributions

dists = [Beta(2,2), Dirichlet(2, 3), Wishart(7, [1 0.5; 0.5 1])]

for dist in dists
  x = rand(dist)    # sample
  y = link(dist, x) # X -> R
  # Test if R -> X is equal to the original value
  @test_approx_eq_eps invlink(dist, y) x 1e-9
end
