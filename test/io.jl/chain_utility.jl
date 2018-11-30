using Turing
using Turing: Chain, Sample
using Test
using MCMCChain: describe

c = Chain()
#@test string(c) == "Empty Chain, weight 0.0"

d = Dict{Symbol, Any}()
d[:m] = [1,2,3]
sp = Sample(1, d)

c2 = Chain(1.0, Vector{Sample}([sp]))

string(c2)
samples = c2[:samples]
@test samples[1][:m] == d[:m]

#@test mean(c2, :m, x -> x) == [1.0, 2.0, 3.0]


# Tests for MCMC Chain

@model mamba_chain_test() = begin
  m ~ Uniform(-1, 1)
  x ~ Wishart(7, [1 0.5; 0.5 1])
  y = Array{Array}(undef, 2,2)
  for i in eachindex(y)
    y[i] ~ Wishart(7, [1 0.5; 0.5 1])
  end
  return(m, x, y)
end

chain = sample(mamba_chain_test(), PG(1,200));
describe(chain)

d1 = Dict{Symbol, Any}()
d1[Symbol("x[1]")] = 1
sp1 = Sample(1, d1)

@test sp1[:x] == [1]

d1 = Dict{Symbol, Any}()
d1[Symbol("x[1][1]")] = 1
sp1 = Sample(1, d1)

@test sp1[:x] == [[1]]



d2 = Dict{Symbol, Any}()
d2[Symbol("x[1]")] = 1
d2[Symbol("x[2]")] = 2
sp2 = Sample(1, d2)

@test sp2[:x] == [1, 2]

d3 = Dict{Symbol, Any}()
d3[Symbol("x[1,1]")] = 1.1
d3[Symbol("x[1,2]")] = 2.2
d3[Symbol("x[2,1]")] = 3.3
d3[Symbol("x[2,2]")] = 4.4
sp3 = Sample(1, d3)

@test sp3[:x] == [1.1 2.2; 3.3 4.4]

d4 = Dict{Symbol, Any}()
d4[Symbol("x[1][1]")] = 1.1
d4[Symbol("x[1][2]")] = 2.2
d4[Symbol("x[2][1]")] = 3.3
d4[Symbol("x[2][2]")] = 4.4
sp4 = Sample(1, d4)

@test sp4[:x] == [Any[1.1,2.2], Any[3.3,4.4]]

d5 = Dict{Symbol, Any}()
d5[Symbol("x[1][1,1]")] = 1
d5[Symbol("x[1][2,1]")] = 2
d5[Symbol("x[2][1,1]")] = 1
d5[Symbol("x[2][2,1]")] = 2
d5[Symbol("x[1][1,2]")] = 3
d5[Symbol("x[1][2,2]")] = 4
d5[Symbol("x[2][1,2]")] = 3
d5[Symbol("x[2][2,2]")] = 4
sp5 = Sample(1, d5)

@test sp5[:x] == [[1 3; 2 4], [1 3; 2 4]]

d5 = Dict{Symbol, Any}()
d5[Symbol("x[1,1,1]")] = 1
d5[Symbol("x[1,1,2]")] = 2
d5[Symbol("x[1,2,1]")] = 3
d5[Symbol("x[1,2,2]")] = 4
d5[Symbol("x[2,1,1]")] = 5
d5[Symbol("x[2,1,2]")] = 6
d5[Symbol("x[2,2,1]")] = 7
d5[Symbol("x[2,2,2]")] = 8
sp5 = Sample(1, d5)
x5 = Array{Any, 3}(undef, 2,2,2)
x5[1,1,1] = 1
x5[1,1,2] = 2
x5[1,2,1] = 3
x5[1,2,2] = 4
x5[2,1,1] = 5
x5[2,1,2] = 6
x5[2,2,1] = 7
x5[2,2,2] = 8
@test sp5[:x] == x5


d6 = Dict{Symbol, Any}()
d6[Symbol("x[1][1]")] = 1.1
d6[Symbol("x[1][2]")] = 2.2
d6[Symbol("x[2][1][1,1]")] = 3.3
d6[Symbol("x[2][1][1,2]")] = 3.3
d6[Symbol("x[2][1][2,1]")] = 3.3
d6[Symbol("x[2][1][2,2]")] = 3.3
d6[Symbol("x[2][2][1]")] = 4.4
d6[Symbol("x[2][2][2]")] = 4.4
sp6 = Sample(1, d6)

@test sp6[Symbol("x")] == Any[Any[1.1, 2.2], Any[Any[3.3 3.3;3.3 3.3], Any[4.4,4.4]]]
