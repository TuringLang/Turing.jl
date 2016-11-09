using Turing: Chain, Sample

c = Chain()
println(c)
d = Dict{Symbol, Any}()
d[:m] = [1,2,3]
s = Sample(1, d)
c2 = Chain(1, Vector{Sample}([s]))
println(c2)
