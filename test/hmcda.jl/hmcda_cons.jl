using Turing
using Turing: Sampler

alg = HMCDA(1000, 0.65, 0.75)
println(alg)
s = Sampler(alg, nothing)

alg = HMCDA(200, 0.65, 0.75, :m)
println(alg)
s = Sampler(alg, nothing)

alg = HMCDA(1000, 200, 0.65, 0.75)
println(alg)
s = Sampler(alg, nothing)

alg = HMCDA(1000, 200, 0.65, 0.75, :s)
println(alg)
s = Sampler(alg, nothing)

println(typeof(alg))
println(isa(alg, HMCDA))
println(isa(s, Sampler{<:Turing.Hamiltonian}))
