using Turing
using Turing: Sampler

alg = SGLD(1000, 0.25)
println(alg)
sampler = Sampler(alg)

alg = SGLD(200, 0.25, :m)
println(alg)
sampler = Sampler(alg)

alg = SGLD(1000, 0.25, :s)
println(alg)
sampler = Sampler(alg)

println(typeof(alg))
println(isa(alg, SGLD))
println(isa(sampler, Sampler{Turing.Hamiltonian}))
