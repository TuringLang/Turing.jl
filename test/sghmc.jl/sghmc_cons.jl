using Turing
using Turing: Sampler

alg = SGHMC(1000, 0.01, 0.1)
println(alg)
sampler = Sampler(alg)

alg = SGHMC(200, 0.01, 0.1, :m)
println(alg)
sampler = Sampler(alg)

alg = SGHMC(1000, 0.01, 0.1, :s)
println(alg)
sampler = Sampler(alg)

println(typeof(alg))
println(isa(alg, SGHMC))
println(isa(sampler, Sampler{Turing.Hamiltonian}))
