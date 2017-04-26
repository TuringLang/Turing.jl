using Turing

alg = HMCDA(1000, 0.65, 0.75)
println(alg)
sampler = Sampler(alg)

alg = HMCDA(200, 0.65, 0.75, :m)
println(alg)
sampler = Sampler(alg)

alg = HMCDA(1000, 200, 0.65, 0.75)
println(alg)
sampler = Sampler(alg)

alg = HMCDA(1000, 200, 0.65, 0.75, :s)
println(alg)
sampler = Sampler(alg)

println(typeof(alg))
println(isa(alg, HMCDA))
println(isa(sampler, Sampler{Turing.Hamiltonian}))
