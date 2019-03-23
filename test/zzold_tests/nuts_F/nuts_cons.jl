using Turing
using Turing: Sampler

alg = NUTS(1000, 200, 0.65)
println(alg)
sampler = Sampler(alg)

alg = NUTS(1000, 0.65)
println(alg)
sampler = Sampler(alg)

alg = NUTS(1000, 200, 0.65, :m)
println(alg)
sampler = Sampler(alg)
