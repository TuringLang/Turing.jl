using Turing

alg = eNUTS(1000, 0.65)
println(alg)
sampler = Sampler(alg)

alg = eNUTS(200, 0.65, :m)
println(alg)
sampler = Sampler(alg)
