using Turing

alg = HMCDA(1000, 0.65, 0.75)
println(alg)
sampler = HMCSampler{HMCDA}(alg)

println(typeof(alg))
println(isa(alg, HMCDA))
