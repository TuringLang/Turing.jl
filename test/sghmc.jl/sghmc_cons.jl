using Turing
using Turing: Sampler
using Test

alg = SGHMC(1000, 0.01, 0.1)
sampler = Sampler(alg)
@test isa(alg, SGHMC)
@test isa(sampler, Sampler{<:Turing.SGHMC})

alg = SGHMC(200, 0.01, 0.1, :m)
sampler = Sampler(alg)
@test isa(alg, SGHMC)
@test isa(sampler, Sampler{<:Turing.SGHMC})

alg = SGHMC(1000, 0.01, 0.1, :s)
sampler = Sampler(alg)
@test isa(alg, SGHMC)
@test isa(sampler, Sampler{<:Turing.SGHMC})
