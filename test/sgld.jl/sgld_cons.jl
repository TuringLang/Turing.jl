using Turing
using Turing: Sampler
using Base.Test

alg = SGLD(1000, 0.25)
sampler = Sampler(alg)
@test isa(alg, SGLD)
@test isa(sampler, Sampler{Turing.SGLD})

alg = SGLD(200, 0.25, :m)
sampler = Sampler(alg)
@test isa(alg, SGLD)
@test isa(sampler, Sampler{Turing.SGLD})

alg = SGLD(1000, 0.25, :s)
sampler = Sampler(alg)
@test isa(alg, SGLD)
@test isa(sampler, Sampler{Turing.SGLD})
