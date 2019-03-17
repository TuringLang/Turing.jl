using Turing
using Turing: Sampler
using Test

alg = SGLD(1000, 0.25)
sampler = Sampler(alg, Turing.VarInfo())
@test isa(alg, SGLD)
@test isa(sampler, Sampler{<:Turing.SGLD})

alg = SGLD(200, 0.25, :m)
sampler = Sampler(alg, Turing.VarInfo())
@test isa(alg, SGLD)
@test isa(sampler, Sampler{<:Turing.SGLD})

alg = SGLD(1000, 0.25, :s)
sampler = Sampler(alg, Turing.VarInfo())
@test isa(alg, SGLD)
@test isa(sampler, Sampler{<:Turing.SGLD})
