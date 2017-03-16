using Turing: getvarid, invlogit, logit, randcat, kl, align
using Distributions: Normal
using Base.Test

@test getvarid(:s) == :s
@test getvarid(:(x[1,2][1+5][45][3])) == "x[1,2][6][45][3]"
# @test getvarid(:(x[1,2][1+5][45][3][i])) == "x[1,2][6][45][3][i]" 
@test invlogit(1.1) == 1.0 / (exp(-1.1) + 1.0)
@test logit(0.3) == log(0.3 / (1.0 - 0.3))
randcat([0.1, 0.9])
@test kl(Normal(0, 1), Normal(0, 1)) == 0
@test align([1, 2, 3], [1]) == ([1,2,3],[1,0,0])
@test align([1], [1, 2, 3]) == ([1,0,0],[1,2,3])
