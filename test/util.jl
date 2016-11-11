using Turing: getvarid, invlogit, logit, randcat, kl, align
using Distributions: Normal
using Base.Test

@test getvarid(:s) == ":(s)"
@test getvarid(parse("x = 1")) == ":(x), 1"
@test invlogit(1.1) == 1.0 / (exp(-1.1) + 1.0)
@test logit(0.3) == log(0.3 / (1.0 - 0.3))
randcat([0.1, 0.9])
@test kl(Normal(0, 1), Normal(0, 1)) == 0
@test align([1, 2, 3], [1]) == ([1,2,3],[1,0,0])
@test align([1], [1, 2, 3]) == ([1,0,0],[1,2,3])
