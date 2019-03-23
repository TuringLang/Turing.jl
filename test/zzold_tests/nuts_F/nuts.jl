using Test
using Turing

alg = NUTS(5000, 1000, 0.65)
res = sample(gdemo_default, alg)

check_gdemo(res[1000:end, :, :])
# v = get(res, [:s, :m])
# @info(mean(v.s[1000:end])," ≈ ", 49/24, "?")
# @info(mean(v.m[1000:end])," ≈ ", 7/6, "?")
# @test mean(v.s[1000:end]) ≈ 49/24 atol=0.2
# @test mean(v.m[1000:end]) ≈ 7/6 atol=0.2
