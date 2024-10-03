using Turing
using Test

@model function test_ex_rt()
    x ~ Normal(10, 1)
    y ~ Normal(x / 2, 1)
    z = y + 1
    x = x - 1
    return x, y, z
end

mf = test_ex_rt()

for alg in
    [HMC(0.2, 3), PG(20, 2000), SMC(), IS(10000), Gibbs(; x=PG(20, 1), y=HMC(0.2, 3))]
    chn = sample(mf, alg)
    @test mean(chn[:x]) ≈ 10.0 atol = 0.2
    @test mean(chn[:y]) ≈ 5.0 atol = 0.2
end
