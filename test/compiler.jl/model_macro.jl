using Turing, Distributions, Test
using MacroTools

# unit test model macro
expr = Turing.generate_observe(:x, :y)
@test expr.head == :block
@test :(vi.logp += Turing.observe(sampler, y, x, vi)) in expr.args

@model testmodel_comp(x, y) = begin
    s ~ InverseGamma(2,3)
    m ~ Normal(0,sqrt(s))

    x ~ Normal(m, sqrt(s))
    y ~ Normal(m, sqrt(s))

    return x, y
end
testmodel_comp(1.0, 1.2)

# check if drawing from the prior works
@model testmodel0(x) = begin
    x ~ Normal()
    return x
end
f0_mm = testmodel0()
@test mean(f0_mm() for _ in 1:1000) ≈ 0. atol=0.1

@model testmodel01(x) = begin
    x ~ Bernoulli(0.5)
    return x
end
f01_mm = testmodel01()
@test mean(f01_mm() for _ in 1:1000) ≈ 0.5 atol=0.1

# test if we get the correct return values
@model testmodel1(x1, x2) = begin
    s ~ InverseGamma(2,3)
    m ~ Normal(0,sqrt(s))

    x1 ~ Normal(m, sqrt(s))
    x2 ~ Normal(m, sqrt(s))

    return x1, x2
end
f1_mm = testmodel1(1., 10.)
@test f1_mm() == (1, 10)

# Test for assertions in observe statements.
@model brokentestmodel_observe1(x1, x2) = begin
    s ~ InverseGamma(2,3)
    m ~ Normal(0,sqrt(s))

    x1 ~ Normal(m, sqrt(s))
    x2 ~ x1 + 2

    return x1, x2
end

btest = brokentestmodel_observe1(1., 2.)
@test_throws ArgumentError btest()

@model brokentestmodel_observe2(x) = begin
    s ~ InverseGamma(2,3)
    m ~ Normal(0,sqrt(s))

    x = Vector{Float64}(undef, 2)
    x ~ [Normal(m, sqrt(s)), 2.0]

    return x
end

btest = brokentestmodel_observe2([1., 2.])
@test_throws ArgumentError btest()

# Test for assertions in assume statements.
@model brokentestmodel_assume1() = begin
    s ~ InverseGamma(2,3)
    m ~ Normal(0,sqrt(s))

    x1 ~ Normal(m, sqrt(s))
    x2 ~ x1 + 2

    return x1, x2
end

btest = brokentestmodel_assume1()
@test_throws ArgumentError btest()

@model brokentestmodel_assume2() = begin
    s ~ InverseGamma(2,3)
    m ~ Normal(0,sqrt(s))

    x = Vector{Float64}(undef, 2)
    x ~ [Normal(m, sqrt(s)), 2.0]

    return x
end

btest = brokentestmodel_assume2()
@test_throws ArgumentError btest()
