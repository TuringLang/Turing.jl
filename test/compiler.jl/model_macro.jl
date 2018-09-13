using Turing, Distributions, Test

@model testmodel0(x) = begin
    x ~ Normal()
    return x
end

f0 = testmodel0()

#@test mean(f0() for _ in 1:1000) ≈ 0.


@model testmodel1(x1, x2) = begin
    s ~ InverseGamma(2,3)
    m ~ Normal(0,sqrt(s))

    x1 ~ Normal(m, sqrt(s))
    x2 ~ Normal(m, sqrt(s))

    return x1, x2
end

f1 = testmodel1(1., 10.)
@test f1() == (1, 10)

@model testmodel2(x) = begin
    x ~ Normal()
end

f2 = testmodel2(2.)
@test f2().logp == logpdf(Normal(), 2.)

@model testmodel3() = begin

    s ~ InverseGamma(2,3)
    m ~ Normal(0,sqrt(s))

    b ~ Bernoulli(0.5)

    if b == 0
        μ1 = m
        x1 ~ Normal(μ1, 1.)
        return (x1, μ1)
    else
        μ2 = m * -100
        x2 ~ Normal(μ2, 1.)
        return (x2, μ2)
    end
end

testmodel3()
