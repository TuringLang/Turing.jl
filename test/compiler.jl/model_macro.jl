using Turing, Distributions, Test
using MacroTools

# unit test model macro
expr = Turing.generate_observe(:x, :y)
@test expr.head == :escape
@test expr.args[1].head == :block
@test :(vi.logp += Turing.observe(sampler, y, x, vi)) in expr.args[1].args

expr = Turing.insertvarinfo(:())
@test expr == :((vi.logp = zero(Real), vi))

@model testmodel_comp(x, y) = begin
    s ~ InverseGamma(2,3)
    m ~ Normal(0,sqrt(s))

    x ~ Normal(m, sqrt(s))
    y ~ Normal(m, sqrt(s))

    return x, y
end

testmodel_comp(1.0, 1.2)
c = deepcopy(Turing._compiler_)

alias1 = Dict(
            :name => :testmodel_comp_model,
            :args => [:(vi::Turing.VarInfo)],
            :kwargs => [],
            :body => :(return testmodel_comp_model(vi, nothing))
           )
@test c[:alias1] == MacroTools.combinedef(alias1)

alias2 = Dict(
            :name => :testmodel_comp_model,
            :args => [:(sampler::Turing.Sampler)],
            :kwargs => [],
            :body => :(return testmodel_comp_model(Turing.VarInfo(), nothing))
           )
@test c[:alias2] == MacroTools.combinedef(alias2)

alias3 = Dict(
            :name => :testmodel_comp_model,
            :args => [],
            :kwargs => [],
            :body => :(return testmodel_comp_model(Turing.VarInfo(), nothing))
           )
@test c[:alias3] == MacroTools.combinedef(alias3)
@test length(c[:closure].args[2].args[2].args) == 6
@test mapreduce(line -> line.head == :macrocall, +, c[:closure].args[2].args[2].args) == 4

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

# test if we get a varinfo object back if no return value is set
@model testmodel2(x) = begin
    x ~ Normal()
end
f2_mm = testmodel2(2.)
@test isa(f2_mm(), Turing.VarInfo)
@test f2_mm().logp == logpdf(Normal(), 2.)
