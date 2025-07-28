module ContainerTests

using AdvancedPS: AdvancedPS
using Distributions: Bernoulli, Beta, Gamma, Normal
using DynamicPPL: DynamicPPL, @model, Sampler
using Test: @test, @testset
using Turing

@testset "container.jl" begin
    @model function test()
        a ~ Normal(0, 1)
        x ~ Bernoulli(1)
        b ~ Gamma(2, 3)
        1 ~ Bernoulli(x / 2)
        c ~ Beta()
        0 ~ Bernoulli(x / 2)
        return x
    end

    @testset "constructor" begin
        vi = DynamicPPL.VarInfo()
        vi = DynamicPPL.setacc!!(vi, Turing.Inference.ProduceLogLikelihoodAccumulator())
        spl = Sampler(PG(10))
        model = test()
        trace = AdvancedPS.Trace(model, spl, vi, AdvancedPS.TracedRNG())

        # Make sure the backreference from taped_globals to the trace is in place.
        @test trace.model.ctask.taped_globals.other === trace

        res = AdvancedPS.advance!(trace, false)
        @test DynamicPPL.get_num_produce(trace.model.f.varinfo) == 1
        @test res ≈ -log(2)

        # Catch broken copy, espetially for RNG / VarInfo
        newtrace = AdvancedPS.fork(trace)
        res2 = AdvancedPS.advance!(trace)
        @test DynamicPPL.get_num_produce(trace.model.f.varinfo) == 2
        @test DynamicPPL.get_num_produce(newtrace.model.f.varinfo) == 1
    end

    @testset "fork" begin
        @model function normal()
            a ~ Normal(0, 1)
            3 ~ Normal(a, 2)
            b ~ Normal(a, 1)
            1.5 ~ Normal(b, 2)
            return a, b
        end
        vi = DynamicPPL.VarInfo()
        vi = DynamicPPL.setacc!!(vi, Turing.Inference.ProduceLogLikelihoodAccumulator())
        sampler = Sampler(PG(10))
        model = normal()

        trace = AdvancedPS.Trace(model, sampler, vi, AdvancedPS.TracedRNG())

        newtrace = AdvancedPS.forkr(trace)
        # Catch broken replay mechanism
        @test AdvancedPS.advance!(trace) ≈ AdvancedPS.advance!(newtrace)
    end
end

end
