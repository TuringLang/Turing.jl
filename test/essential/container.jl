module ContainerTests

using Distributions: Bernoulli, Beta, Gamma, Normal
using DynamicPPL
using Test: @test, @testset
using Turing
using Turing.Inference: Particle, get_varinfo
using StableRNGs

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

    @testset "traced model" begin
        trace = TracedModel(StableRNG(23), test());
        loglikes = collect(trace)

        # ensure correctness of incremental likelihoods
        @test all(loglikes .≈ -log(2))

        # ensure consistency of ProduceLogLikelihoodAccumulator
        @test sum(loglikes) == DynamicPPL.getloglikelihood(get_varinfo(trace))

        accs = DynamicPPL.OnlyAccsVarInfo()
        accs = DynamicPPL.setacc!!(accs, DynamicPPL.RawValueAccumulator(true))
        _, accs = DynamicPPL.init!!(
            TracedRNG(StableRNG(23)),
            test(),
            accs,
            DynamicPPL.InitFromPrior(),
            DynamicPPL.UnlinkAll(),
        )

        # ensure that traced models evaluate the same as basic ones
        traced_acc = get_varinfo(trace)
        @test DynamicPPL.getloglikelihood(accs) == DynamicPPL.getloglikelihood(traced_acc)
        @test DynamicPPL.get_raw_values(traced_acc) == DynamicPPL.get_raw_values(accs)
    end

    @testset "replay" begin
        # this mimics propagate without resampling
        function advance_trace!(particle::Particle, isref::Bool)
            isdone = 0.0
            while !isnothing(isdone)
                !isref && Turing.Inference.update_key!(particle)
                isdone = Turing.Inference.advance!(particle, isref)
            end
            return get_varinfo(particle.value)
        end

        @model function normal()
            a ~ Normal(0, 1)
            3 ~ Normal(a, 2)
            b ~ Normal(a, 1)
            1.5 ~ Normal(b, 2)
            return a, b
        end

        # advance the trace one step
        trace = TracedModel(StableRNG(23), normal());
        particle = Particle(trace);
        vi = advance_trace!(particle, false)

        # set trace as reference and replay
        new_trace = Turing.Inference.set_reference(particle.value)
        ref_particle = Particle(new_trace)
        ref_vi = advance_trace!(ref_particle, true)

        @test ref_vi == vi
    end
end

end
