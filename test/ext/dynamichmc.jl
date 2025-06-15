module DynamicHMCTests

using ..Models: gdemo_default
using ..NumericalTests: check_gdemo
using Test: @test, @testset
using Distributions: sample
using DynamicHMC: DynamicHMC
using DynamicPPL: DynamicPPL
using DynamicPPL: Sampler
using Random: Random
using StableRNGs: StableRNG
using Turing

@testset "TuringDynamicHMCExt" begin
    spl = externalsampler(DynamicHMC.NUTS())

    @testset "alg_str" begin
        @test DynamicPPL.alg_str(Sampler(spl)) == "DynamicNUTS"
    end

    @testset "sample() interface" begin
        @model function demo_normal(x)
            a ~ Normal()
            return x ~ Normal(a)
        end
        model = demo_normal(2.0)
        # note: passing LDF to a Hamiltonian sampler requires explicit adtype
        ldf = LogDensityFunction(model; adtype=AutoForwardDiff())
        sampling_objects = Dict("DynamicPPL.Model" => model, "LogDensityFunction" => ldf)
        seed = 468
        @testset "sampling with $name" for (name, model_or_ldf) in sampling_objects
            # check sampling works without rng
            @test sample(model_or_ldf, spl, 5) isa Chains
            # check reproducibility with rng
            chn1 = sample(Random.Xoshiro(seed), model_or_ldf, spl, 5)
            chn2 = sample(Random.Xoshiro(seed), model_or_ldf, spl, 5)
            @test mean(chn1[:a]) == mean(chn2[:a])
        end
    end

    @testset "numerical accuracy" begin
        rng = StableRNG(468)
        chn = sample(rng, gdemo_default, spl, 10_000)
        check_gdemo(chn)
    end
end

end
