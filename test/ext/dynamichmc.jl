module DynamicHMCTests

using ..Models: gdemo_default
using ..NumericalTests: check_gdemo
using Test: @test, @testset
using Distributions: sample
using DynamicHMC: DynamicHMC
using DynamicPPL: DynamicPPL
using LinearAlgebra: I
using Random: Random
using StableRNGs: StableRNG
using Turing

@testset "TuringDynamicHMCExt" begin
    spl = externalsampler(DynamicHMC.NUTS())
    chn = sample(StableRNG(100), gdemo_default, spl, 10_000)
    check_gdemo(chn)
end

@model function counted_gaussian(evaluations)
    evaluations[] += 1
    return x ~ MvNormal(zeros(10), 0.95 * ones(10, 10) + 0.05I)
end

@testset "DynamicHMC respects tree depth during initialization" begin
    evaluations = Ref(0)
    model = counted_gaussian(evaluations)

    function count_evaluations(max_depth)
        evaluations[] = 0
        sampler = externalsampler(DynamicHMC.NUTS(; max_depth=max_depth))
        sample(StableRNG(1), model, sampler, 1; progress=false)
        return evaluations[]
    end

    @test count_evaluations(1) < count_evaluations(10)
end

end
