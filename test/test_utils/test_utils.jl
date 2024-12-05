"""Module for testing the test utils themselves."""
module TestUtilsTests

using ..ADUtils: ADTypeCheckContext, AbstractWrongADBackendError, adbackends
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Test: @test, @testset, @test_throws
using Turing: Turing
using Turing: DynamicPPL
using Zygote: Zygote

# Check that the ADTypeCheckContext works as expected.
@testset "ADTypeCheckContext" begin
    Turing.@model test_model() = x ~ Turing.Normal(0, 1)
    tm = test_model()
    # These tests don't make sense for Enzyme, since it doesn't have its own element type.
    # TODO(mhauru): Make these tests work for more Mooncake.
    adtypes = filter(
        adtype -> !(adtype isa Turing.AutoMooncake || adtype isa Turing.AutoEnzyme),
        adbackends,
    )
    for actual_adtype in adtypes
        sampler = Turing.HMC(0.1, 5; adtype=actual_adtype)
        for expected_adtype in adtypes
            if (
                actual_adtype == Turing.AutoForwardDiff() &&
                expected_adtype == Turing.AutoZygote()
            )
                # TODO(mhauru) We are currently unable to check this case.
                continue
            end
            contextualised_tm = DynamicPPL.contextualize(
                tm, ADTypeCheckContext(expected_adtype, tm.context)
            )
            @testset "Expected: $expected_adtype, Actual: $actual_adtype" begin
                if actual_adtype == expected_adtype
                    # Check that this does not throw an error.
                    Turing.sample(contextualised_tm, sampler, 2)
                else
                    @test_throws AbstractWrongADBackendError Turing.sample(
                        contextualised_tm, sampler, 2
                    )
                end
            end
        end
    end
end

end
