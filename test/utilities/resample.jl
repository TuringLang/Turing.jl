using Turing, Random, Test

include("../test_utils/AllUtils.jl")

@turing_testset "resample.jl" begin
    D = [0.3, 0.4, 0.3]
    num_samples = Int(1e6)
    resSystematic = Turing.resample_systematic(D, num_samples )
    resStratified = Turing.resample_stratified(D, num_samples )
    resMultinomial= Turing.resample_multinomial(D, num_samples )
    resResidual   = Turing.resample_residual(D, num_samples )
    Turing.resample(D)
    resSystematic2=Turing.resample(D, num_samples )

    @test sum(resSystematic .== 2) ≈ (num_samples * 0.4) atol=1e-3*num_samples
    @test sum(resSystematic2 .== 2) ≈ (num_samples * 0.4) atol=1e-3*num_samples
    @test sum(resStratified .== 2) ≈ (num_samples * 0.4) atol=1e-3*num_samples
    @test sum(resMultinomial .== 2) ≈ (num_samples * 0.4) atol=1e-2*num_samples
    @test sum(resResidual .== 2) ≈ (num_samples * 0.4) atol=1e-2*num_samples
end
