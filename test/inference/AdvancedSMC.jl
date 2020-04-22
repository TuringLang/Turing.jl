using Turing, Random, Test
using Turing.Core: ResampleWithESSThreshold
using Turing.Inference: getspace, resample_systematic, resample_multinomial

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@testset "SMC" begin
    @turing_testset "SMC constructor" begin
        s = SMC()
        @test s.resampler == ResampleWithESSThreshold()
        @test getspace(s) === ()

        s = SMC(:x)
        @test s.resampler == ResampleWithESSThreshold()
        @test getspace(s) === (:x,)

        s = SMC((:x,))
        @test s.resampler == ResampleWithESSThreshold()
        @test getspace(s) === (:x,)

        s = SMC(:x, :y)
        @test s.resampler == ResampleWithESSThreshold()
        @test getspace(s) === (:x, :y)

        s = SMC((:x, :y))
        @test s.resampler == ResampleWithESSThreshold()
        @test getspace(s) === (:x, :y)

        s = SMC(0.6)
        @test s.resampler === ResampleWithESSThreshold(resample_systematic, 0.6)
        @test getspace(s) === ()

        s = SMC(0.6, (:x,))
        @test s.resampler === ResampleWithESSThreshold(resample_systematic, 0.6)
        @test getspace(s) === (:x,)

        s = SMC(resample_multinomial, 0.6)
        @test s.resampler === ResampleWithESSThreshold(resample_multinomial, 0.6)
        @test getspace(s) === ()

        s = SMC(resample_multinomial, 0.6, (:x,))
        @test s.resampler === ResampleWithESSThreshold(resample_multinomial, 0.6)
        @test getspace(s) === (:x,)

        s = SMC(resample_systematic)
        @test s.resampler === resample_systematic
        @test getspace(s) === ()

        s = SMC(resample_systematic, (:x,))
        @test s.resampler === resample_systematic
        @test getspace(s) === (:x,)
    end

    @turing_testset "models" begin
        @model normal() = begin
            a ~ Normal(4,5)
            3 ~ Normal(a,2)
            b ~ Normal(a,1)
            1.5 ~ Normal(b,2)
            a, b
        end

        tested = sample(normal(), SMC(), 100);

        # failing test
        @model fail_smc() = begin
            a ~ Normal(4,5)
            3 ~ Normal(a,2)
            b ~ Normal(a,1)
            if a >= 4.0
                1.5 ~ Normal(b,2)
            end
            a, b
        end

        @test_throws ErrorException sample(fail_smc(), SMC(), 100)
    end
end

# @testset "pmmh.jl" begin
#     @turing_testset "pmmh constructor" begin
#         N = 2000
#         s1 = PMMH(N, SMC(10, :s), MH(1,(:m, s -> Normal(s, sqrt(1)))))
#         s2 = PMMH(N, SMC(10, :s), MH(1, :m))
#         s3 = PIMH(N, SMC())
#
#         c1 = sample(gdemo_default, s1)
#         c2 = sample(gdemo_default, s2)
#         c3 = sample(gdemo_default, s3)
#     end
#     @numerical_testset "pmmh inference" begin
#         alg = PMMH(2000, SMC(20, :m), MH(1, (:s, GKernel(1))))
#         chain = sample(gdemo_default, alg)
#         check_gdemo(chain, atol = 0.1)
#
#         # PMMH with prior as proposal
#         alg = PMMH(2000, SMC(20, :m), MH(1, :s))
#         chain = sample(gdemo_default, alg)
#         check_gdemo(chain, atol = 0.1)
#
#         # PIMH
#         alg = PIMH(2000, SMC())
#         chain = sample(gdemo_default, alg)
#         check_gdemo(chain)
#
#         # MoGtest
#         pmmh = PMMH(2000,
#             SMC(10, :z1, :z2, :z3, :z4),
#             MH(1, :mu1, :mu2))
#         chain = sample(MoGtest_default, pmmh)
#
#         check_MoGtest_default(chain, atol = 0.1)
#     end
# end

# @testset "ipmcmc.jl" begin
#     @turing_testset "ipmcmc constructor" begin
#         Random.seed!(125)
#
#         N = 50
#         s1 = IPMCMC(10, N, 4, 2)
#         s2 = IPMCMC(10, N, 4)
#
#         c1 = sample(gdemo_default, s1)
#         c2 = sample(gdemo_default, s2)
#     end
#     @numerical_testset "ipmcmc inference" begin
#         alg = IPMCMC(30, 500, 4)
#         chain = sample(gdemo_default, alg)
#         check_gdemo(chain)
#
#         alg2 = IPMCMC(15, 100, 10)
#         chain2 = sample(MoGtest_default, alg2)
#         check_MoGtest_default(chain2, atol = 0.2)
#     end
# end

@turing_testset "resample.jl" begin
    D = [0.3, 0.4, 0.3]
    num_samples = Int(1e6)
    resSystematic = Turing.Inference.resample_systematic(D, num_samples )
    resStratified = Turing.Inference.resample_stratified(D, num_samples )
    resMultinomial= Turing.Inference.resample_multinomial(D, num_samples )
    resResidual   = Turing.Inference.resample_residual(D, num_samples )
    Turing.Inference.resample(D)
    resSystematic2=Turing.Inference.resample(D, num_samples )

    @test sum(resSystematic .== 2) ≈ (num_samples * 0.4) atol=1e-3*num_samples
    @test sum(resSystematic2 .== 2) ≈ (num_samples * 0.4) atol=1e-3*num_samples
    @test sum(resStratified .== 2) ≈ (num_samples * 0.4) atol=1e-3*num_samples
    @test sum(resMultinomial .== 2) ≈ (num_samples * 0.4) atol=1e-2*num_samples
    @test sum(resResidual .== 2) ≈ (num_samples * 0.4) atol=1e-2*num_samples
end
