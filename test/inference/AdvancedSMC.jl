using Turing, Random, Test

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@testset "smc.jl" begin
    # No tests.
end

@testset "pmmh.jl" begin
    @turing_testset "pmmh constructor" begin
        N = 2000
        s1 = PMMH(N, SMC(10, :s), MH(1,(:m, s -> Normal(s, sqrt(1)))))
        s2 = PMMH(N, SMC(10, :s), MH(1, :m))
        s3 = PIMH(N, SMC(10))

        c1 = sample(gdemo_default, s1)
        c2 = sample(gdemo_default, s2)
        c3 = sample(gdemo_default, s3)
    end
    @numerical_testset "pmmh inference" begin
        alg = PMMH(2000, SMC(20, :m), MH(1, (:s, GKernel(1))))
        chain = sample(gdemo_default, alg)
        check_gdemo(chain, eps=0.1)

        # PMMH with prior as proposal
        alg = PMMH(2000, SMC(20, :m), MH(1, :s))
        chain = sample(gdemo_default, alg)
        check_gdemo(chain, eps=0.1)

        # PIMH
        alg = PIMH(2000, SMC(20))
        chain = sample(gdemo_default, alg)
        check_gdemo(chain)

        # MoGtest
        pmmh = PMMH(2000,
            SMC(10, :z1, :z2, :z3, :z4),
            MH(1, :mu1, :mu2))
        chain = sample(MoGtest_default, pmmh)

        check_MoGtest_default(chain, eps = 0.1)
    end
end

@testset "ipmcmc.jl" begin
    @turing_testset "ipmcmc constructor" begin
        Random.seed!(125)

        N = 50
        s1 = IPMCMC(10, N, 4, 2)
        s2 = IPMCMC(10, N, 4)

        c1 = sample(gdemo_default, s1)
        c2 = sample(gdemo_default, s2)
    end
    @numerical_testset "ipmcmc inference" begin
        alg = IPMCMC(30, 500, 4)
        chain = sample(gdemo_default, alg)
        check_gdemo(chain)

        alg2 = IPMCMC(15, 100, 10)
        chain2 = sample(MoGtest_default, alg2)
        check_MoGtest_default(chain2, eps=0.2)
    end
end

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
