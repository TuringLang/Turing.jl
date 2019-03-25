using Turing

@testset "ipmcmc.jl" begin
    @testset "ipmcmc constructor" begin
        Random.seed!(125)

        N = 50
        s1 = IPMCMC(10, N, 4, 2)
        s2 = IPMCMC(10, N, 4)

        c1 = sample(gdemo_default, s1)
        c2 = sample(gdemo_default, s2)

        # Very loose bound, only for testing constructor.
        for c in [c1, c2]
            check_gdemo(c, eps = 1.0)
        end
    end
    @testset "ipmcmc inference" begin
        alg = IPMCMC(30, 500, 4)
        chain = sample(gdemo_default, alg)
        check_gdemo(chain)

        alg2 = IPMCMC(15, 100, 10)
        chain2 = sample(MoGtest_default, alg2)
        check_MoGtest_default(chain2, eps=0.2)
    end
end
