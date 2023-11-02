@testset "advi.jl" begin
    @turing_testset "advi constructor" begin
        Random.seed!(0)
        s1 = ADVI()
        q  = vi(gdemo_default, s1, 1000)
        c1 = rand(q, 500)
    end

    @numerical_testset "advi inference" begin
        opt = Adam()
        Random.seed!(1)
        N = 500

        alg = ADVI(10)
        q = vi(gdemo_default, alg, 5000; optimizer = opt)
        samples = transpose(rand(q, N))
        chn = Chains(reshape(samples, size(samples)..., 1), ["s", "m"])

        # TODO: uhmm, seems like a large `eps` here...
        check_gdemo(chn, atol = 0.5)
    end

    @turing_testset "advi user supplied q interface" begin
        Random.seed!(1234)

        opt = Adam()
        Random.seed!(1)
        N = 500

        q = meanfield(model)
        alg = ADVI(10)
        q = vi(gdemo_default, alg, q, 5000; optimizer = opt)
        samples = transpose(rand(q, N))
        chn = Chains(reshape(samples, size(samples)..., 1), ["s", "m"])

        check_gdemo(chn, atol = 0.5)
    end

    # regression test for:
    # https://github.com/TuringLang/Turing.jl/issues/2065
    @turing_testset "simplex bijector" begin
        @model function dirichlet()
            x ~ Dirichlet([1.0,1.0])
            return x
        end
        
        m = dirichlet()
        b = bijector(m)
        x0 = m()
        z0 = b(x0)
        @test size(z0) == (1,)
        x0_inv = inverse(b)(z0)
        @test size(x0_inv) == size(x0)
        @test all(x0 .≈ x0_inv)
    end
end
