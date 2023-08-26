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
end
