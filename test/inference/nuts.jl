using Turing: Sampler

@testset "nuts.jl" begin
    @testset "nuts inference" begin
        alg = NUTS(5000, 1000, 0.65)
        res = sample(gdemo_default, alg)
        check_gdemo(res[1000:end, :, :])
    end
    @testset "nuts constructor" begin
        alg = NUTS(1000, 200, 0.65)
        sampler = Sampler(alg)

        alg = NUTS(1000, 0.65)
        sampler = Sampler(alg)

        alg = NUTS(1000, 200, 0.65, :m)
        sampler = Sampler(alg)
    end
end
