using Turing, Random, Test
using Turing: NUTS, Sampler

include("../test_utils/AllUtils.jl")

@testset "nuts.jl" begin
    @numerical_testset "nuts inference" begin
        alg = NUTS(5000, 1000, 0.65)
        res = sample(gdemo_default, alg)
        check_gdemo(res[1000:end, :, :])
    end
    @turing_testset "nuts constructor" begin
        alg = NUTS(1000, 200, 0.65)
        sampler = Sampler(alg)

        alg = NUTS(1000, 0.65)
        sampler = Sampler(alg)

        alg = NUTS(1000, 200, 0.65, :m)
        sampler = Sampler(alg)
    end
end
