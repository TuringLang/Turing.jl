using Turing, Random, Test
using Turing: Sampler

include("../test_utils/AllUtils.jl")

@testset "hmcda.jl" begin
    Random.seed!(1234)
    
    @numerical_testset "hmcda inference" begin
        alg1 = HMCDA(3000, 1000, 0.65, 0.015)
        # alg2 = Gibbs(3000, HMCDA(1, 200, 0.65, 0.35, :m), HMC(1, 0.25, 3, :s))
        alg3 = Gibbs(1500,
            PG(30, 10, :s),
            HMCDA(1, 500, 0.65, 0.005, :m))
        # alg3 = Gibbs(2000, HMC(1, 0.25, 3, :m), PG(30, 3, :s))
        # alg3 = PG(50, 2000)

        res1 = sample(gdemo_default, alg1)
        check_gdemo(res1)

        # res2 = sample(gdemo([1.5, 2.0]), alg2)
        #
        # @test mean(res2[:s]) ≈ 49/24 atol=0.2
        # @test mean(res2[:m]) ≈ 7/6 atol=0.2

        res3 = sample(gdemo_default, alg3)
        check_gdemo(res3)
    end
    @turing_testset "hmcda constructor" begin
        alg = HMCDA(1000, 0.65, 0.75)
        println(alg)
        sampler = Sampler(alg)

        alg = HMCDA(200, 0.65, 0.75, :m)
        println(alg)
        sampler = Sampler(alg)

        alg = HMCDA(1000, 200, 0.65, 0.75)
        println(alg)
        sampler = Sampler(alg)

        alg = HMCDA(1000, 200, 0.65, 0.75, :s)
        println(alg)
        sampler = Sampler(alg)

        @test isa(alg, HMCDA)
        @test isa(sampler, Sampler{<:Turing.Hamiltonian})
    end
end
