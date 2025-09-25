module ISTests

using DynamicPPL: logpdf
using Random: Random
using StableRNGs: StableRNG
using StatsFuns: logsumexp
using Test: @test, @testset
using Turing

@testset "is.jl" begin
    @testset "numerical accuracy" begin
        function reference(n)
            rng = StableRNG(468)
            as = Vector{Float64}(undef, n)
            bs = Vector{Float64}(undef, n)

            for i in 1:n
                as[i] = rand(rng, Normal(4, 5))
                bs[i] = rand(rng, Normal(as[i], 1))
            end
            # logevidence = logsumexp(logps) - log(n)
            return (as=as, bs=bs)
        end

        @model function normal()
            a ~ Normal(4, 5)
            3 ~ Normal(a, 2)
            b ~ Normal(a, 1)
            1.5 ~ Normal(b, 2)
            return a, b
        end

        function expected_loglikelihoods(as, bs)
            return logpdf.(Normal.(as, 2), 3) .+ logpdf.(Normal.(bs, 2), 1.5)
        end

        alg = IS()
        N = 1000
        model = normal()
        chain = sample(StableRNG(468), model, alg, N)
        ref = reference(N)

        @test isapprox(mean(chain[:a]), mean(ref.as); atol=0.1)
        @test isapprox(mean(chain[:b]), mean(ref.bs); atol=0.1)
        @test isapprox(chain[:loglikelihood], expected_loglikelihoods(chain[:a], chain[:b]))
        @test isapprox(chain.logevidence, logsumexp(chain[:loglikelihood]) - log(N))
    end

    @testset "logevidence" begin
        @model function test()
            a ~ Normal(0, 1)
            x ~ Bernoulli(1)
            b ~ Gamma(2, 3)
            1 ~ Bernoulli(x / 2)
            c ~ Beta()
            0 ~ Bernoulli(x / 2)
            return x
        end

        chains = sample(test(), IS(), 1_000)

        @test all(isone, chains[:x])
        @test chains.logevidence ≈ -2 * log(2)
    end
end

end
