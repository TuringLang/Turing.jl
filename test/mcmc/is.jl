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
            return (as=as, bs=bs)
        end

        @model function normal()
            a ~ Normal(4, 5)
            3 ~ Normal(a, 2)
            b ~ Normal(a, 1)
            1.5 ~ Normal(b, 2)
            return a, b
        end

        alg = IS()
        N = 1000
        model = normal()
        chain = sample(StableRNG(468), model, alg, N)
        ref = reference(N)

        # Note that in general, mean(chain) will differ from mean(ref). This is because the
        # sampling process introduces extra calls to rand(), etc. which changes the output.
        # These tests therefore are only meant to check that the results are qualitatively
        # similar to the reference implementation of IS, and hence the atol is set to
        # something fairly large.
        @test isapprox(mean(chain[:a]), mean(ref.as); atol=0.1)
        @test isapprox(mean(chain[:b]), mean(ref.bs); atol=0.1)

        function expected_loglikelihoods(as, bs)
            return logpdf.(Normal.(as, 2), 3) .+ logpdf.(Normal.(bs, 2), 1.5)
        end
        @test isapprox(chain[:loglikelihood], expected_loglikelihoods(chain[:a], chain[:b]))
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

        N = 1_000
        chains = sample(test(), IS(), N)

        @test all(isone, chains[:x])
        # The below is equivalent to log(mean(exp.(chains[:loglikelihood]))), but more
        # numerically stable
        logevidence = logsumexp(chains[:loglikelihood]) - log(N)
        @test logevidence ≈ -2 * log(2)
    end
end

end
