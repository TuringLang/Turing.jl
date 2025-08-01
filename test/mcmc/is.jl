module ISTests

using Distributions: Normal, sample
using DynamicPPL: logpdf
using Random: Random
using StatsFuns: logsumexp
using Test: @test, @testset
using Turing

@testset "is.jl" begin
    function reference(n)
        as = Vector{Float64}(undef, n)
        bs = Vector{Float64}(undef, n)
        logps = Vector{Float64}(undef, n)

        for i in 1:n
            as[i], bs[i], logps[i] = reference()
        end
        logevidence = logsumexp(logps) - log(n)

        return (as=as, bs=bs, logps=logps, logevidence=logevidence)
    end

    function reference()
        x = rand(Normal(4, 5))
        y = rand(Normal(x, 1))
        loglik = logpdf(Normal(x, 2), 3) + logpdf(Normal(y, 2), 1.5)
        return x, y, loglik
    end

    @model function normal()
        a ~ Normal(4, 5)
        3 ~ Normal(a, 2)
        b ~ Normal(a, 1)
        1.5 ~ Normal(b, 2)
        return a, b
    end

    alg = IS()
    seed = 0
    n = 10

    model = normal()
    for i in 1:100
        Random.seed!(seed)
        ref = reference(n)

        Random.seed!(seed)
        chain = sample(model, alg, n; check_model=false)
        sampled = get(chain, [:a, :b, :loglikelihood])

        @test vec(sampled.a) == ref.as
        @test vec(sampled.b) == ref.bs
        @test vec(sampled.loglikelihood) == ref.logps
        @test chain.logevidence == ref.logevidence
    end

    @testset "logevidence" begin
        Random.seed!(100)

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
