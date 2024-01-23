@testset "ad.jl" begin
    @testset "Simplex Tracker, Zygote and ReverseDiff (with and without caching) AD" begin
        @model function dir()
            theta ~ Dirichlet(1 ./ fill(4, 4))
        end
        sample(dir(), HMC(0.01, 1; adtype=AutoZygote()), 1000)
        sample(dir(), HMC(0.01, 1; adtype=AutoReverseDiff(false)), 1000)
        sample(dir(), HMC(0.01, 1; adtype=AutoReverseDiff(true)), 1000)
    end
    
    @testset "PDMatDistribution AD" begin
        @model function wishart()
            theta ~ Wishart(4, Matrix{Float64}(I, 4, 4))
        end

        sample(wishart(), HMC(0.01, 1; adtype=AutoReverseDiff(false)), 1000)
        sample(wishart(), HMC(0.01, 1; adtype=AutoZygote()), 1000)

        @model function invwishart()
            theta ~ InverseWishart(4, Matrix{Float64}(I, 4, 4))
        end

        sample(invwishart(), HMC(0.01, 1; adtype=AutoReverseDiff(false)), 1000)
        sample(invwishart(), HMC(0.01, 1; adtype=AutoZygote()), 1000)
    end

    @testset "memoization: issue #1393" begin
        @model function demo(data)
            sigma ~ Uniform(0.0, 20.0)
            data ~ Normal(0, sigma)
        end

        N = 1000
        for i in 1:5
            d = Normal(0.0, i)
            data = rand(d, N)
            chn = sample(demo(data), NUTS(0.65; adtype=AutoReverseDiff(true)), 1000)
            @test mean(Array(chn[:sigma])) ≈ std(data) atol = 0.5
        end
    end
end
