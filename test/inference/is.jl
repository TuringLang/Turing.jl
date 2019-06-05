using Turing, Random, Test
using StatsFuns

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@turing_testset "is.jl" begin
    function reference(n :: Int)
        logweights = zeros(Float64, n)
        samples = Array{Dict{Symbol,Any}}(undef, n)
        for i = 1:n
            samples[i] = reference()
            logweights[i] = samples[i][:logweight]
        end
        logevidence = logsumexp(logweights) - log(n)
        results = Dict{Symbol,Any}()
        results[:lp] = logevidence
        results[:lp] = logweights
        results[:samples] = samples
        return results
    end

    function reference()
        x = rand(Normal(4,5))
        y = rand(Normal(x,1))
        log_lik = logpdf(Normal(x,2), 3) + logpdf(Normal(y,2), 1.5)
        d = Dict()
        d[:logweight] = log_lik
        d[:a] = x
        d[:b] = y
        return d
    end

    let n = 10
      @model normal() = begin
        a ~ Normal(4,5)
        3 ~ Normal(a,2)
        b ~ Normal(a,1)
        1.5 ~ Normal(b,2)
        a, b
      end

      alg = IS(n)
      seed = 0

      _f = normal();
      for i=1:100
        Random.seed!(seed)
        exact = reference(n)
        Random.seed!(seed)
        tested = sample(_f, alg)
        t_vals = get(tested, [:a, :b, :lp])
        for i = 1:n
            @test exact[:samples][i][:a] == t_vals.a[i]
            @test exact[:samples][i][:b] == t_vals.b[i]
            @test exact[:lp][i] == t_vals.lp[i]
        end
        @test all(exact[:lp] .== t_vals.lp)
      end
    end
end
