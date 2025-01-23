module DynamicPPLCompilerTests

using ..NumericalTests: check_numerical
using LinearAlgebra: I
using Test: @test, @testset, @test_throws
using Turing

# TODO(penelopeysm): Move this to a DynamicPPL Test Utils module
# We use this a lot!
@model function gdemo_d()
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    1.5 ~ Normal(m, sqrt(s))
    2.0 ~ Normal(m, sqrt(s))
    return s, m
end
const gdemo_default = gdemo_d()

@testset "compiler.jl" begin
    @info "compiler.jl"

    @testset "assume" begin
        @info "assume"

        @model function test_assume()
            x ~ Bernoulli(1)
            y ~ Bernoulli(x / 2)
            return x, y
        end

        smc = SMC()
        pg = PG(10)

        res1 = sample(test_assume(), smc, 1000)
        res2 = sample(test_assume(), pg, 1000)

        check_numerical(res1, [:y], [0.5]; atol=0.1)
        check_numerical(res2, [:y], [0.5]; atol=0.1)

        # Check that all xs are 1.
        @test all(isone, res1[:x])
        @test all(isone, res2[:x])
    end

    @testset "beta binomial" begin
        @info "beta binomial"

        prior = Beta(2, 2)
        obs = [0, 1, 0, 1, 1, 1, 1, 1, 1, 1]
        exact = Beta(prior.α + sum(obs), prior.β + length(obs) - sum(obs))
        meanp = exact.α / (exact.α + exact.β)

        @model function testbb(obs)
            p ~ Beta(2, 2)
            x ~ Bernoulli(p)
            for i in eachindex(obs)
                obs[i] ~ Bernoulli(p)
            end
            return p, x
        end

        smc = SMC()
        pg = PG(10)
        gibbs = Gibbs(:p => HMC(0.2, 3), :x => PG(10))

        chn_s = sample(testbb(obs), smc, 1000)
        chn_p = sample(testbb(obs), pg, 2000)
        chn_g = sample(testbb(obs), gibbs, 1500)

        check_numerical(chn_s, [:p], [meanp]; atol=0.05)
        check_numerical(chn_p, [:x], [meanp]; atol=0.1)
        check_numerical(chn_g, [:x], [meanp]; atol=0.1)
    end

    @testset "model with global variables" begin
        @info "model with global variables"
        xs = [1.5 2.0]
        # xx = 1

        @model function fggibbstest(xs)
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))
            # xx ~ Normal(m, sqrt(s)) # this is illegal

            for i in eachindex(xs)
                xs[i] ~ Normal(m, sqrt(s))
                # for xx in xs
                # xx ~ Normal(m, sqrt(s))
            end
            return s, m
        end

        gibbs = Gibbs(:s => PG(10), :m => HMC(0.4, 8))
        chain = sample(fggibbstest(xs), gibbs, 2)
    end

    @testset "new grammar" begin
        @info "new grammar"
        x = Float64[1 2]

        @model function gauss(x)
            priors = Array{Float64}(undef, 2)
            priors[1] ~ InverseGamma(2, 3)         # s
            priors[2] ~ Normal(0, sqrt(priors[1])) # m
            for i in eachindex(x)
                x[i] ~ Normal(priors[2], sqrt(priors[1]))
            end
            return priors
        end

        chain = sample(gauss(x), PG(10), 10)
        chain = sample(gauss(x), SMC(), 10)

        # Test algorithm that does not support models with keyword arguments. See issue #2007 for more details.
        @model function gauss2(::Type{TV}=Vector{Float64}; x) where {TV}
            priors = TV(undef, 2)
            priors[1] ~ InverseGamma(2, 3)         # s
            priors[2] ~ Normal(0, sqrt(priors[1])) # m
            for i in eachindex(x)
                x[i] ~ Normal(priors[2], sqrt(priors[1]))
            end
            return priors
        end

        @test_throws ErrorException chain = sample(gauss2(; x=x), PG(10), 10)
        @test_throws ErrorException chain = sample(gauss2(; x=x), SMC(), 10)

        @test_throws ErrorException chain = sample(
            gauss2(DynamicPPL.TypeWrap{Vector{Float64}}(); x=x), PG(10), 10
        )
        @test_throws ErrorException chain = sample(
            gauss2(DynamicPPL.TypeWrap{Vector{Float64}}(); x=x), SMC(), 10
        )
    end

    @testset "new interface" begin
        @info "new interface"
        obs = [0, 1, 0, 1, 1, 1, 1, 1, 1, 1]

        @model function newinterface(obs)
            p ~ Beta(2, 2)
            for i in eachindex(obs)
                obs[i] ~ Bernoulli(p)
            end
            return p
        end

        chain = sample(
            newinterface(obs),
            HMC(0.75, 3, :p, :x; adtype=AutoForwardDiff(; chunksize=2)),
            100,
        )
    end

    @testset "no return" begin
        @info "no return"
        @model function noreturn(x)
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))
            for i in eachindex(x)
                x[i] ~ Normal(m, sqrt(s))
            end
        end

        chain = sample(noreturn([1.5 2.0]), HMC(0.15, 6), 1000)
        check_numerical(chain, [:s, :m], [49 / 24, 7 / 6])
    end

    @testset "observe with literals" begin
        @info "observe with literals"
        @model function test()
            z ~ Normal(0, 1)
            x ~ Bernoulli(1)
            1 ~ Bernoulli(x / 2)
            0 ~ Bernoulli(x / 2)
            return x
        end

        is = IS()
        smc = SMC()
        pg = PG(10)

        res_is = sample(test(), is, 10000)
        res_smc = sample(test(), smc, 1000)
        res_pg = sample(test(), pg, 100)

        @test all(isone, res_is[:x])
        @test res_is.logevidence ≈ 2 * log(0.5)

        @test all(isone, res_smc[:x])
        @test res_smc.logevidence ≈ 2 * log(0.5)

        @test all(isone, res_pg[:x])
    end

    @testset "sample" begin
        @info "sample"
        alg = Gibbs(:m => HMC(0.2, 3), :s => PG(10))
        chn = sample(gdemo_default, alg, 1000)
    end

    @testset "vectorization @." begin
        @info "vectorization @."
        @model function vdemo1(x)
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))
            @. x ~ Normal(m, sqrt(s))
            return s, m
        end

        alg = HMC(0.01, 5)
        x = randn(100)
        res = sample(vdemo1(x), alg, 250)

        @model function vdemo1b(x)
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))
            @. x ~ Normal(m, $(sqrt(s)))
            return s, m
        end

        res = sample(vdemo1b(x), alg, 250)

        @model function vdemo2(x)
            μ ~ MvNormal(zeros(size(x, 1)), I)
            @. x ~ $(MvNormal(μ, I))
        end

        D = 2
        alg = HMC(0.01, 5)
        res = sample(vdemo2(randn(D, 100)), alg, 250)

        # Vector assumptions
        N = 10
        alg = HMC(0.2, 4; adtype=AutoForwardDiff(; chunksize=N))

        @model function vdemo3()
            x = Vector{Real}(undef, N)
            for i in 1:N
                x[i] ~ Normal(0, sqrt(4))
            end
        end

        t_loop = @elapsed res = sample(vdemo3(), alg, 1000)

        # Test for vectorize UnivariateDistribution
        @model function vdemo4()
            x = Vector{Real}(undef, N)
            @. x ~ Normal(0, 2)
        end

        t_vec = @elapsed res = sample(vdemo4(), alg, 1000)

        @model vdemo5() = x ~ MvNormal(zeros(N), 4 * I)

        t_mv = @elapsed res = sample(vdemo5(), alg, 1000)

        println("Time for")
        println("  Loop : ", t_loop)
        println("  Vec  : ", t_vec)
        println("  Mv   : ", t_mv)

        # Transformed test
        @model function vdemo6()
            x = Vector{Real}(undef, N)
            @. x ~ InverseGamma(2, 3)
        end

        sample(vdemo6(), alg, 1000)

        N = 3
        @model function vdemo7()
            x = Array{Real}(undef, N, N)
            @. x ~ [InverseGamma(2, 3) for i in 1:N]
        end

        sample(vdemo7(), alg, 1000)
    end

    @testset "vectorization .~" begin
        @info "vectorization .~"
        @model function vdemo1(x)
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))
            x .~ Normal(m, sqrt(s))
            return s, m
        end

        alg = HMC(0.01, 5)
        x = randn(100)
        res = sample(vdemo1(x), alg, 250)

        @model function vdemo2(x)
            μ ~ MvNormal(zeros(size(x, 1)), I)
            return x .~ MvNormal(μ, I)
        end

        D = 2
        alg = HMC(0.01, 5)
        res = sample(vdemo2(randn(D, 100)), alg, 250)

        # Vector assumptions
        N = 10
        alg = HMC(0.2, 4; adtype=AutoForwardDiff(; chunksize=N))

        @model function vdemo3()
            x = Vector{Real}(undef, N)
            for i in 1:N
                x[i] ~ Normal(0, sqrt(4))
            end
        end

        t_loop = @elapsed res = sample(vdemo3(), alg, 1000)

        # Test for vectorize UnivariateDistribution
        @model function vdemo4()
            x = Vector{Real}(undef, N)
            return x .~ Normal(0, 2)
        end

        t_vec = @elapsed res = sample(vdemo4(), alg, 1000)

        @model vdemo5() = x ~ MvNormal(zeros(N), 4 * I)

        t_mv = @elapsed res = sample(vdemo5(), alg, 1000)

        println("Time for")
        println("  Loop : ", t_loop)
        println("  Vec  : ", t_vec)
        println("  Mv   : ", t_mv)

        # Transformed test
        @model function vdemo6()
            x = Vector{Real}(undef, N)
            return x .~ InverseGamma(2, 3)
        end

        sample(vdemo6(), alg, 1000)

        @model function vdemo7()
            x = Array{Real}(undef, N, N)
            return x .~ [InverseGamma(2, 3) for i in 1:N]
        end

        sample(vdemo7(), alg, 1000)
    end

    @testset "Type parameters" begin
        @info "Type parameters"
        N = 10
        alg = HMC(0.01, 5; adtype=AutoForwardDiff(; chunksize=N))
        x = randn(1000)
        @model function vdemo1(::Type{T}=Float64) where {T}
            x = Vector{T}(undef, N)
            for i in 1:N
                x[i] ~ Normal(0, sqrt(4))
            end
        end

        t_loop = @elapsed res = sample(vdemo1(), alg, 250)
        t_loop = @elapsed res = sample(vdemo1(DynamicPPL.TypeWrap{Float64}()), alg, 250)

        vdemo1kw(; T) = vdemo1(T)
        t_loop = @elapsed res = sample(
            vdemo1kw(; T=DynamicPPL.TypeWrap{Float64}()), alg, 250
        )

        @model function vdemo2(::Type{T}=Float64) where {T<:Real}
            x = Vector{T}(undef, N)
            @. x ~ Normal(0, 2)
        end

        t_vec = @elapsed res = sample(vdemo2(), alg, 250)
        t_vec = @elapsed res = sample(vdemo2(DynamicPPL.TypeWrap{Float64}()), alg, 250)

        vdemo2kw(; T) = vdemo2(T)
        t_vec = @elapsed res = sample(
            vdemo2kw(; T=DynamicPPL.TypeWrap{Float64}()), alg, 250
        )

        @model function vdemo3(::Type{TV}=Vector{Float64}) where {TV<:AbstractVector}
            x = TV(undef, N)
            @. x ~ InverseGamma(2, 3)
        end

        sample(vdemo3(), alg, 250)
        sample(vdemo3(DynamicPPL.TypeWrap{Vector{Float64}}()), alg, 250)

        vdemo3kw(; T) = vdemo3(T)
        sample(vdemo3kw(; T=DynamicPPL.TypeWrap{Vector{Float64}}()), alg, 250)
    end
end

end  # module
