using Turing, Distributions, Test, Random

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

Random.seed!(129)

@turing_testset "prob_macro" begin
    @testset "scalar" begin
        @model demo(x) = begin
            m ~ Normal()
            x ~ Normal(m, 1)
        end

        mval = 3
        xval = 2
        iters = 1000

        logprior = logpdf(Normal(), mval)
        loglike = logpdf(Normal(mval, 1), xval)
        logjoint = logprior + loglike

        @test logprob"m = mval | model = demo" == logprior
        @test logprob"m = mval | x = xval, model = demo" == logprior
        @test logprob"x = xval | m = mval, model = demo" == loglike
        @test logprob"x = xval, m = mval | model = demo" == logjoint

        varinfo = Turing.VarInfo(demo(xval))
        @test logprob"m = mval | model = demo, varinfo = varinfo" == logprior
        @test logprob"m = mval | x = xval, model = demo, varinfo = varinfo" == logprior
        @test logprob"x = xval | m = mval, model = demo, varinfo = varinfo" == loglike
        varinfo = Turing.VarInfo(demo(missing))
        @test logprob"x = xval, m = mval | model = demo, varinfo = varinfo" == logjoint

        chain = sample(demo(xval), IS(), iters)
        chain2 = Chains(chain.value, chain.logevidence, chain.name_map, NamedTuple())
        lps = logpdf.(Normal.(vec(chain["m"].value), 1), xval)
        @test logprob"x = xval | chain = chain" == lps
        @test logprob"x = xval | chain = chain2, model = demo" == lps
        varinfo = Turing.VarInfo(demo(xval))
        @test logprob"x = xval | chain = chain, varinfo = varinfo" == lps
        @test logprob"x = xval | chain = chain2, model = demo, varinfo = varinfo" == lps
    end

    @testset "vector" begin
        n = 5
        @model demo(x, n = n, ::Type{T} = Float64) where {T} = begin
            m = Vector{T}(undef, n)
            @. m ~ Normal()
            @. x ~ Normal.(m, 1)
        end
        mval = rand(n)
        xval = rand(n)
        iters = 1000

        logprior = sum(logpdf.(Normal(), mval))
        like(m, x) = sum(logpdf.(Normal.(m, 1), x))
        loglike = like(mval, xval)
        logjoint = logprior + loglike

        @test logprob"m = mval | model = demo" == logprior
        @test logprob"x = xval | m = mval, model = demo" == loglike
        @test logprob"x = xval, m = mval | model = demo" == logjoint

        varinfo = Turing.VarInfo(demo(xval))
        @test logprob"m = mval | model = demo, varinfo = varinfo" == logprior
        @test logprob"x = xval | m = mval, model = demo, varinfo = varinfo" == loglike
        # Currently, we cannot easily pre-allocate `VarInfo` for vector data

        chain = sample(demo(xval), HMC(0.5, 1), iters)
        chain2 = Chains(chain.value, chain.logevidence, chain.name_map, NamedTuple())
        lps = like.([[chain["m[$i]"].value[j] for i in 1:n] for j in 1:iters], Ref(xval))
        @test logprob"x = xval | chain = chain" == lps
        @test logprob"x = xval | chain = chain2, model = demo" == lps
        @test logprob"x = xval | chain = chain, varinfo = varinfo" == lps
        @test logprob"x = xval | chain = chain2, model = demo, varinfo = varinfo" == lps
    end
end
