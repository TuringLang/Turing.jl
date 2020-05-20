using Turing, Random, Test
using DynamicPPL: getlogp
import MCMCChains

using Random

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

struct DynamicDist <: DiscreteMultivariateDistribution end
function Distributions.logpdf(::DynamicDist, dsl_numeric::AbstractVector{Int})
    return sum([log(0.5) * 0.5^i for i in 1:length(dsl_numeric)])
end
function Random.rand(rng::Random.AbstractRNG, ::DynamicDist)
    fst = rand(rng, [0, 1])
    dsl_numeric = [fst]
    while rand() < 0.5
        push!(dsl_numeric, rand(rng, [0, 1]))
    end
    return dsl_numeric
end

@testset "io.jl" begin
    # Only test threading if 1.3+.
    if VERSION > v"1.2"
        @testset "threaded sampling" begin
            # Test that chains with the same seed will sample identically.
            chain1 = sample(Random.seed!(5), gdemo_default, HMC(0.1, 7), MCMCThreads(),
                            1000, 4)
            chain2 = sample(Random.seed!(5), gdemo_default, HMC(0.1, 7), MCMCThreads(),
                            1000, 4)
            #https://github.com/TuringLang/Turing.jl/issues/1260
            @test_skip all(chain1.value .== chain2.value)
            check_gdemo(chain1)

            # Smoke test for default sample call.
            chain = sample(gdemo_default, HMC(0.1, 7), MCMCThreads(), 1000, 4)
            check_gdemo(chain)

            # run sampler: progress logging should be disabled and
            # it should return a Chains object
            sampler = Turing.Sampler(HMC(0.1, 7), gdemo_default)
            chains = sample(gdemo_default, sampler, MCMCThreads(), 1000, 4)
            @test chains isa MCMCChains.Chains
        end
    end
    @testset "chain save/resume" begin
        Random.seed!(1234)

        alg1 = HMCDA(1000, 0.65, 0.15)
        alg2 = PG(20)
        alg3 = Gibbs(PG(30, :s), HMCDA(500, 0.65, 0.05, :m))

        chn1 = sample(gdemo_default, alg1, 5000; save_state=true)
        check_gdemo(chn1)

        chn1_resumed = Turing.Inference.resume(chn1, 1000)
        check_gdemo(chn1_resumed)

        chn1_contd = sample(gdemo_default, alg1, 5000; resume_from=chn1)
        check_gdemo(chn1_contd)

        chn1_contd2 = sample(gdemo_default, alg1, 5000; resume_from=chn1, reuse_spl_n=1000)
        check_gdemo(chn1_contd2)

        chn2 = sample(gdemo_default, alg2, 1000; save_state=true)
        check_gdemo(chn2)

        chn2_contd = sample(gdemo_default, alg2, 1000; resume_from=chn2)
        check_gdemo(chn2_contd)

        chn3 = sample(gdemo_default, alg3, 5000; save_state=true)
        check_gdemo(chn3)

        chn3_contd = sample(gdemo_default, alg3, 5000; resume_from=chn3)
        check_gdemo(chn3_contd)
    end
    @testset "Contexts" begin
        # Test LikelihoodContext
        @model testmodel(x) = begin
            a ~ Beta()
            lp1 = getlogp(_varinfo)
            x[1] ~ Bernoulli(a)
            global loglike = getlogp(_varinfo) - lp1
        end
        model = testmodel([1.0])
        varinfo = Turing.VarInfo(model)
        model(varinfo, Turing.SampleFromPrior(), Turing.LikelihoodContext())
        @test getlogp(varinfo) == loglike

        # Test MiniBatchContext
        @model testmodel(x) = begin
            a ~ Beta()
            x[1] ~ Bernoulli(a)
        end
        model = testmodel([1.0])
        varinfo1 = Turing.VarInfo(model)
        varinfo2 = deepcopy(varinfo1)
        model(varinfo1, Turing.SampleFromPrior(), Turing.LikelihoodContext())
        model(varinfo2, Turing.SampleFromPrior(), Turing.MiniBatchContext(Turing.LikelihoodContext(), 10))
        @test isapprox(getlogp(varinfo2) / getlogp(varinfo1), 10)
    end
    @testset "Prior" begin
        N = 5000

        # Note that all chains contain 3 values per sample: 2 variables + log probability
        Random.seed!(100)
        chains = sample(gdemo_d(), Prior(), N)
        @test chains isa MCMCChains.Chains
        @test size(chains) == (N, 3, 1)
        @test mean(chains, :s) ≈ 3 atol=0.1
        @test mean(chains, :m) ≈ 0 atol=0.1

        Random.seed!(100)
        chains = sample(gdemo_d(), Prior(), MCMCThreads(), N, 4)
        @test chains isa MCMCChains.Chains
        @test size(chains) == (N, 3, 4)
        @test mean(chains, :s) ≈ 3 atol=0.1
        @test mean(chains, :m) ≈ 0 atol=0.1

        Random.seed!(100)
        chains = sample(gdemo_d(), Prior(), N; chain_type = Vector{NamedTuple})
        @test chains isa Vector{<:NamedTuple}
        @test length(chains) == N
        @test all(length(x) == 3 for x in chains)
        @test all(haskey(x, :lp) for x in chains)
        @test mean(x[:s][1] for x in chains) ≈ 3 atol=0.1
        @test mean(x[:m][1] for x in chains) ≈ 0 atol=0.1
    end
    @testset "stochastic control flow" begin
        @model demo(p) = begin
            x ~ Categorical(p)
            if x == 1
                y ~ Normal()
            elseif x == 2
                z ~ Normal()
            else 
                k ~ Normal()
            end
        end
        chain = sample(demo(fill(1/3, 3)), PG(4), 7000)
        check_numerical(chain, [:x, :y, :z, :k], [2, 0, 0, 0], atol=0.05, skip_missing=true)

        chain = sample(demo(fill(1/3, 3)), Gibbs(PG(4, :x, :y), PG(4, :z, :k)), 7000)
        check_numerical(chain, [:x, :y, :z, :k], [2, 0, 0, 0], atol=0.05, skip_missing=true)

        @model function mwe()
		    dsl ~ DynamicDist()
		end
        chain = sample(mwe(), PG(10), 500)
    end
end
