using Random, Turing, Test
import AbstractMCMC
import MCMCChains
import Turing.Inference
using Turing.RandomMeasures

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@testset "gibbs.jl" begin
    @turing_testset "gibbs constructor" begin
        N = 500
        s1 = Gibbs(HMC(0.1, 5, :s, :m))
        s2 = Gibbs(PG(10, :s, :m))
        s3 = Gibbs(PG(3, :s), HMC( 0.4, 8, :m))
        s4 = Gibbs(PG(3, :s), HMC(0.4, 8, :m))
        s5 = Gibbs(CSMC(3, :s), HMC(0.4, 8, :m))
        s6 = Gibbs(HMC(0.1, 5, :s), ESS(:m))


        c1 = sample(gdemo_default, s1, N)
        c2 = sample(gdemo_default, s2, N)
        c3 = sample(gdemo_default, s3, N)
        c4 = sample(gdemo_default, s4, N)
        c5 = sample(gdemo_default, s5, N)
        c6 = sample(gdemo_default, s6, N)

        # Test gid of each samplers
        g = Turing.Sampler(s3, gdemo_default)

        @test g.state.samplers[1].selector != g.selector
        @test g.state.samplers[2].selector != g.selector
        @test g.state.samplers[1].selector != g.state.samplers[2].selector

        # run sampler: progress logging should be disabled and
        # it should return a Chains object
        @test sample(gdemo_default, g, N) isa MCMCChains.Chains
    end
    @numerical_testset "gibbs inference" begin
        Random.seed!(100)
        alg = Gibbs(
            CSMC(10, :s),
            HMC(0.2, 4, :m))
        chain = sample(gdemo(1.5, 2.0), alg, 3000)
        check_numerical(chain, [:s, :m], [49/24, 7/6], atol=0.1)

        Random.seed!(100)

        alg = Gibbs(
            MH(:s),
            HMC(0.2, 4, :m))
        chain = sample(gdemo(1.5, 2.0), alg, 5000)
        check_numerical(chain, [:s, :m], [49/24, 7/6], atol=0.1)

        alg = Gibbs(
            CSMC(15, :s),
            ESS(:m))
        chain = sample(gdemo(1.5, 2.0), alg, 10_000)
        check_numerical(chain, [:s, :m], [49/24, 7/6], atol=0.1)

        alg = CSMC(10)
        chain = sample(gdemo(1.5, 2.0), alg, 5000)
        check_numerical(chain, [:s, :m], [49/24, 7/6], atol=0.1)

        setadsafe(true)

        Random.seed!(200)
        gibbs = Gibbs(
            PG(10, :z1, :z2, :z3, :z4),
            HMC(0.15, 3, :mu1, :mu2))
        chain = sample(MoGtest_default, gibbs, 1500)
        check_MoGtest_default(chain, atol = 0.15)

        setadsafe(false)

        Random.seed!(200)
        gibbs = Gibbs(
            PG(10, :z1, :z2, :z3, :z4),
            ESS(:mu1), ESS(:mu2))
        chain = sample(MoGtest_default, gibbs, 1500)
        check_MoGtest_default(chain, atol = 0.15)
    end

    @turing_testset "transitions" begin
        @model gdemo_copy() = begin
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))
            1.5 ~ Normal(m, sqrt(s))
            2.0 ~ Normal(m, sqrt(s))
            return s, m
        end
        model = gdemo_copy()

        function AbstractMCMC.sample_end!(
            ::AbstractRNG,
            ::typeof(model),
            ::Turing.Sampler{<:Gibbs},
            ::Integer,
            transitions::Vector;
            kwargs...
        )
            transitions isa Vector{<:Inference.Transition} ||
                error("incorrect transitions")
            return
        end

        function callback(rng, model, sampler, transition, i; kwargs...)
            transition isa Inference.GibbsTransition || error("incorrect transition")
            return
        end

        alg = Gibbs(MH(:s), HMC(0.2, 4, :m))
        sample(model, alg, 100; callback = callback)
    end
    
    @turing_testset "dynamic model" begin
        @model imm(y, alpha, ::Type{M}=Vector{Float64}) where {M} = begin
            N = length(y)
            rpm = DirichletProcess(alpha)
        
            z = tzeros(Int, N)
            cluster_counts = tzeros(Int, N)
            fill!(cluster_counts, 0)
        
            for i in 1:N
                z[i] ~ ChineseRestaurantProcess(rpm, cluster_counts)
                cluster_counts[z[i]] += 1
            end
        
            Kmax = findlast(!iszero, cluster_counts)
            m = M(undef, Kmax)
            for k = 1:Kmax
                m[k] ~ Normal(1.0, 1.0)
            end
        end
        model = imm(randn(100), 1.0);
        sample(model, Gibbs(MH(10, :z), HMC(0.01, 4, :m)), 100);
        sample(model, Gibbs(PG(10, :z), HMC(0.01, 4, :m)), 100);
    end
    
    @turing_testset "gibbs conditionals" begin
        let α₀ = 2.0,
            θ₀ = inv(3.0),
            x = [1.5, 2.0]
            
            @model inverse_gdemo(x) = begin
                λ ~ Gamma(α₀, θ₀)
                m ~ Normal(0, √(1 / λ))
                x .~ Normal(m, √(1 / λ))
            end

            function gdemo_statistics(x)
                # The conditionals and posterior can be formulated in terms of the following statistics:
                N = length(x) # number of samples
                x̄ = mean(x) # sample mean
                s² = var(x; mean=x̄, corrected=false) # sample variance
                return N, x̄, s²
            end

            function cond_m(c)
                N, x̄, s² = gdemo_statistics(x)
                mₙ = N * x̄ / (N + 1)
                λₙ = c.λ * (N + 1)
                σₙ = √(1 / λₙ)
                return Normal(mₙ, σₙ)
            end

            function cond_λ(c)
                N, x̄, s² = gdemo_statistics(x)
                αₙ = α₀ + (N - 1) / 2
                βₙ = (s² * N / 2 + c.m^2 / 2 + inv(θ₀))
                return Gamma(αₙ, inv(βₙ))
            end

            Random.seed!(100)
            alg = Gibbs(
                GibbsConditional(:m, cond_m),
                GibbsConditional(:λ, cond_λ))
            chain = sample(inverse_gdemo(x), alg, 3000)
            check_numerical(chain, [:m, :λ], [7/6, 24/49], atol=0.1)
        end

        let π = [0.5, 0.5],
            K = length(π),
            λ = 5.0,
            σ = 1.0,
            x = [rand(10); 2 .+ rand(10)],
            N = length(x)

            @model mixture(x) = begin
                μ ~ arraydist(Normal.(0, fill(λ, K)))
                z ~ arraydist(Categorical.(fill(π, N)))
                x ~ arraydist(Normal.(μ[z], σ))
                
                return x
            end

            # see http://www.cs.columbia.edu/~blei/fogm/2015F/notes/mixtures-and-gibbs.pdf
            function cond_z(c)
                μ = c.μ
                function ϖ(ξ)
                    p = π .* pdf.(Normal.(μ, σ), Ref(ξ))
                    return p / sum(p)
                end
                return arraydist([Categorical(ϖ(x[n])) for n = 1:N])
            end

            function cond_μ(c)
                z = c.z
                n = [count(z .== k) for k = 1:K]
                x̄ = [sum(x[z .== k]) / (n[k] == 0 ? Inf : n[k]) for k = 1:K]
                λ̂ = [inv(n[k] / σ^2 + 1/λ^2) for k = 1:K]
                μ̂ = [x̄[k] * (n[k]/σ^2) * λ̂[k] for k = 1:K]

                return MvNormal(μ̂, λ̂)
            end

            Random.seed!(100)
            alg = Gibbs(
                GibbsConditional(:z, cond_z),
                GibbsConditional(:μ, cond_μ))
            chain = sample(mixture(x), alg, 3000)
            check_numerical(chain, [:z, :μ], [[fill(1, 10); fill(2, 10)], [0.0, 2..5]], atol=0.1)
        end
    end
end
