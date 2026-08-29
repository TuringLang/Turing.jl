module RandomMeasuresTests

using Distributions: Normal, logpdf, sample
using Random: randn
using StableRNGs: StableRNG
using Statistics: mean, std
using Test: @test, @testset
using Turing

const TuringDistributionsExt = Base.get_extension(Turing, :TuringDistributionsExt)
using .TuringDistributionsExt:
    ChineseRestaurantProcess,
    DirichletProcess,
    PitmanYorProcess,
    SizeBiasedSamplingProcess,
    StickBreakingProcess

@testset "TuringDistributionsExt" begin
    @test TuringDistributionsExt !== nothing

    @testset "representations" begin
        dirichlet_process = DirichletProcess(0.5)
        chinese_restaurant = ChineseRestaurantProcess(dirichlet_process, Int32[2, 1])
        probabilities = exp.([logpdf(chinese_restaurant, k) for k in Int32(1):Int32(3)])
        @test probabilities ≈ [2, 1, 0.5] ./ 3.5

        chinese_restaurant = ChineseRestaurantProcess(dirichlet_process, Int32[2, 0, 1])
        probabilities = exp.([logpdf(chinese_restaurant, k) for k in Int32(1):Int32(3)])
        @test probabilities ≈ [2, 0.5, 1] ./ 3.5

        integer_process = DirichletProcess(1)
        @test logpdf(ChineseRestaurantProcess(integer_process, [1]), 1) ≈ -log(2)

        pitman_yor_process = PitmanYorProcess(0.25, 0.5, 2)
        chinese_restaurant = ChineseRestaurantProcess(pitman_yor_process, [2, 1])
        probabilities = exp.([logpdf(chinese_restaurant, k) for k in 1:3])
        @test probabilities ≈ [1.75, 0.75, 1.0] ./ 3.5

        chinese_restaurant = ChineseRestaurantProcess(pitman_yor_process, [2, 0, 1])
        probabilities = exp.([logpdf(chinese_restaurant, k) for k in 1:3])
        @test probabilities ≈ [1.75, 1.0, 0.75] ./ 3.5

        @test minimum(StickBreakingProcess(dirichlet_process)) == 0
        @test maximum(StickBreakingProcess(dirichlet_process)) == 1
        @test minimum(SizeBiasedSamplingProcess(dirichlet_process, 2.0)) == 0
        @test maximum(SizeBiasedSamplingProcess(dirichlet_process, 2.0)) == 2
        @test TuringDistributionsExt.stickbreak([0.2, 0.5]) ≈ [0.2, 0.4, 0.4]
        @test TuringDistributionsExt.stickbreak(Float32[]) == Float32[1]
    end

    @testset "infinite mixture model" begin
        @model function infinite_gmm(x)
            random_measure = DirichletProcess(1.0)
            base_distribution = Normal(0.0, 1.0)
            assignments = zeros(Int, length(x))
            locations = zeros(length(x))

            for i in eachindex(x)
                nclusters = maximum(assignments)
                counts = Int[count(==(k), assignments) for k in 1:nclusters]
                assignments[i] ~ ChineseRestaurantProcess(random_measure, counts)
                if assignments[i] > nclusters
                    locations[assignments[i]] ~ base_distribution
                end
                x[i] ~ Normal(locations[assignments[i]], 1.0)
            end
        end

        rng = StableRNG(1)
        data = vcat(randn(rng, 10), randn(rng, 10) .- 5, randn(rng, 10) .+ 10)
        data = (data .- mean(data)) ./ std(data)

        chain = sample(StableRNG(2), infinite_gmm(data), SMC(), 500)
        @test chain isa VNChain
    end
end

end
