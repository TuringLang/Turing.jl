module VariationalOptimisersTests

using AdvancedVI: DecayedADAGrad, TruncatedADAGrad, apply!
import ForwardDiff
import ReverseDiff
using Test: @test, @testset
using Turing

function test_opt(ADPack, opt)
    θ = randn(10, 10)
    θ_fit = randn(10, 10)
    loss(x, θ_) = mean(sum(abs2, θ * x - θ_ * x; dims=1))
    for t in 1:(10 ^ 4)
        x = rand(10)
        Δ = ADPack.gradient(θ_ -> loss(x, θ_), θ_fit)
        Δ = apply!(opt, θ_fit, Δ)
        @. θ_fit = θ_fit - Δ
    end
    @test loss(rand(10, 100), θ_fit) < 0.01
    @test length(opt.acc) == 1
end
for opt in [TruncatedADAGrad(), DecayedADAGrad(1e-2)]
    test_opt(ForwardDiff, opt)
end
for opt in [TruncatedADAGrad(), DecayedADAGrad(1e-2)]
    test_opt(ReverseDiff, opt)
end

end
