using Turing, Random, Test
using Turing.Inference: _leapfrog

include("../../test_utils/AllUtils.jl")

@testset "hmc_core.jl" begin
    Random.seed!(150)
    D = 10
    lp_grad_func(x) = nothing, -x
    step_size = 0.1
    @turing_testset "1. single mutiple-step call v.s. plain Julia" begin
        for _ = 1:100
            theta = rand(D)
            p = rand(D)

            theta_turing, p_turing, _ = _leapfrog(theta, p, 1, step_size, lp_grad_func)

            p .+=
                step_size .* lp_grad_func(theta)[2] / 2
            theta +=
                step_size .* p
            p .+=
                step_size .* lp_grad_func(theta)[2] / 2

            @test theta ≈ theta_turing
            @test p ≈ p_turing
        end
    end
    @turing_testset "2. multiple single-step call v.s. single mutiple-step call" begin
        for _ = 1:100
            theta_0 = rand(2)
            p_0 = rand(2)

            theta_1, p_1, _ = _leapfrog(theta_0, p_0, 1, step_size, lp_grad_func)
            theta_2, p_2, _ = _leapfrog(theta_1, p_1, 1, step_size, lp_grad_func)

            theta_2_one_call, p_2_one_call, _ =
                _leapfrog(theta_0, p_0, 2, step_size, lp_grad_func)

            @test theta_2 ≈ theta_2_one_call
            @test p_2 ≈ p_2_one_call
        end
    end
end
