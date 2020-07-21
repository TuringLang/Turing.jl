using LinearAlgebra

using Turing
import Turing.Inference: ABC

import AdvancedMH
const AMH = AdvancedMH

Turing.turnprogress(false)

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@testset "abc.jl" begin
    @turing_testset "abc inference" begin
        Random.seed!(125)

        f(x, y) = norm(x - y) / length(x) # mean squared error
        proposal = (m = AMH.RandomWalkProposal(Normal(0.0, 0.1)), );
        alg = ABC(proposal, f; epsilon = 0.5);

        chain = sample(gdemo_default, alg, 1000)
        check_gdemo(chain, atol = 0.1)
    end
end
