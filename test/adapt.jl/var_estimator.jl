using Turing: VarEstimator, add_sample!, get_var, reset!
using Test

let
    D = 1000
    ve = VarEstimator(0, zeros(D), zeros(D))
    add_sample!(ve, randn(D))
    reset!(ve)

    # Check that reseting zeros everything.
    @test ve.n === 0
    @test ve.μ == zeros(D)
    @test ve.M == zeros(D)

    # Ensure that asking for the variance doesn't mutate the VarEstimator.
    add_sample!(ve, randn(D))
    add_sample!(ve, randn(D))
    μ, M = deepcopy(ve.μ), deepcopy(ve.M)
    get_var(ve)
    @test ve.μ == μ
    @test ve.M == M
end

# Check that the estimated variance is approximately correct.
let
    D = 10
    ve = VarEstimator(0, zeros(D), zeros(D))

    for _ = 1:10000
        s = randn(D)
        add_sample!(ve, s)
    end

    var = get_var(ve)

    @test var ≈ ones(D) atol=0.5
end
