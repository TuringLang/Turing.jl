using Test, Turing, Distributions
using Turing: VarInfo, gradient_logp_forward, gradient_logp_reverse

@model foo_ad() = begin
    x ~ Normal(3, 1)
    y ~ Normal(x, 1)
end

# Check that gradient_logp_forward doesn't change the RV values or logp of a VarInfo.
let
    # Construct model.
    model, sampler, vi = foo_ad(), nothing, VarInfo()
    model(vi, sampler)

    # Record initial values.
    θ, logp = deepcopy(vi.vals), deepcopy(vi.logp)

    # Compute gradient using ForwardDiff.
    gradient_logp_forward(deepcopy(θ), vi, model, sampler)

    # Check that θ and logp haven't changed.
    @test θ == vi.vals
    @test logp == vi.logp
end

# Check that gradient_logp_reverse doesn't change the RV values or logp of a VarInfo.
let
    # Construct model
    model, sampler, vi = foo_ad(), nothing, VarInfo()
    model(vi, sampler)

    # Record initial values.
    θ, logp = deepcopy(vi.vals), deepcopy(vi.logp)

    # Compute gradient using ForwardDiff.
    gradient_logp_reverse(deepcopy(θ), vi, model, sampler)

    # Check that θ and logp haven't changed.
    @test θ == vi.vals
    @test logp == vi.logp
end

# Check that gradient_logp_forward gives the same result as gradient_logp_reverse.
let
    # Construct model.
    model, sampler, vi = foo_ad(), nothing, VarInfo()
    model(vi, sampler)

    # Record initial values.
    θ, logp = deepcopy(vi.vals), deepcopy(vi.logp)

    # Compute gradient using ForwardDiff.
    l1, d1 = gradient_logp_forward(deepcopy(θ), vi, model, sampler)
    # Compute gradient using Flux.Tracker.
    l2, d2 = gradient_logp_reverse(deepcopy(θ), vi, model, sampler)

    # Check that the values are the same.
    @test l1 == l2
    @test d1 == d2
end
