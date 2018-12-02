using Test, Turing, Distributions
using Turing: VarInfo, gradient_forward, gradient_reverse

@model foo_ad() = begin
    x ~ Normal(3, 1)
    y ~ Normal(x, 1)
    return x, y
end

# Check that gradient_forward doesn't change the RV values or logp of a VarInfo.
let
    # Construct model.
    model, sampler, vi = foo_ad(), nothing, VarInfo()
    model(vi, sampler)

    # Record initial values.
    θ, logp = deepcopy(vi.vals), deepcopy(vi.logp)

    # Compute gradient using ForwardDiff.
    gradient_forward(deepcopy(θ), vi, model, sampler)

    # Check that θ and logp haven't changed.
    @test θ == vi.vals
    @test logp == vi.logp
end

# Check that gradient_reverse doesn't change the RV values or logp of a VarInfo.
let
    # Construct model
    model, sampler, vi = foo_ad(), nothing, VarInfo()
    model(vi, sampler)

    # Record initial values.
    θ, logp = deepcopy(vi.vals), deepcopy(vi.logp)

    # Compute gradient using ForwardDiff.
    gradient_reverse(deepcopy(θ), vi, model, sampler)

    # Check that θ and logp haven't changed.
    @test θ == vi.vals
    @test logp == vi.logp
end
