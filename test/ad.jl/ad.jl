using Test, Turing, Distributions
using Turing: VarInfo, gradient

@model foo_ad() = begin
    x ~ Normal(3, 1)
    y ~ Normal(x, 1)
end

# Check that gradient with forward and reverse mode doesn't change the RV values or logp of a VarInfo.
for backend ∈ (:forward_diff, :reverse_diff)
  let
      # Construct model.
      model, sampler, vi = foo_ad(), nothing, VarInfo()
      model(vi, sampler)

      # Record initial values.
      θ, logp = deepcopy(vi.vals), deepcopy(vi.logp)

      # Compute gradient using ForwardDiff.
      gradient(deepcopy(θ), vi, model, sampler; backend=backend)

      # Check that θ and logp haven't changed.
      @test θ == vi.vals
      @test logp == vi.logp
  end
end
