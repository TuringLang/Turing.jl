# Test resample

using Turing
using Distributions
using Test

num_samples = Int(1e6)

resSystematic = Turing.resampleSystematic( [0.3, 0.4, 0.3], num_samples )
resStratified = Turing.resampleStratified( [0.3, 0.4, 0.3], num_samples )
resMultinomial= Turing.resampleMultinomial( [0.3, 0.4, 0.3], num_samples )
resResidual   = Turing.resampleResidual( [0.3, 0.4, 0.3], num_samples )
Turing.resample( [0.3, 0.4, 0.3])
resSystematic2=Turing.resample( [0.3, 0.4, 0.3], num_samples )

@test sum(resSystematic .== 2) ≈ (num_samples * 0.4) atol=1e-3*num_samples
@test sum(resSystematic2 .== 2) ≈ (num_samples * 0.4) atol=1e-3*num_samples
@test sum(resStratified .== 2) ≈ (num_samples * 0.4) atol=1e-3*num_samples
@test sum(resMultinomial .== 2) ≈ (num_samples * 0.4) atol=1e-2*num_samples
@test sum(resResidual .== 2) ≈ (num_samples * 0.4) atol=1e-2*num_samples
