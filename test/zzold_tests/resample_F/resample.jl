# Test resample

using Turing
using Test

num_samples = Int(1e6)

resSystematic = Turing.resample_systematic( [0.3, 0.4, 0.3], num_samples )
resStratified = Turing.resample_stratified( [0.3, 0.4, 0.3], num_samples )
resMultinomial= Turing.resample_multinomial( [0.3, 0.4, 0.3], num_samples )
resResidual   = Turing.resample_residual( [0.3, 0.4, 0.3], num_samples )
Turing.resample( [0.3, 0.4, 0.3])
resSystematic2=Turing.resample( [0.3, 0.4, 0.3], num_samples )

@test sum(resSystematic .== 2) ≈ (num_samples * 0.4) atol=1e-3*num_samples
@test sum(resSystematic2 .== 2) ≈ (num_samples * 0.4) atol=1e-3*num_samples
@test sum(resStratified .== 2) ≈ (num_samples * 0.4) atol=1e-3*num_samples
@test sum(resMultinomial .== 2) ≈ (num_samples * 0.4) atol=1e-2*num_samples
@test sum(resResidual .== 2) ≈ (num_samples * 0.4) atol=1e-2*num_samples
