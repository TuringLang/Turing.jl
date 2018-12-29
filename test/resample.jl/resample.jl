# Test resample

using Turing
using Turing.Utilities: resample_systematic, resample_stratified, resample_multinomial, resample_residual, resample
using Test

num_samples = Int(1e6)

resSystematic = resample_systematic( [0.3, 0.4, 0.3], num_samples )
resStratified = resample_stratified( [0.3, 0.4, 0.3], num_samples )
resMultinomial= resample_multinomial( [0.3, 0.4, 0.3], num_samples )
resResidual   = resample_residual( [0.3, 0.4, 0.3], num_samples )
resample( [0.3, 0.4, 0.3])
resSystematic2=resample( [0.3, 0.4, 0.3], num_samples )

@test sum(resSystematic .== 2) ≈ (num_samples * 0.4) atol=1e-3*num_samples
@test sum(resSystematic2 .== 2) ≈ (num_samples * 0.4) atol=1e-3*num_samples
@test sum(resStratified .== 2) ≈ (num_samples * 0.4) atol=1e-3*num_samples
@test sum(resMultinomial .== 2) ≈ (num_samples * 0.4) atol=1e-2*num_samples
@test sum(resResidual .== 2) ≈ (num_samples * 0.4) atol=1e-2*num_samples
