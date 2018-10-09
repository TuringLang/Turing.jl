# Test resample

using Turing
using Test

num_samples = Int(1e6)
resSystematic = Turing.resampleSystematic( [0.3, 0.4, 0.3], num_samples )
@test sum(resSystematic .== 2) ≈ (num_samples * 0.4) atol=1e-3*num_samples

