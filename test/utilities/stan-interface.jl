using CmdStan

@testset "stan-interface.jl" begin
    chn = sample(gdemo_default, 2000, 1000, false, 1,
        CmdStan.Adapt(), CmdStan.Hmc(stepsize_jitter=0))
    check_gdemo(chn)
end
