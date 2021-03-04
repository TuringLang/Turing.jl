@stage_testset "stan" "stan-interface.jl" begin
    using CmdStan

    chn = sample(gdemo_default, 2000, 1000, false, 1,
        CmdStan.Adapt(), CmdStan.Hmc(stepsize_jitter=0))
    check_gdemo(chn)
end
