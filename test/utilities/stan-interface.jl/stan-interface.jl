using CmdStan, Turing

chn = sample(gdemo_default, 2000, 1000, false, 1, CmdStan.Adapt(), CmdStan.Hmc())
