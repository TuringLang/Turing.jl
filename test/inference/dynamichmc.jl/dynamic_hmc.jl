using DynamicHMC, Turing, Test

chn = sample(gdemo_default, DynamicNUTS(2000));

check_numerical(chn, [:s, :m], [49/24, 7/6], eps=0.2)
