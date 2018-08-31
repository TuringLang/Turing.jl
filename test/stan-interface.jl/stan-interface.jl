@init @require Stan="682df890-35be-576f-97d0-3d8c8b33a550" begin

  using Turing, Stan

  @model gdemo(x) = begin
    s ~ InverseGamma(2,3)
    m ~ Normal(0,sqrt.(s))
    x[1] ~ Normal(m, sqrt.(s))
    x[2] ~ Normal(m, sqrt.(s))
    return s, m
  end

  model_f = gdemo([1.5, 2.0])

  chn = sample(model_f, 2000, 1000, false, 1, Stan.Adapt(), Stan.Hmc())

end
