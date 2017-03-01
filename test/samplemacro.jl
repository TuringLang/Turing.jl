using Distributions
using Turing

@model gdemo(x) begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  for i = 1:length(x)
    x[i] ~ Normal(m, sqrt(s))
  end
  return(s, m, x)
end

x = [2.0, 3.0]

alg = Gibbs(10, HMC(1, 0.2, 3, :m), PG(10, 1, :s))
# NOTE: want to translate below to
#       chn = sample(gdemo, Dict(:x => x), alg)
chn = @sample2(gdemo(x), alg);

Turing.TURING[:modelex]

macro sample2(modelcall, alg)
  # println(typeof(modelcall))
  modelf = modelcall.args[1]
  modelt = eval(parse(string(modelf)))
  # println(1)
  psyms = modelcall.args[2:end]
  # println(psyms)
  data = Dict()
  for sym in psyms
    data[sym] = eval(sym)
  end
  # res = sample(modelt, data, eval(alg))
  # print(res)
  esc(:(sample($modelt, $data, $alg)))
end
