using Turing
using Distributions
using Base.Test

Profile.clear()

p() = begin

  include("data.jl")
  include("model.jl")

  # Produce an error.
  @sample(negbinmodel(negbindata), HMC(1000, 0.02, 1));
end

@profile p()

f = open("profile_res.txt","w")
Profile.print(f)
