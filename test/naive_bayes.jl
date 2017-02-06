# Turing.jl version of model at https://github.com/stan-dev/example-models/blob/master/misc/cluster/naive-bayes/naive-bayes.stan

using Distributions
using Turing
using Base.Test

include("naive_bayes_data.jl")

@model naive_bayes begin
  theta ~ Dirichlet(alpha)
  phi = Array{Any}(K)
  for k = 1:K
    phi[k] ~ Dirichlet(β)
  end
  for m = 1:M
    z[m] ~ Categorical(theta)
  end
  for n = 1:N
    w[n] ~ Categorical(phi[z[doc[n]]])
  end
  phi
end

chain = sample(naive_bayes, data, HMC(1000, 0.1, 3))
print(mean([[realpart(n) for n in ns] for ns in chain[:phi]]))


# This example sometimes gives non-determinstic error:
# ERROR: LoadError: DomainError:
#  in nan_dom_err at ./math.jl:196 [inlined]
#  in log(::Float64) at ./math.jl:202
#  in #31 at ./<missing>:0 [inlined]
#  in next at ./generator.jl:26 [inlined]
#  in collect_to!(::Array{Float64,1}, ::Base.Generator{UnitRange{Int64},Turing.##31#32{Int64,Array{Float64,1}}}, ::Int64, ::Int64) at ./array.jl:340
#  in collect(::Base.Generator{UnitRange{Int64},Turing.##31#32{Int64,Array{Float64,1}}}) at ./array.jl:308
#  in link(::Distributions.Dirichlet{Float64}, ::Array{Float64,1}) at /home/kai/.julia/v0.5/Turing/src/samplers/support/transform.jl:110
#  in assume(::Turing.HMCSampler{Turing.HMC}, ::Distributions.Dirichlet{Float64}, ::Turing.Var, ::Turing.VarInfo) at /home/kai/.julia/v0.5/Turing/src/samplers/hmc.jl:133
#  in macro expansion at /home/kai/.julia/v0.5/Turing/src/core/compiler.jl:45 [inlined]
#  in macro expansion at /home/kai/projects/Turing.jl/test/naive_bayes.jl:13 [inlined]
#  in naive_bayes(::Dict{Symbol,Any}, ::Turing.VarInfo, ::Turing.HMCSampler{Turing.HMC}) at /home/kai/.julia/v0.5/Turing/src/core/compiler.jl:12
#  in step(::#naive_bayes, ::Dict{Symbol,Any}, ::Turing.HMCSampler{Turing.HMC}, ::Turing.VarInfo, ::Bool) at /home/kai/.julia/v0.5/Turing/src/samplers/hmc.jl:59
#  in run(::Function, ::Dict{Symbol,Any}, ::Turing.HMCSampler{Turing.HMC}) at /home/kai/.julia/v0.5/Turing/src/samplers/hmc.jl:109
#  in sample(::Function, ::Dict{Symbol,Any}, ::Turing.HMC) at /home/kai/.julia/v0.5/Turing/src/samplers/hmc.jl:178
#  in include_from_node1(::String) at ./loading.jl:488
#  in process_options(::Base.JLOptions) at ./client.jl:262
#  in _start() at ./client.jl:318
# while loading /home/kai/projects/Turing.jl/test/naive_bayes.jl, in expression starting on line 24
