using Turing
using HDF5, JLD

TPATH = Pkg.dir("Turing")

chain = load(TPATH*"/nips-2017/sv/sv-exps-Gibbs(1000,PG(50,1),NUTS(1,200,0.65))-chain.jld")["chain"]

include(TPATH*"/example-models/nuts-paper/sv_helper.jl")

y = readsvdata()

using Gadfly
N=length(y)
l1 = layer(x=1:N,y=y,Geom.point)

s = chain[:s]

s_1 = s[331]

l2 = layer(x=1:N,y=map(e->exp(e),s_1),Geom.point)

plot(l2)
