using Turing
using HDF5, JLD

TPATH = Pkg.dir("Turing")

# chain = load("/home/kai/bak/sv-gibbs-pg-hmc.jld")["chain"]
chain = load("/home/kai/bak/sv-gibbs-pg-nuts.jld")["chain"]

include(TPATH*"/example-models/nuts-paper/sv_helper.jl")

y = readsvdata()
y = filter!(each -> each > 0.1, y)
N = length(y)
logy = log(y)

using Gadfly
N=length(y)
l1 = layer(x=1:N,y=y,Geom.line)
plot(l1)

s_1 = chain[:s1][500]
s_2_to_2519 = [chain[Symbol("logs[$i]")][500] for i = 2:2519]

s_1 = s[331]

l2 = layer(x=1:N,y=map(e->exp(e),s_1),Geom.point)

plot(l2)

lps = chain[:lp]
N = length(lps)
p_lps = plot(x=2:N,y=-lps[2:N],Geom.line)

draw(PNG(TPATH*"/nips-2017/sv/old/sv-gibbs-pg-nuts-lps.png", 8inch, 4.5inch), p_lps)
