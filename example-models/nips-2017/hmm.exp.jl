include("hmm.model.jl")

TPATH = Pkg.dir("Turing")
using HDF5, JLD
const hmm_semisup_data = load(TPATH*"/example-models/nips-2017/hmm_semisup_data.jld")["data"]

N = 1000

collapsed = false

S = 4     # number of samplers
spls = [Gibbs(N,PG(50,1,:y),HMC(1,0.25,6,:phi,:theta)),
        Gibbs(N,PG(50,1,:y),HMCDA(1,200,0.65,0.75,:phi,:theta)),
        Gibbs(N,PG(50,1,:y),NUTS(1,200,0.65,:phi,:theta)),
        PG(50,N)][1:S]


spl_names = ["Gibbs($N,PG(50,1,:y),HMC(1,0.25,6,:phi,:theta))",
             "Gibbs($N,PG(50,1,:y),HMCDA(1,200,0.65,0.75,:phi,:theta))",
             "Gibbs($N,PG(50,1,:y),NUTS(1,200,0.65,:phi,:theta))",
             "PG(50,$N)"][1:S]
for i in 1:S
  println("$(spl_names[i]) running")
  #if i != 1 && i != 2 # i=1 already done
    chain = sample(hmm_semisup(data=hmm_semisup_data[1]), spls[i])
    # describe(chain)

    save(TPATH*"/example-models/nips-2017/hmm-uncollapsed-$(spl_names[i])-chain.jld", "chain", chain)
  #end
end

collapsed = true

S = 4     # number of samplers
spls = [HMC(N,0.25,6),HMCDA(N,200,0.65,0.75),NUTS(N,200,0.65),PG(50,N)][1:S]
spl_names = ["HMC($N,0.05,6)","HMCDA($N,200,0.65,0.35)","NUTS($N,200,0.65)","PG(50,$N)"][1:S]
for i in 1:S
  chain = sample(hmm_semisup(data=hmm_semisup_data[1]), spls[i])
  # describe(chain)

  save(TPATH*"/example-models/nips-2017/hmm-collapsed-$(spl_names[i])-chain.jld", "chain", chain)
end
