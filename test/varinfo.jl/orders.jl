using Turing, Test
using Utilities: reconstruct
using Bijectors: invlink
using Turing.Inference: step
using Turing: CACHERESET
using Turing.Core.VarReplay
using Turing.Core.VarReplay: uid, cuid, getvals, getidcs, set_retained_vns_del_by_spl!, is_flagged, unset_flag!


# Mock assume method for CSMC cf src/samplers/pgibbs.jl
randr(vi::UntypedVarInfo, vn::VarName, dist::Distribution, spl::Turing.Sampler) = begin
  if ~haskey(vi, vn)
    r = rand(dist)
    Turing.push!(vi, vn, r, dist, spl.alg.gid)
    spl.info[:cache_updated] = CACHERESET
    r
  elseif is_flagged(vi, vn, "del")
    unset_flag!(vi, vn, "del")
    r = rand(dist)
    vi[vn] = Turing.vectorize(dist, r)
    Turing.setorder!(vi, vn, vi.num_produce)
    r
  else
    Turing.updategid!(vi, vn, spl)
    vi[vn]
  end
end

csym = gensym() # unique per model
vn_z1 = VarName(csym, :z, "[1]", 1)
vn_z2 = VarName(csym, :z, "[2]", 1)
vn_z3 = VarName(csym, :z, "[3]", 1)
vn_z4 = VarName(csym, :z, "[4]", 1)
vn_a1 = VarName(csym, :a, "[1]", 1)
vn_a2 = VarName(csym, :a, "[2]", 1)
vn_b = VarName(csym, :b, "", 1)

vi = VarInfo()
dists = [Categorical([0.7, 0.3]), Normal()]

alg1 = PG(PG(5,5),1)
spl1 = Turing.Sampler(alg1)
alg2 = PG(PG(5,5),2)
spl2 = Turing.Sampler(alg2)

# First iteration, variables are added to vi
# variables samples in order: z1,a1,z2,a2,z3
vi.num_produce += 1
randr(vi, vn_z1, dists[1], spl1)
randr(vi, vn_a1, dists[2], spl1)
vi.num_produce += 1
randr(vi, vn_b, dists[2], spl2)
randr(vi, vn_z2, dists[1], spl1)
randr(vi, vn_a2, dists[2], spl1)
vi.num_produce += 1
randr(vi, vn_z3, dists[1], spl1)
@test vi.orders == [1, 1, 2, 2, 2, 3]
@test vi.num_produce == 3

# Check getretain at different stages
# variables samples in different order: z1,a1,z2,z3,a2
vi.num_produce = 0
set_retained_vns_del_by_spl!(vi, spl1)

vi.num_produce += 1
randr(vi, vn_z1, dists[1], spl1)
randr(vi, vn_a1, dists[2], spl1)
vi.num_produce += 1
randr(vi, vn_z2, dists[1], spl1)
vi.num_produce += 1
randr(vi, vn_z3, dists[1], spl1)
randr(vi, vn_a2, dists[2], spl1)
@test vi.orders == [1, 1, 2, 2, 3, 3]
@test vi.num_produce == 3
