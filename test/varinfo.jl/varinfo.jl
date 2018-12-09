using Turing, Test
using Turing: reconstruct, invlink, step
using Turing.VarReplay
using Turing.VarReplay: uid, cuid, getvals, getidcs, set_retained_vns_del_by_spl!, is_flagged, unset_flag!

randr(vi::VarInfo, vn::VarName, dist::Distribution, spl::Turing.Sampler, count::Bool) = begin
  if ~haskey(vi, vn)
    r = rand(dist)
    Turing.push!(vi, vn, r, dist, spl.alg.gid)
    r
  elseif is_flagged(vi, vn, "del")
    unset_flag!(vi, vn, "del")
    r = rand(dist)
    Turing.setval!(vi, Turing.vectorize(dist, r), vn)
    r
  else
    if count Turing.checkindex(vn, vi, spl) end
    Turing.updategid!(vi, vn, spl)
    vi[vn]
  end
end

# Test for uid() (= string())
csym = gensym()
vn1 = VarName(csym, :x, "[1]", 1)
@test string(vn1) == "{$csym,x[1]}:1"
# println(string(vn1))

vn2 = VarName(csym, :x, "[1]", 2)
vn11 = VarName(csym, :x, "[1]", 1)

@test cuid(vn1) == cuid(vn2)
@test vn11 == vn1


vi = VarInfo()
dists = [Normal(0, 1), MvNormal([0; 0], [1.0 0; 0 1.0]), Wishart(7, [1 0.5; 0.5 1])]

alg = PG(PG(5,5),2)
spl2 = Turing.Sampler(alg)
vn_w = VarName(gensym(), :w, "", 1)
randr(vi, vn_w, dists[1], spl2, true)

vn_x = VarName(gensym(), :x, "", 1)
vn_y = VarName(gensym(), :y, "", 1)
vn_z = VarName(gensym(), :z, "", 1)
vns = [vn_x, vn_y, vn_z]

alg = PG(PG(5,5),1)
spl1 = Turing.Sampler(alg)
for i = 1:3
  r = randr(vi, vns[i], dists[i], spl1, false)
  val = vi[vns[i]]
  @test sum(val - r) <= 1e-9
end

# println(vi)

@test length(getvals(vi, spl1)) == 3
@test length(getvals(vi, spl2)) == 1


vn_u = VarName(gensym(), :u, "", 1)
randr(vi, vn_u, dists[1], spl2, true)

# println(vi)
vi.num_produce = 1
set_retained_vns_del_by_spl!(vi, spl2)

# println(vi)

vals_of_1 = collect(getvals(vi, spl1))
# println(vals_of_1)
filter!(v -> ~any(map(x -> isnan.(x), v)), vals_of_1)
@test length(vals_of_1) == 3

vals_of_2 = collect(getvals(vi, spl2))
# println(vals_of_2)
filter!(v -> ~any(map(x -> isnan.(x), v)), vals_of_2)
@test length(vals_of_2) == 1

@model gdemo() = begin
  x ~ InverseGamma(2,3)
  y ~ InverseGamma(2,3)
  z ~ InverseGamma(2,3)
  w ~ InverseGamma(2,3)
  u ~ InverseGamma(2,3)
end

# Test the update of group IDs
g_demo_f = gdemo()
g = Turing.Sampler(g_demo_f, Gibbs(1000, PG(10, 2, :x, :y, :z), HMC(1, 0.4, 8, :w, :u)))

pg, hmc = g.info[:samplers]

vi = Turing.VarInfo()
g_demo_f(vi, nothing)
vi, _ = Turing.step(g_demo_f, pg, vi)
@test vi.gids == [1,1,1,0,0]

g_demo_f(vi, hmc)
@test vi.gids == [1,1,1,2,2]
