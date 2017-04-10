using Turing, Base.Test
using Turing: uid, cuid, reconstruct, invlink, groupvals, retain, randr
using Turing: GibbsSampler

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

vn_w = VarName(gensym(), :w, "", 1)
randr(vi, vn_w, dists[1], 2, nothing, true)

vn_x = VarName(gensym(), :x, "", 1)
vn_y = VarName(gensym(), :y, "", 1)
vn_z = VarName(gensym(), :z, "", 1)
vns = [vn_x, vn_y, vn_z]

for i = 1:3
  r = randr(vi, vns[i], dists[i], 1, nothing, false)
  val = vi[vns[i]]
  @test sum(val - r) <= 1e-9
end

# println(vi)

@test length(groupvals(vi, 1)) == 3
@test length(groupvals(vi, 2)) == 1

vn_u = VarName(gensym(), :u, "", 1)
randr(vi, vn_u, dists[1], 2, nothing, true)

# println(vi)

retain(vi, 2, 1)

# println(vi)

@test length(groupvals(vi, 1)) == 3
@test length(groupvals(vi, 2)) == 1

@model gdemo() = begin
  x ~ InverseGamma(2,3)
  y ~ InverseGamma(2,3)
  z ~ InverseGamma(2,3)
  w ~ InverseGamma(2,3)
  u ~ InverseGamma(2,3)
end

# println("Test 2")

g = GibbsSampler{Gibbs}(gdemo(), Gibbs(1000, PG(10, 2, :x, :y, :z), HMC(1, 0.4, 8, :w, :u)))

pg = g.samplers[1]
println(pg)
hmc = g.samplers[2]
dist= Normal(0, 1)

vi = VarInfo()

r = rand(vi, vn_w, dist, pg, false)
r = rand(vi, vn_u, dist, pg, false)
r = rand(vi, vn_x, dist, pg)
r = rand(vi, vn_y, dist, pg)
r = rand(vi, vn_z, dist, pg)

@test vi.gids == [0,0,1,1,1]

# println(vi)

# println(vi)

r = rand(vi, vn_w, dist, hmc)
r = rand(vi, vn_u, dist, hmc)
r = rand(vi, vn_x, dist, hmc, false)
r = rand(vi, vn_y, dist, hmc, false)
r = rand(vi, vn_z, dist, hmc, false)

@test vi.gids == [2,2,1,1,1]
