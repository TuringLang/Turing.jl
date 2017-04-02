using Turing, Base.Test
using Turing: uid, cuid, reconstruct, invlink

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
vn_x = VarName(gensym(), :x, "", 1)
vn_y = VarName(gensym(), :y, "", 1)
vn_z = VarName(gensym(), :z, "", 1)
vns = [vn_x, vn_y, vn_z]
dists = [Normal(0, 1), MvNormal([0; 0], [1.0 0; 0 1.0]), Wishart(7, [1 0.5; 0.5 1])]
for i = 1:3
  r = randrn(vi, vns[i], dists[i])
  val = vi[vns[i]]
  val = reconstruct(dists[i], val)
  val = invlink(dists[i], val)
  @test sum(realpart(val) - r) <= 1e-9
end
