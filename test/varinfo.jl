using Turing, Base.Test
using Turing: uid, cuid

# Test for uid() (= string())
csym = gensym()
vn1 = VarName(csym, :x, "[1]", 1)
@test string(vn1) == "{$csym,x[1]}:1"
# println(string(vn1))

vn2 = VarName(csym, :x, "[1]", 2)
vn11 = VarName(csym, :x, "[1]", 1)

@test cuid(vn1) == cuid(vn2)
@test vn11 == vn1
