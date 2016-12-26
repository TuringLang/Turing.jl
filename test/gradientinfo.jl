# Test GradientInfo

using Turing, Base.Test

pc = GradientInfo()
p1 = VarInfo(gensym())
p2 = VarInfo(gensym())

pc.values[p1] = 1
pc.values[p1] = 2
pc.values[p1] = 3
pc.values[p2] = 4

@test pc[p1] == 3
@test pc[p2] == 4

pc[p1] = 5
pc[p1] = 6
pc[p1] = 7

@test pc[p1] == 7

keys(pc)
