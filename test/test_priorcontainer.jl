using Turing, Base.Test

pa = PriorArray()
pa.add(1)
pa.add(2)
@test pa.get() == 1
@test pa.get() == 2
@test pa.get() == 1
pa.set(3)
pa.set(4)
pa.set(5)
@test pa.get() == 4
@test pa.get() == 5


pc = PriorContainer()
p1 = Prior(gensym())
p2 = Prior(gensym())

pc.addPrior(p1, 1)
pc.addPrior(p1, 2)
pc.addPrior(p1, 3)
pc.addPrior(p2, 4)

@test pc[p1] == 1
@test pc[p1] == 2
@test pc[p1] == 3
@test pc[p1] == 1
@test pc[p1] == 2
@test pc[p1] == 3

@test pc[p2] == 4

pc[p1] = 5
pc[p1] = 6
pc[p1] = 7

@test pc[p1] == 5
@test pc[p1] == 6
@test pc[p1] == 7

keys(pc)
