using Turing: VarInfo
using Base.Test

v_sym = VarInfo(:x)
@test v_sym.id == :x

v_arr = VarInfo(:(x[i]), :i, 1)
@test v_arr.id == Symbol(:(x[1]))

# TODO: implement below
# v_mat = VarInfo(:(x[i,j]), :i, 1, :j, 2)
# @test v_mat.id == :(x[1,2])
#
# v_arrarr = VarInfo(:(x[i][j]), :i, 1, :j, 2)
# @test v_arrarr.id == :(x[1][2])
