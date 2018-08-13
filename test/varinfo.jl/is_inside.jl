using Turing: reconstruct, invlink, step, varname
using Turing.VarReplay
using Turing.VarReplay: is_inside

using Test

genvn(sym::Symbol) = VarName(gensym(), sym, "", 1)
genvn(expr::Expr) = begin
  ex, sym = varname(expr)
  VarName(gensym(), sym, eval(ex), 1)
end



space = Set([:x, :y, :(z[1])])
vn1 = genvn(:x)
vn2 = genvn(:y)
vn3 = genvn(:(x[1]))
vn4 = genvn(:(z[1][1]))
vn5 = genvn(:(z[2]))
vn6 = genvn(:z)

@test is_inside(vn1, space)
@test is_inside(vn2, space)
@test is_inside(vn3, space)
@test is_inside(vn4, space)
@test ~is_inside(vn5, space)
@test ~is_inside(vn6, space)
