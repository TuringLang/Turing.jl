using Distributions
using ForwardDiff: Dual
using Turing
using Turing: VarInfo
using Base.Test

# Symbol
v_sym = VarInfo(:x)
@test v_sym.id == :x

# Array
v_arr = VarInfo(:(x[i]), :i, 1)
@test v_arr.id == Symbol("x[1]")

# Matrix
v_mat = VarInfo(:(x[i,j]), :i, 1, :j, 2)
@test v_mat.id == Symbol("x[1,2]")

@model mat_name_test begin
  p = Array{Dual}((2, 2))
  for i in 1:2, j in 1:2
    p[i,j] ~ Normal(0, 1)
  end
  @predict p
end
chain = sample(mat_name_test, HMC(2500, 0.75, 5))
@test_approx_eq_eps mean(mean(chain[:p])) 0 5e-2

# Multi array
v_arrarr = VarInfo(:(x[i][j]), :i, 1, :j, 2)
@test v_arrarr.id == Symbol("x[1][2]")

@model marr_name_test begin
  p = Array{Array{Dual}}(2)
  p[1] = Array{Dual}(2)
  p[2] = Array{Dual}(2)
  for i in 1:2, j in 1:2
    p[i][j] ~ Normal(0, 1)
  end
  @predict p
end
chain = sample(marr_name_test, HMC(2500, 0.75, 5))
@test_approx_eq_eps mean(mean(mean(chain[:p]))) 0 5e-2
