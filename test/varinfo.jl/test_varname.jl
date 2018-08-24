using Distributions
using ForwardDiff: Dual
using Turing
using Turing: varname
using Test

# Symbol
v_sym = string(:x)
@test v_sym == "x"

# Array
i = 1
v_arr = eval(varname(:(x[i]))[1])
@test v_arr == "[1]"

# Matrix
i, j, k = 1, 2, 3
v_mat = eval(varname(:(x[i,j]))[1])
@test v_mat== "[1,2]"

v_mat = eval(varname(:(x[i,j,k]))[1])
@test v_mat== "[1,2,3]"

v_mat = eval(varname(:((x[1,2][1+5][45][3][i])))[1])
@test v_mat == "[1,2][6][45][3][1]"


@model mat_name_test() = begin
  p = Array{Any}(undef, 2, 2)
  for i in 1:2, j in 1:2
    p[i,j] ~ Normal(0, 1)
  end
  p
end
chain = sample(mat_name_test(), HMC(1000, 0.2, 4))

@test mean(chain[Symbol("p[1, 1]")]) ≈ 0 atol=0.25

# Multi array
i, j = 1, 2
v_arrarr = eval(varname(:(x[i][j]))[1])
@test v_arrarr == "[1][2]"

@model marr_name_test() = begin
  p = Array{Array{Any}}(undef, 2)
  p[1] = Array{Any}(undef, 2)
  p[2] = Array{Any}(undef, 2)
  for i in 1:2, j in 1:2
    p[i][j] ~ Normal(0, 1)
  end
  p
end
chain = sample(marr_name_test(), HMC(1000, 0.2, 4))
@test mean(chain[Symbol("p[1][1]")]) ≈ 0 atol=0.25
