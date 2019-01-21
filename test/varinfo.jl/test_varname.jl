using ForwardDiff: Dual
using Turing
using Test


varname(s::Symbol)  = nothing, s
function varname(expr::Expr)
    # Initialize an expression block to store the code for creating uid
    local sym
    @assert expr.head == :ref "expr needs to be an indexing expression, e.g. :(x[1])"
    indexing_ex = Expr(:block)
    # Add the initialization statement for uid
    push!(indexing_ex.args, quote indexing_list = [] end)
    # Initialize a local container for parsing and add the expr to it
    to_eval = []; pushfirst!(to_eval, expr)
    # Parse the expression and creating the code for creating uid
    find_head = false
    while length(to_eval) > 0
        evaling = popfirst!(to_eval)   # get the current expression to deal with
        if isa(evaling, Expr) && evaling.head == :ref && ~find_head
            # Add all the indexing arguments to the left
            pushfirst!(to_eval, "[", insdelim(evaling.args[2:end])..., "]")
            # Add first argument depending on its type
            # If it is an expression, it means it's a nested array calling
            # Otherwise it's the symbol for the calling
            if isa(evaling.args[1], Expr)
                pushfirst!(to_eval, evaling.args[1])
            else
                # push!(indexing_ex.args, quote pushfirst!(indexing_list, $(string(evaling.args[1]))) end)
                push!(indexing_ex.args, quote sym = Symbol($(string(evaling.args[1]))) end) # store symbol in runtime
                find_head = true
                sym = evaling.args[1] # store symbol in compilation time
            end
        else
            # Evaluting the concrete value of the indexing variable
            push!(indexing_ex.args, quote push!(indexing_list, string($evaling)) end)
        end
    end
    push!(indexing_ex.args, quote indexing = reduce(*, indexing_list) end)
    indexing_ex, sym
end


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

