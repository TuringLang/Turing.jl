using Turing.Core.VarReplay: VarName, is_inside
using Test

include("../utility.jl")

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
    return indexing_ex, sym
end


genvn(sym::Symbol) = VarName(gensym(), sym, "", 1)
function genvn(expr::Expr)
    ex, sym = varname(expr)
    return VarName(gensym(), sym, eval(ex), 1)
end



space = (:x, :y, :(z[1]))
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
