###################
# Helper function #
###################

insdelim(c, deli=",") = reduce((e, res) -> append!(e, [res, ","]), [], c)[1:end-1]

function varname(expr)
  # Initialize an expression block to store the code for creating uid
  local sym
  indexing_ex = Expr(:block)
  # Add the initialization statement for uid
  push!(indexing_ex.args, quote indexing_list = [] end)
  # Initialize a local container for parsing and add the expr to it
  to_eval = []; unshift!(to_eval, expr)
  # Parse the expression and creating the code for creating uid
  find_head = false
  while length(to_eval) > 0
    evaling = shift!(to_eval)   # get the current expression to deal with
    if isa(evaling, Expr) && evaling.head == :ref && ~find_head
      # Add all the indexing arguments to the left
      unshift!(to_eval, "[", insdelim(evaling.args[2:end])..., "]")
      # Add first argument depending on its type
      # If it is an expression, it means it's a nested array calling
      # Otherwise it's the symbol for the calling
      if isa(evaling.args[1], Expr)
        unshift!(to_eval, evaling.args[1])
      else
        # push!(indexing_ex.args, quote unshift!(indexing_list, $(string(evaling.args[1]))) end)
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

#################
# Overload of ~ #
#################

macro ~(left, right)
  # Is multivariate a subtype of real, e.g. Vector, Matrix?
  if isa(left, Real)                  # value
    # Call observe
    esc(
      quote
        Turing.observe(
          sampler,
          $(right),   # Distribution
          $(left),    # Data point
          vi
        )
      end
    )
  else
    _left = left  # left is the variable (symbol) itself
    # Get symbol from expr like x[1][2]
    while typeof(_left) != Symbol
      _left = _left.args[1]
    end
    left_sym = string(_left)
    # Require all data to be stored in data dictionary.
    if _left in Turing._compiler_[:fargs]
      if ~(_left in Turing._compiler_[:dvars])
        println("[Turing]: Observe - `" * left_sym * "` is an observation")
        push!(Turing._compiler_[:dvars], _left)
      end
      esc(
        quote
          # Call observe
          Turing.observe(
            sampler,
            $(right),   # Distribution
            $(left),    # Data point
            vi
          )
        end
      )
    else
      if ~(_left in Turing._compiler_[:pvars])
        msg = "[Turing]: Assume - `" * left_sym * "` is a parameter"
        isdefined(Symbol(left_sym)) && (msg  *= " (ignoring `$(left_sym)` found in global scope)")
        println(msg)
        push!(Turing._compiler_[:pvars], _left)
      end
      # The if statement is to deterimnet how to pass the prior.
      # It only supports pure symbol and Array(/Dict) now.
      #csym_str = string(gensym())
      if isa(left, Symbol)
        # Symbol
        assume_ex = quote
          csym_str = string(Turing._compiler_[:fname])*"_var"* string(@__LINE__)
          if isa(sampler, Union{Sampler{PG},Sampler{SMC}})
            vi = Turing.current_trace().vi
          end
          sym = Symbol($(string(left)))
          vn = nextvn(vi, Symbol(csym_str), sym, "")
          $(left) = Turing.assume(
            sampler,
            $(right),   # dist
            vn,         # VarName
            vi          # VarInfo
          )
        end
      else
        # Indexing
        assume_ex, sym = varname(left)    # sym here is used in predict()
        # NOTE:
        # The initialization of assume_ex is indexing_ex,
        # in which sym will store the variable symbol (Symbol),
        # and indexing will store the indexing (String)
        # csym_str = string(gensym())
        push!(
          assume_ex.args,
          quote
            csym_str = string(Turing._compiler_[:fname]) * string(@__LINE__)
            if isa(sampler, Union{Sampler{PG},Sampler{SMC}})
              vi = Turing.current_trace().vi
            end
            vn = nextvn(vi, Symbol(csym_str), sym, indexing)
            $(left) = Turing.assume(
              sampler,
              $(right),   # dist
              vn,         # VarName
              vi          # VarInfo
            )
          end
        )
      end
      esc(assume_ex)
    end
  end
end

#################
# Main Compiler #
#################

doc"""
    @model(name, fbody)

Wrapper for models.

Usage:

```julia
@model model() = begin
  # body
end
```

Example:

```julia
@model gauss() = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  1.5 ~ Normal(m, sqrt(s))
  2.0 ~ Normal(m, sqrt(s))
  return(s, m)
end
```
"""
macro model(fexpr)
   # compiler design: sample(fname_compiletime(x,y), sampler)
   #   fname_compiletime(x,y;fname=fname,fargs=fargs,fbody=fbody) = begin
   #      ex = quote
   #          fname_runtime(x,y,fobs) = begin
   #              x=x,y=y,fobs=Set(:x,:y)
   #              fname(vi=VarInfo,sampler=nothing) = begin
   #              end
   #          end
   #          fname_runtime(x,y,fobs)
   #      end
   #      Main.eval(ex)
   #   end
   #   fname_compiletime(;data::Dict{Symbol,Any}=data) = begin
   #      ex = quote
   #          # check fargs[2:end] == symbols(data)
   #          fname_runtime(x,y,fobs) = begin
   #          end
   #          fname_runtime(data[:x],data[:y],Set(:x,:y))
   #      end
   #      Main.eval(ex)
   #   end


  dprintln(1, fexpr)

  fname = fexpr.args[1].args[1]      # Get model name f
  fargs = fexpr.args[1].args[2:end]  # Get model parameters (x,y;z=..)
  fbody = fexpr.args[2].args[end]    # NOTE: nested args is used here because the orignal model expr is in a block


  # Add keyword arguments, e.g.
  #   f(x,y,z; c=1)
  #       ==>  f(x,y,z; c=1, vi = VarInfo(), sampler = IS(1))
  if (length(fargs) == 0 ||         # e.g. f()
          isa(fargs[1], Symbol) ||  # e.g. f(x,y)
          fargs[1].head == :kw)     # e.g. f(x,y=1)
    insert!(fargs, 1, Expr(:parameters))
  # else                  # e.g. f(x,y; k=1)
  #  do nothing;
  end

  dprintln(1, fname)
  dprintln(1, fargs)
  dprintln(1, fbody)

  # Remove positional arguments from inner function, e.g.
  #  f(y,z,w; vi=VarInfo(),sampler=IS(1))
  #      ==>   f(; vi=VarInfo(),sampler=IS(1))
  fargs_inner = deepcopy(fargs)[1:1]
  push!(fargs_inner[1].args, Expr(:kw, :vi, :(VarInfo())))
  push!(fargs_inner[1].args, Expr(:kw, :sampler, :(nothing)))
  dprintln(1, fargs_inner)


  # Modify fbody, so that we always return VarInfo
  fbody2 = deepcopy(fbody)
  return_ex = fbody.args[end]   # get last statement of defined model
  if typeof(return_ex) == Symbol ||
       return_ex.head == :return ||
       return_ex.head == :tuple
    pop!(fbody2.args)
  end
  push!(fbody2.args, Expr(:return, :vi))
  dprintln(1, fbody2)

  ## Create function definition
  fdefn = Expr(:function, Expr(:call, fname)) # fdefn = :( $fname() )
  push!(fdefn.args[1].args, fargs_inner...)   # Set parameters (x,y;data..)
  push!(fdefn.args, deepcopy(fbody2))         # Set function definition
  dprintln(1, fdefn)

  fdefn2 = Expr(:function, Expr(:call, Symbol("$(fname)_model")))
  push!(fdefn2.args[1].args, fargs_inner...)   # Set parameters (x,y;data..)
  push!(fdefn2.args, deepcopy(fbody2))    # Set function definition
  dprintln(1, fdefn2)


  compiler = Dict(:fname => fname,
                  :fargs => fargs,
                  :fbody => fbody,
                  :dvars => Set{Symbol}(), # Data
                  :pvars => Set{Symbol}(), # Parameter
                  :fdefn2 => fdefn2)

  # outer function def 1: f(x,y) ==> f(x,y;data=Dict())
  fargs_outer1 = deepcopy(fargs)
  # Add data argument to outer function
  push!(fargs_outer1[1].args, Expr(:kw, :data, :(Dict{Symbol,Any}())))
  # Add data argument to outer function
  push!(fargs_outer1[1].args, Expr(:kw, :compiler, compiler))
  for i = 2:length(fargs_outer1)
    s = fargs_outer1[i]
    if isa(s, Symbol)
      fargs_outer1[i] = Expr(:kw, s, :nothing) # Turn f(x;..) into f(x=nothing;..)
    end
  end

  # outer function def 2: f(x,y) ==> f(;data=Dict())
  fargs_outer2 = deepcopy(fargs)[1:1]
  # Add data argument to outer function
  push!(fargs_outer2[1].args, Expr(:kw, :data, :(Dict{Symbol,Any}())))

  ex = Expr(:function, Expr(:call, fname, fargs_outer1...),
                        Expr(:block, Expr(:return, Symbol("$(fname)_model"))))

  unshift!(ex.args[2].args, :(Main.eval(fdefn2)))
  unshift!(ex.args[2].args,  quote
      # check fargs, data
      eval(Turing, :(_compiler_ = deepcopy($compiler)))
      fargs    = Turing._compiler_[:fargs];
      fdefn2   = Turing._compiler_[:fdefn2];
      # copy (x,y,z) to fbody
      # for k in fargs
      #   if isa(k, Symbol)       # f(x,..)
      #     if haskey(data, k)
      #       warn("[Turing]: parameter $k found twice, value in data dictionary will be used.")
      #       ex = nothing
      #     else
      #       ex = Expr(:(=), k, k)  # NOTE: need to get k's value
      #     end
      #   elseif k.head == :kw    # f(z=1,..)
      #     if haskey(data, k.args[1])
      #       warn("[Turing]: parameter $(k.args[1]) found twice, value in data dictionary will be used.")
      #       ex = nothing
      #     else
      #       ex = Expr(:(=), k.args[1], k.args[1])
      #     end
      #   else
      #     ex = nothing
      #   end
      #   if ex != nothing
      #     if fdefn2.args[2].args[1].head == :line
      #       # Preserve comments, useful for debuggers to correctly
      #       #   locate source code oringin.
      #       insert!(fdefn2.args[2].args, 2, ex)
      #     else
      #       insert!(fdefn2.args[2].args, 1, ex)
      #     end
      #   end
      # end
      # copy data dictionary
      for k in keys(data)
        if fdefn2.args[2].args[1].head == :line
          # Preserve comments, useful for debuggers to correctly
          #   locate source code oringin.
          insert!(fdefn2.args[2].args, 2, Expr(:(=), k, data[k]))
        else
          insert!(fdefn2.args[2].args, 1, Expr(:(=), k, data[k]))
        end
      end
      dprintln(1, fdefn2)
  end )
  # unshift!(ex.args[2].args, :(println(compiler)))

  # for k in fargs
  #   if isa(k, Symbol)       # f(x,..)
  #     _ = Expr(:(=), :(data[$(k)]), k)  # NOTE: need to get k's value
  #   elseif k.head == :kw    # f(z=1,..)
  #     _ = Expr(:(=), :(data[$(k.args[1])]), k.args[1])
  #   else
  #     _ = nothing
  #   end
  #   _ != nothing && unshift!(ex.args[2].args, _)
  # end

  for k in fargs
    if isa(k, Symbol)       # f(x,..)
      _k = k
    elseif k.head == :kw    # f(z=1,..)
      _k = k.args[1]
    else
      _k = nothing
    end
    if _k != nothing
      _k_str = string(_k)
      _ = quote
            if haskey(data, Symbol($_k_str))
              warn("[Turing]: parameter "*$_k_str*" found twice, value in data dictionary will be used.")
            else
              data[Symbol($_k_str)] = $_k
            end
          end
      unshift!(ex.args[2].args, _)
    end
  end
  unshift!(ex.args[2].args, quote data = copy(data) end)

  dprintln(1, esc(ex))
  esc(ex)
end
