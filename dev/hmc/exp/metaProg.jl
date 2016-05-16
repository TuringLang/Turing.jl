# Every julia prog is a string
prog = "1 + 1"

# , and can be parsed into an expression
ex1 = parse(prog)
typeof(ex1)

# Expr objs have three parts
ex1.head  # type of exp
ex1.args  # arguments
ex1.typ   # return type

# Expressions can aslo be created by construction
ex2 = Expr(:call, :+, 1, 1)
ex1 == ex2

# Julia code is internally represented as a data structure that is accessible from the language itself.

# The dump() function provides indented and annotated display of Expr objects.
dump(ex2)

# Expr objects may also be nested.
ex3 = parse("(4 + 4) / 2")
dump(ex3)

# Another way to view expressions is with Meta.show_sexpr, which displays the S-expression form of a given Expr, which may look very familiar to users of Lisp.
Meta.show_sexpr(ex3)

# A Symbol is an intern string used as one building-block of expressions.
s1 = :foo
typeof(s1)

# Symbols can aslo be created using symbol()
:foo == symbol("foo")
s2 = symbol("func", 10)
s3 = symbol(:var, '_', "sym")
:var_sym == s3

# Sometimes extra parentheses around the argument to : are needed to avoid ambiguity in parsing.
:(:)
typeof(:(:))
:(::)
typeof(:(::))

# Quoting means using : character to create expression objects.
ex = :(a + b * c + 1)
typeof(ex)
ex.head
ex.args
ex.typ
dump(ex)

# Equivalent expressions may be constructed using parse() or the direct Expr form.
:(a + b * c + 1) == parse(" a + b * c + 1") == Expr(:call, :+, :a, Expr(:call, :*, :b, :c), 1)

# A second syntactic form of quoting for multiple expressions is
# blocks of code enclosed in quote ... end.
ex = quote
  x = 1
  y = 2
  x + y
end
typeof(ex)

# Interpolation can be sued to construct expressions.
a = 1;
ex = :($a + b)
# Note: interpolating into an unquoted expression is not supported and will cause a compile-time error.
#       e.g.
#         julia> $a + b
#         ERROR: unsupported or misplaced expression $

ex = :(a in $:((1,2,3)))
Meta.show_sexpr(ex)

# Interpolating symbols into a nested expression requires enclosing each symbol in an enclosing quote block.
:(:a in $(:(:a + :b)))

# Given an expression object, one can cause Julia to evaluate (execute) it at global scope using eval().
ex = :(1 + 2)
eval(ex)

ex = :(a + b)
a = 1; b = 2;
eval(ex)

ex = :(x = 1)
x
eval(ex)
x

a = 1;
ex = Expr(:call, :+, a, :b)
a = 0; b = 2;
eval(ex)
# Note: after the expression being created, the interpoated variable is no
#       longer a variable but a fixed value

# We can use a function to create an expression
function math_expr(op, op1, op2)
  expr = Expr(:call, op, op1, op2)
  return expr
end

ex = math_expr(:+, 1, Expr(:call, :*, 4, 5))
eval(ex)

function math_expr2(op, opr1, opr2)
  opr1f, opr2f = map(x -> isa(x, Number) ? 2 * x : x, (opr1, opr2)) # meaningless
  retexpr = Expr(:call, op, opr1f, opr2f)
  return retexpr
end

math_expr2(:+, 1, 2)
ex = math_expr2(:+, 1, Expr(:call, :*, 5, 8))
eval(ex)

# Macros provide a method to include generated code in the final body of a program.
# Macros should return an expression
macro sayhello()
  return :(println("Hello, world!"))
end

@sayhello()

macro sayhello(name)
  return :(println("Hello, ", $name, "!"))
end

@sayhello("Yao")

typeof(@sayhello("Yao"))    # Void
typeof(:(@sayhello("Yao"))) # Expr

# We can view the quoted return expression using the function macroexpand().
ex = macroexpand(:(@sayhello("Yao")))
# Note: this is different from :(@sayhello("Yao"))

# Macros are necessary because they execute **WHEN CODE IS PARSED**, therefore, macros allow the programmer to generate and include fragments of customized code before the full program is run.
macro twostep(arg)
  println("I execute at parse time. The argument is: ", arg)
  return :(println("I execute at runtime. The argument is: ", $arg))
end

ex = macroexpand( :(@twostep :(1, 2, 3)) );
typeof(ex)
ex
eval(ex)

# Macros are invoked with the following general syntax.
# @name expr1 expr2 ...
# @name(expr1, expr2, ...)
@twostep :(1, 2, 3)
@twostep(:(1, 2, 3))

# It is important to emphasize that macros receive their arguments as expressions, literals, or symbols. One way to explore macro arguments is to call the show() function within the macro body.
macro showarg(x)
  return show(x)
end

@showarg(a)               # equivalent to show(:a)
@showarg(1 + 1)
@showarg(println("Yo!"))

# This is a simplified definition of Julia's @assert macro.
macro assert(ex)
    return :( $ex ? nothing : throw(AssertionError($(string(ex)))) )
end
@assert 1 == 1.0
@assert 1 == 0
# Note: it would not be possible to write this as a function, since only the value of the condition is available and it would be impossible to display the **EXPRESSION** that computed it in the error message.

# The actual definition of @assert in the standard library is more complicated. It allows the user to optionally specify their own error message, instead of just printing the failed expression.
macro assert(ex, msgs...)
    msg_body = isempty(msgs) ? ex : msgs[1]
    msg = string(msg_body)
    return :($ex ? nothing : throw(AssertionError($msg)))
end

macroexpand(:(@assert a==b))
macroexpand(:(@assert a==b "a should equal b!"))

# TODO: what is this?
# Hygiene
macro time(ex)
  return quote
    local t0 = time()
    local val = $(esc(ex))
    local t1 = time()
    println("elapsed time: ", t1-t0, " seconds")
    val
  end
end

macro zerox()
  return esc(:(x = 0))
end

function foo()
  x = 1
  @zerox
  x  # is zero
end

dump(@zerox)
foo()

for op = (:+, :*, :&, :|, :$)
  eval(quote
    ($op)(a,b,c) = ($op)(($op)(a,b),c)
  end)
end

for op = (:+, :*, :&, :|, :$)
  eval(:(($op)(a,b,c) = ($op)(($op)(a,b),c)))
end

for op = (:+, :*, :&, :|, :$)
  @eval ($op)(a,b,c) = ($op)(($op)(a,b),c)
end

@eval (+)(a, b, c) = (+)((+)(a, b), c)

ex = :((+)(1, 2, 3))
dump(ex)
eval(ex)
1 + 2 + 3 + 4

# Non-Standard String Literals
macro r_str(p)
  Regex(p)
end

r_s = @r_str "^\\s*(?:#|\$)"
r_s == r"^\s*(?:#|$)"





# A very special macro is @generated, which allows you to define so-called generated functions. These have the capability to generate specialized code depending on the types of their arguments with more flexibility and/or less code than what can be achieved with multiple dispatch. While macros work with expressions at parsing-time and cannot access the types of their inputs, a generated function gets expanded at a time **when the types of the arguments are known, but the function is not yet compiled**.

@generated function foo(x)
  println(x)
  return :(x*x)
 end

x = foo(2);
y = foo("bar");

@generated function bar(x)
  if x <: Integer
    return :(x^2)
  else
    return :(x)
  end
end

bar(4)
bar("baz")

@generated function baz(x)
  if rand() < .9
    return :(x^2)
  else
    return :("boo!")
  end
end

baz(1)

function sample()
  if rand() < .6
    return 1.5
  else
    return 0.25
  end
end
sample()
x = [sample() for dummy_i = 1:10]

function sub2ind_loop{N}(dims::NTuple{N}, I::Integer...)
    ind = I[N] - 1
    for i = N-1:-1:1
        ind = I[i]-1 + dims[i]*ind
    end
    return ind + 1
end
