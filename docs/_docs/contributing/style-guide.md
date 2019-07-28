---
title: Style Guide
---

# Style Guide

This style guide is adapted from [Invenia](https://invenia.ca/labs/)'s style guide. We would like to thank them for allowing us to access and use it. Please don't let not having read it stop you from contributing to Turing! No one will be annoyed if you open a PR whose style doesn't follow these conventions; we will just help you correct it before it gets merged.


These conventions were originally written at Invenia, taking inspiration from a variety of sources including Python's [PEP8](http://legacy.python.org/dev/peps/pep-0008), Julia's [Notes for Contributors](https://github.com/JuliaLang/julia/blob/master/CONTRIBUTING.md), and Julia's [Style Guide](https://docs.julialang.org/en/latest/manual/style-guide/).


What follows is a mixture of a verbatim copy of Invenia's original guide and some of our own modifications.


## A Word on Consistency


When adhering to this style it's important to realize that these are guidelines and not rules. This is [stated best in the PEP8](http://legacy.python.org/dev/peps/pep-0008/#a-foolish-consistency-is-the-hobgoblin-of-little-minds):


> A style guide is about consistency. Consistency with this style guide is important. Consistency within a project is more important. Consistency within one module or function is most important.



> But most importantly: know when to be inconsistent – sometimes the style guide just doesn't apply. When in doubt, use your best judgment. Look at other examples and decide what looks best. And don't hesitate to ask!



## Synopsis


Attempt to follow both the [Julia Contribution Guidelines](https://github.com/JuliaLang/julia/blob/master/CONTRIBUTING.md#general-formatting-guidelines-for-julia-code-contributions), the [Julia Style Guide](https://docs.julialang.org/en/latest/manual/style-guide/), and this guide. When convention guidelines conflict this guide takes precedence (known conflicts will be noted in this guide).


  * Use 4 spaces per indentation level, no tabs.
  * Try to adhere to a 92 character line length limit.
  * Use upper camel case convention for [modules](https://docs.julialang.org/en/latest/manual/modules/) and [types](https://docs.julialang.org/en/latest/manual/types/).
  * Use lower case with underscores for method names (note: Julia code likes to use lower case without underscores).
  * Comments are good, try to explain the intentions of the code.
  * Use whitespace to make the code more readable.
  * No whitespace at the end of a line (trailing whitespace).
  * Avoid padding brackets with spaces. ex. `Int64(value)` preferred over `Int64( value )`.


## Editor Configuration


### Sublime Text Settings


If you are a user of Sublime Text we recommend that you have the following options in your Julia syntax specific settings. To modify these settings first open any Julia file (`*.jl`) in Sublime Text. Then navigate to: `Preferences > Settings - More > Syntax Specific - User`


```json
{
    "translate_tabs_to_spaces": true,
    "tab_size": 4,
    "trim_trailing_white_space_on_save": true,
    "ensure_newline_at_eof_on_save": true,
    "rulers": [92]
}
```


### Vim Settings


If you are a user of Vim we recommend that you add the following options to your `.vimrc` file.


```
set tabstop=4                             " Sets tabstops to a width of four columns.
set softtabstop=4                         " Determines the behaviour of TAB and BACKSPACE keys with expandtab.
set shiftwidth=4                          " Determines the results of >>, <<, and ==.

au FileType julia setlocal expandtab      " Replaces tabs with spaces.
au FileType julia setlocal colorcolumn=93 " Highlights column 93 to help maintain the 92 character line limit.
```


By default, Vim seems to guess that `.jl` files are written in Lisp. To ensure that Vim recognizes Julia files you can manually have it check for the `.jl` extension, but a better solution is to install [Julia-Vim](https://github.com/JuliaLang/julia-vim), which also includes proper syntax highlighting and a few cool other features.


### Atom Settings


Atom defaults preferred line length to 80 characters. We want that at 92 for julia. To change it:


1. Go to `Atom -> Preferences -> Packages`.
2. Search for the "language-julia" package and open the settings for it.
3. Find preferred line length (under "Julia Grammar") and change it to 92.


## Code Formatting


### Function Naming


Names of functions should describe an action or property irrespective of the type of the argument; the argument's type provides this information instead. For example, `buyfood(food)` should be `buy(food::Food)`.


Names of functions should usually be limited to one or two lowercase words. Ideally write `buyfood` not `buy_food`, but if you are writing a function whose name is hard to read without underscores then please do use them.


### Method Definitions


Only use short-form function definitions when they fit on a single line:


```julia
# Yes:
foo(x::Int64) = abs(x) + 3
# No:
foobar(array_data::AbstractArray{T}, item::T) where {T<:Int64} = T[
    abs(x) * abs(item) + 3 for x in array_data
]

# No:
foobar(
    array_data::AbstractArray{T},
    item::T,
) where {T<:Int64} = T[abs(x) * abs(item) + 3 for x in array_data]
# Yes:
function foobar(array_data::AbstractArray{T}, item::T) where T<:Int64
    return T[abs(x) * abs(item) + 3 for x in array_data]
end
```


When using long-form functions [always use the `return` keyword](https://groups.google.com/forum/#!topic/julia-users/4RVR8qQDrUg):


```julia
# Yes:
function fnc(x::T) where T
    result = zero(T)
    result += fna(x)
    return result
end
# No:
function fnc(x::T) where T
    result = zero(T)
    result += fna(x)
end

# Yes:
function Foo(x, y)
    return new(x, y)
end
# No:
function Foo(x, y)
    new(x, y)
end
```


Functions definitions with parameter lines which exceed 92 characters should separate each parameter by a newline and indent by one-level:


```julia
# Yes:
function foobar(
    df::DataFrame,
    id::Symbol,
    variable::Symbol,
    value::AbstractString,
    prefix::AbstractString="",
)
    # code
end

# Ok:
function foobar(df::DataFrame, id::Symbol, variable::Symbol, value::AbstractString, prefix::AbstractString="")
    # code
end
# No:
function foobar(df::DataFrame, id::Symbol, variable::Symbol, value::AbstractString,
    prefix::AbstractString="")

    # code
end
# No:
function foobar(
        df::DataFrame,
        id::Symbol,
        variable::Symbol,
        value::AbstractString,
        prefix::AbstractString="",
    )
    # code
end
```


### Keyword Arguments


When calling a function always separate your keyword arguments from your positional arguments with a semicolon. This avoids mistakes in ambiguous cases (such as splatting a `Dict`).


```julia
# Yes:
xy = foo(x; y=3)
# No:
xy = foo(x, y=3)
```


### Whitespace


Avoid extraneous whitespace in the following situations:


  * Immediately inside parentheses, square brackets or braces.


```julia
Yes: spam(ham[1], [eggs])
No:  spam( ham[ 1 ], [ eggs ] )
```


  * Immediately before a comma or semicolon:


```julia
Yes: if x == 4 @show(x, y); x, y = y, x end
No:  if x == 4 @show(x , y) ; x , y = y , x end
```


  * When using ranges unless additional operators are used:


```julia
Yes: ham[1:9], ham[1:3:9], ham[1:3:end]
No:  ham[1: 9], ham[1 : 3: 9]
```


```julia
Yes: ham[lower:upper], ham[lower:step:upper]
Yes: ham[lower + offset : upper + offset]
Yes: ham[(lower + offset):(upper + offset)]
No:  ham[lower + offset:upper + offset]
```


  * More than one space around an assignment (or other) operator to align it with another:


```julia
# Yes:
x = 1
y = 2
long_variable = 3

# No:
x             = 1
y             = 2
long_variable = 3
```


  * Always surround these binary operators with a single space on either side: assignment ($=$), [updating operators](https://docs.julialang.org/en/latest/manual/mathematical-operations/#Updating-operators-1) ($+=$, $-=$, etc.), [numeric comparisons operators](https://docs.julialang.org/en/latest/manual/mathematical-operations/#Numeric-Comparisons-1) ($==$, $<$, $>$, $!=$, etc.). Note that this guideline does not apply when performing assignment in method definitions.


```julia
Yes: i = i + 1
No:  i=i+1

Yes: submitted += 1
No:  submitted +=1

Yes: x^2 < y
No:  x^2<y
```


  * Assignments using expanded array, tuple, or function notation should have the first open bracket on the same line assignment operator and the closing bracket should match the indentation level of the assignment. Alternatively you can perform assignments on a single line when they are short:


```julia
# Yes:
arr = [
    1,
    2,
    3,
]
arr = [
    1, 2, 3,
]
result = Function(
    arg1,
    arg2,
)
arr = [1, 2, 3]


# No:
arr =
[
    1,
    2,
    3,
]
arr =
[
    1, 2, 3,
]
arr = [
    1,
    2,
    3,
    ]
```


  * Nested array or tuples that are in expanded notation should have the opening and closing brackets at the same indentation level:


```julia
# Yes:
x = [
    [
        1, 2, 3,
    ],
    [
        "hello",
        "world",
    ],
    ['a', 'b', 'c'],
]

# No:
y = [
    [
        1, 2, 3,
    ], [
        "hello",
        "world",
    ],
]
z = [[
        1, 2, 3,
    ], [
        "hello",
        "world",
    ],
]
```


  * Always include the trailing comma when working with expanded arrays, tuples or functions notation. This allows future edits to easily move elements around or add additional elements. The trailing comma should be excluded when the notation is only on a single-line:


```julia
# Yes:
arr = [
    1,
    2,
    3,
]
result = Function(
    arg1,
    arg2,
)
arr = [1, 2, 3]

# No:
arr = [
    1,
    2,
    3
]
result = Function(
    arg1,
    arg2
)
arr = [1, 2, 3,]
```


  * Triple-quotes use the indentation of the lowest indented line (excluding the opening triple-quote). This means the closing triple-quote should be aligned to least indented line in the string. Triple-backticks should also follow this style even though the indentation does not matter for them.


```julia
# Yes:
str = """
    hello
    world!
    """
str = """
        hello
    world!
    """
cmd = ```
    program
        --flag value
        parameter
      ```
# No:
str = """
    hello
    world!
"""
```


### Comments


Comments should be used to state the intended behaviour of code. This is especially important when the code is doing something clever that may not be obvious upon first inspection. Avoid writing comments that state exactly what the code obviously does.


```julia
# Yes:
x = x + 1      # Compensate for border

# No:
x = x + 1      # Increment x
```


Comments that contradict the code are much worse than no comments. Always make a priority of keeping the comments up-to-date with code changes!


Comments should be complete sentences. If a comment is a phrase or sentence, its first word should be capitalized, unless it is an identifier that begins with a lower case letter (never alter the case of identifiers!).


If a comment is short, the period at the end can be omitted. Block comments generally consist of one or more paragraphs built out of complete sentences, and each sentence should end in a period.


Comments should be separated by at least two spaces from the expression and have a single space after the `#`.


When referencing Julia in documentation note that "Julia" refers to the programming language while "julia" (typically in backticks, e.g. `julia`) refers to the executable.


```julia

# A commment
code

# Another comment
more code

TODO
```


### Documentation


It is recommended that most modules, types and functions should have [docstrings](http://docs.julialang.org/en/latest/manual/documentation/). That being said, only exported functions are required to be documented. Avoid documenting methods like `==` as the built in docstring for the function already covers the details well. Try to document a function and not individual methods where possible as typically all methods will have similar docstrings. If you are adding a method to a function which was defined in `Base` or another package only add a docstring if the behaviour of your function deviates from the existing docstring.


Docstrings are written in [Markdown](https://en.wikipedia.org/wiki/Markdown) and should be concise. Docstring lines should be wrapped at 92 characters.


```julia
"""
    bar(x[, y])

Compute the Bar index between `x` and `y`. If `y` is missing, compute the Bar index between
all pairs of columns of `x`.
"""
function bar(x, y) ...
```


When types or methods have lots of parameters it may not be feasible to write a concise docstring. In these cases it is recommended you use the templates below. Note if a section doesn't apply or is overly verbose (for example "Throws" if your function doesn't throw an exception) it can be excluded. It is recommended that you have a blank line between the headings and the content when the content is of sufficient length. Try to be consistent within a docstring whether you use this additional whitespace. Note that the additional space is only for reading raw markdown and does not effect the rendered version.


Type Template (should be skipped if is redundant with the constructor(s) docstring):


```julia
"""
    MyArray{T,N}

My super awesome array wrapper!

# Fields
- `data::AbstractArray{T,N}`: stores the array being wrapped
- `metadata::Dict`: stores metadata about the array
"""
struct MyArray{T,N} <: AbstractArray{T,N}
    data::AbstractArray{T,N}
    metadata::Dict
end
```


Function Template (only required for exported functions):


```julia
"""
    mysearch(array::MyArray{T}, val::T; verbose=true) where {T} -> Int

Searches the `array` for the `val`. For some reason we don't want to use Julia's
builtin search :)

# Arguments
- `array::MyArray{T}`: the array to search
- `val::T`: the value to search for

# Keywords
- `verbose::Bool=true`: print out progress details

# Returns
- `Int`: the index where `val` is located in the `array`

# Throws
- `NotFoundError`: I guess we could throw an error if `val` isn't found.
"""
function mysearch(array::AbstractArray{T}, val::T) where T
    ...
end
```


If your method contains lots of arguments or keywords you may want to exclude them from the method signature on the first line and instead use `args...` and/or `kwargs...`.


```julia
"""
    Manager(args...; kwargs...) -> Manager

A cluster manager which spawns workers.

# Arguments

- `min_workers::Integer`: The minimum number of workers to spawn or an exception is thrown
- `max_workers::Integer`: The requested number of worker to spawn

# Keywords

- `definition::AbstractString`: Name of the job definition to use. Defaults to the
    definition used within the current instance.
- `name::AbstractString`: ...
- `queue::AbstractString`: ...
"""
function Manager(...)
    ...
end
```


Feel free to document multiple methods for a function within the same docstring. Be careful to only do this for functions you have defined.


```julia
"""
    Manager(max_workers; kwargs...)
    Manager(min_workers:max_workers; kwargs...)
    Manager(min_workers, max_workers; kwargs...)

A cluster manager which spawns workers.

# Arguments

- `min_workers::Int`: The minimum number of workers to spawn or an exception is thrown
- `max_workers::Int`: The number of requested workers to spawn

# Keywords

- `definition::AbstractString`: Name of the job definition to use. Defaults to the
    definition used within the current instance.
- `name::AbstractString`: ...
- `queue::AbstractString`: ...
"""
function Manager end

```


If the documentation for bullet-point exceeds 92 characters the line should be wrapped and slightly indented. Avoid aligning the text to the `:`.


```julia
"""
...

# Keywords
- `definition::AbstractString`: Name of the job definition to use. Defaults to the
    definition used within the current instance.
"""
```


For additional details on documenting in Julia see the [official documentation](http://docs.julialang.org/en/latest/manual/documentation/).


## Test Formatting


### Testsets


Julia provides [test sets](https://docs.julialang.org/en/latest/stdlib/Test/#Working-with-Test-Sets-1) which allows developers to group tests into logical groupings. Test sets can be nested and ideally packages should only have a single "root" test set. It is recommended that the "runtests.jl" file contains the root test set which contains the remainder of the tests:


```julia
@testset "PkgExtreme" begin
    include("arithmetic.jl")
    include("utils.jl")
end
```


The file structure of the `test` folder should mirror that of the `src` folder. Every file in `src` should have a complementary file in the `test` folder, containing tests relevant to that file's contents.


### Comparisons


Most tests are written in the form `@test x == y`. Since the `==` function doesn't take types into account tests like the following are valid: `@test 1.0 == 1`. Avoid adding visual noise into test comparisons:


```julia
# Yes:
@test value == 0

# No:
@test value == 0.0
```


In cases where you are checking the numerical validity of a model's parameter estimates, please use the `check_numerical` function found in `test/test_utils/numerical_tests.jl`. This function will evaluate a model's parameter estimates using a given tolerance level, and test will only be performed if you are running the test suite locally or if Travis is executing the "Numerical" testing stage.


Here is an example of usage:


```julia
# Check that m and s are plus or minus one from 1.5 and 2.2, respectively.
check_numerical(chain, [:m, :s], [1.5, 2.2], eps = 1.0)

# Checks the estimates for a default gdemo model using values 1.5 and 2.0.
check_gdemo(chain, eps = 0.1)

# Checks the estimates for a default MoG model.
check_MoGtest_default(chain, eps = 0.1)
```

