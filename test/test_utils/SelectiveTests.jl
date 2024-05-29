module SelectiveTests

"""
    parse_args(args)

Parse the command line arguments to get the included and excluded test file paths.

The arguments are expected to be in the form:
```
a b c --skip d e f
```
where a test file is to be included if and only if
1) the argument list is empty, in which case all files are included,
or
2)
    a) it has as a substring of its path any of the strings `a`, `b`, or `c`,
    and
    b) it does not have as a substring of its path any of the strings `d`, `e`, or `f`.

The substring checks are done case-insensitively.
"""
function parse_args(args)
    included_paths = Vector{String}()
    excluded_paths = Vector{String}()
    for (i, arg) in enumerate(args)
        if arg == "--skip"
            append!(excluded_paths, args[i+1:end])
            break
        else
            push!(included_paths, arg)
        end
    end
    return included_paths, excluded_paths
end

"""
    isincluded(filepath, included_paths, excluded_paths)

Check if a file should be included in the tests.

`included_paths` and `excluded_paths` are the output of [`parse_args`](@ref).

See [`parse_args`](@ref) for the logic of when a file should be included.
"""
function isincluded(
    filepath::AbstractString,
    included_paths::Vector{<:AbstractString},
    excluded_paths::Vector{<:AbstractString},
)
    if any(excl -> occursin(lowercase(excl), lowercase(filepath)), excluded_paths)
        return false
    end
    if any(incl -> occursin(lowercase(incl), lowercase(filepath)), included_paths)
        return true
    end
    return isempty(included_paths)
end

end
