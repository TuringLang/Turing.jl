module SelectiveTests

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

function isincluded(filepath, included_paths, excluded_paths)
    if any(excl -> occursin(lowercase(excl), lowercase(filepath)), excluded_paths)
        return false
    end
    if any(incl -> occursin(lowercase(incl), lowercase(filepath)), included_paths)
        return true
    end
    return isempty(included_paths)
end

end
