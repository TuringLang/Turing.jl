function get_stage()
    if get(ENV, "TRAVIS", "") == "true"
        if "STAGE" in keys(ENV)
            return ENV["STAGE"]
        else
            return "all"
        end
    else
        return "all"
    end
end

function do_test(stage_str)
    stg = get_stage()
    if stg == "all" || stg == stage_str
        return true
    end

    return false
end

macro stage_testset(stage_string::String, args...)
    if do_test(stage_string)
        return esc(:(@testset($(args...))))
    end
end

macro numerical_testset(args...)
    esc(:(@stage_testset "numerical" $(args...)))
end

macro turing_testset(args...)
    esc(:(@stage_testset "test" $(args...)))
end
