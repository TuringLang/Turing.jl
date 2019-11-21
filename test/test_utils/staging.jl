function get_stage()
    # Appveyor uses "True" for non-Ubuntu images.
    if get(ENV, "APPVEYOR", "") == "True" || get(ENV, "APPVEYOR", "") == "true"
        return "nonnumeric"
    end

    # Handle Travis and Github Actions specially.
    if get(ENV, "TRAVIS", "") == "true" || get(ENV, "GITHUB_ACTIONS", "") == "true"
        if "STAGE" in keys(ENV)
            return ENV["STAGE"]
        else
            return "all"
        end
    end

    return "all"
end

function do_test(stage_str)
    stg = get_stage()

    # If the tests are being run by Appveyor, don't run
    # any numerical tests.
    if stg == "nonnumeric"
        if stage_str == "numerical"
            return false
        else
            return true
        end
    end

    # Otherwise run the regular testing procedure.
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
