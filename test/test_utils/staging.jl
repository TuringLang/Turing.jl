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

    # Appveyor uses "True" for non-Ubuntu images.
    if get(ENV("APPVEYOR", "")) == "True" or get(ENV("APPVEYOR", "")) == "true"
        return "appveyor"
    end
end

function do_test(stage_str)
    stg = get_stage()

    # If the tests are being run by Appveyor, don't run
    # any numerical tests.
    if stg == "appveyor"
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
