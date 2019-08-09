using Base64

## Text Utilities

function find_yaml_header(lines)
    yaml = []
    yaml_start, yaml_stop = 1, length(lines)

    # read out YAML block.
    if strip(lines[1]) == "---"
        for i = 2:length(lines)
            if strip(lines[i]) == "---"
                yaml = lines[1:i]
                yaml_stop = i
                break
            end
        end
    end
    return yaml, yaml_start, yaml_stop
end

function remove_yaml(file, key=nothing)
    lines = readlines(file, keep=true)
    yaml, yaml_start, yaml_stop = find_yaml_header(lines)
    if !isempty(yaml)
        open(file, "w+") do f
            # write back YAML lines.
            if key != nothing
                for line in yaml
                    startswith(line, key * ":") || write(f, line)
                end
            end
            # write back non-YAML lines
            for line in lines[yaml_stop + 1 : end]
                write(f, line)
            end
        end
    end

    return yaml
end

"""
    fix_header_1

This function is used to add a first-level header into the
markdown file.

There are no H1 titles for documents in tutorials, so we need
this.
"""
function fix_header_1(file)
    lines = readlines(file, keep=true)
    yaml, yaml_start, yaml_stop = find_yaml_header(lines)
    isempty(yaml) && return

    for line in lines[yaml_stop + 1 : end]
        strip(line) == "" && continue
        if startswith(line, "# ") # has a h1 title
            return
        else # no h1 title
            break
        end
    end

    title = "Untitled"
    for line in yaml
        if startswith(line, "title:")
            title = line[7:end] |> strip
            break
        end
    end

    open(file, "w+") do f
        for line in yaml
            write(f, line)
        end
        write(f, "\n# $title\n")
        # write back non-YAML lines
        for line in lines[yaml_stop + 1 : end]
            write(f, line)
        end
    end
end

function tidy_api(file)
	lines = readlines(file, keep=true)

    # Find the ID sections.
    for i = 1:(length(lines)-1)
		first = lines[i]
		second = lines[i+1]
        if startswith(first, "<a id=") && startswith(second, "**")
            first = replace(first, "\n" => "")
			second = replace(second, "\n" => "")
			# final_line = replace(first, ">" => ">$second", count = 1)

			lines[i] = "### $first $second"
			lines[i+1] = ""
        end
    end

    # Write lines back.
    open(file, "w+") do f
        for line in lines
            write(f, line)
        end
    end
end

## File Utilities

function withfile(func, file::AbstractString, contents::AbstractString)
    hasfile = isfile(file)
    original = hasfile ? read(file, String) : ""
    open(file, "w") do stream
        print(stream, contents)
        flush(stream) # Make sure file is written before continuing.
    end
    try
        func()
    finally
        if hasfile
            open(file, "w") do stream
                print(stream, original)
            end
        else
            #rm(file)
        end
    end
end

function filecopy(src, dst, force_copy = true)
    for item in readdir(src)
        cp(joinpath(src, item), joinpath(dst, item), force = force_copy)
    end
end

function filecopy_deep(src, dst)
    for (root, dirs, files) in walkdir(src)
        for file in files
            source = joinpath(root, file)
            target = replace(source, src => dst)
            isdir(dirname(target)) || mkpath(dirname(target))
            cp(source, target, force=true)
        end
    end
end

## Building Utilities

function preprocess_markdown(folder)
    yaml_dict = Dict()

    try
        for (root, dirs, files) in walkdir(folder)
            for file in files
				if endswith(file, ".md")
	                full_path = joinpath(root, file)
	                yaml_dict[full_path] = remove_yaml(full_path)
				end
            end
        end
    catch e
        # println("Markdown copy error: $e")
        rethrow(e)
    end

    return yaml_dict
end

function postprocess_markdown(folder, yaml_dict; original = "")
    try
        for (root, dirs, files) in walkdir(folder)
            for file in files
                full_path = joinpath(root, file)
                original_path = full_path

                if length(original) > 0
                    # original_path = abspath(original, file)
                    original_path = replace(full_path, folder => original)
                    # original_path = replace(
                    #     full_path,
                    #     joinpath("docs", "site") => joinpath("docs", "src")
                    # )
                end

                if haskey(yaml_dict, original_path)
                    # println("Original: $original_path => Full path: $full_path")

                    txt = open(f -> read(f, String), full_path)
                    open(full_path, "w+") do f
                        # Add in the yaml block.
                        for line in yaml_dict[original_path]
                            write(f, line)
                        end

						txt = replace(txt, "api.md" => "{{site.baseurl}}/docs/library/")

                        # Add the rest of the text.
						if original_path == full_path
							write(f, txt)
						else
							write(f, replace(txt, "![](figures" => "![](/{{site.baseurl}}/tutorials/figures"))
						end
                    end
                elseif endswith(file, ".md")
                    println("Original: $original_path")
                    println("Full:     $full_path \n")
                end

				# Make specific api items headers.
				if file == "api.md" && original_path != full_path
					tidy_api(full_path)
				end
            end
        end
    catch e
        # println("Markdown copy error: $e")
        rethrow(e)
    end

    return yaml_dict
end

function with_clean_docs(func, source, target)
    src_temp = mktempdir()
    cp(source, src_temp, force=true)
    yaml_dict = preprocess_markdown(source)

    try
        func(source, target)
    catch e
        rethrow(e)
    finally
        # Put back the original files in the event of an error.
        cp(src_temp, source_path, force=true)
        rm(src_temp, recursive=true)
    end
    postprocess_markdown(build_path, yaml_dict, original=source_path)
end

function copy_tutorial(tutorial_path)
    isdir(tutorial_path) || mkpath(tutorial_path)
    # Clone TuringTurorials
    tmp_path = tempname()
    mkdir(tmp_path)
    clone("https://github.com/TuringLang/TuringTutorials", tmp_path)

    # Move to markdown folder.
    md_path = joinpath(tmp_path, "markdown")

    # Copy the .md versions of all examples.
    try
        @debug(md_path)
        for file in readdir(md_path)
            full_path = joinpath(md_path, file)
            target_path = joinpath(tutorial_path, file)
            println("Copying $full_path to $target_path")
            cp(full_path, target_path, force=true)
            # endswith(target_path, ".md") && remove_yaml(target_path, "permalink")
            endswith(target_path, ".md") && fix_header_1(target_path)
        end
        index = joinpath(@__DIR__, "src/tutorials/index.md")
        cp(index, tutorial_path * "/index.md", force=true)
    catch e
        rethrow(e)
    finally
        rm(tmp_path, recursive=true)
    end
end

## Deployment Utilities

# This is adapted from the Documenter.jl 'git_push' function.
function update_homepage(repo, branch, target)
    println("Updating homepage...")

    dirname = ""

    mktempdir() do temp
        # Get current directory.
        root = Base.source_dir()
        root === nothing ? pwd() : root

        # Other variables.
        sha = cd(root) do
            # We'll make sure we run the git commands in the source directory (root), in case
            # the working directory has been changed (e.g. if the makedocs' build argument is
            # outside root).
            try
                readchomp(`git rev-parse --short HEAD`)
            catch
                # git rev-parse will throw an error and return code 128 if it is not being
                # run in a git repository, which will make run/readchomp throw an exception.
                # We'll assume that if readchomp fails it is due to this and set the sha
                # variable accordingly.
                "(not-git-repo)"
            end
        end

        key = ""
        target_dir = ""
        upstream = ""
        keyfile = ""

        cd(root) do
            # Grab the push key.
            key = get(ENV, "DOCUMENTER_KEY", "")

            dirname = isempty(dirname) ? temp : joinpath(temp, dirname)
            isdir(dirname) || mkpath(dirname)

            keyfile = abspath(joinpath(root, ".documenter"))
            target_dir = abspath(target)

            # The upstream URL to which we push new content and the ssh decryption commands.
            upstream = "git@$(replace(repo, "github.com/" => "github.com:"))"

            write(keyfile, String(base64decode(key)))
            chmod(keyfile, 0o600)
        end

        try
            # Use a custom SSH config file to avoid overwriting the default user config.
            withfile(joinpath(homedir(), ".ssh", "config"),
                """
                Host github.com
                    StrictHostKeyChecking no
                    HostName github.com
                    IdentityFile $keyfile
                """
            ) do
                # cd(temp) do
                # end

                cd(temp) do
                    println("Setting up git...")
                    # Setup git.
                    run(`git init`)
                    run(`git config user.name "autodocs"`)
                    run(`git config user.email "autodocs"`)

                    # Fetch from remote and checkout the branch.
                    run(`git remote add upstream $upstream`)
                    run(`git fetch upstream`)

                    try
                        run(`git checkout -b $branch upstream/$branch`)
                    catch e
                        run(`git checkout --orphan $branch`)
                        run(`git commit --allow-empty -m "Initial empty commit for homepage"`)
                    end

                    # Copy the target directory to the specified directory.
                    filecopy(target_dir, dirname)

                    # Add, commit, and push the docs to the remote.
                    run(`git add -A .`)
                    if !success(`git diff --cached --exit-code`)
                        run(`git commit -m "homepage update based on $sha"`)
                        run(`git push -q upstream HEAD:$branch`)
                    end
                end
            end
        catch e
            println("Errored out building homepage, $e")
        finally
            # Remove the unencrypted private key.
            isfile(keyfile) && rm(keyfile)
        end
    end
end
