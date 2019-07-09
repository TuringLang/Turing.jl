using Base64

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

function remove_yaml(file)
    lines = readlines(file, keep=true)
    yaml = []

    yaml_start = 1
    yaml_stop = length(lines)

    # Read out YAML block.
    if replace(lines[1], "\n" => "") == "---"
        for i = 2:length(lines)
            if replace(lines[i], "\n" => "") == "---"
                yaml = lines[1:i]
                yaml_stop = i
                break
            end
        end

        # Write non-YAML back to file.
        open(file, "w+") do f
            for line in lines[yaml_stop + 1 : end]
                write(f, line)
            end
        end
    end

    return yaml
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
