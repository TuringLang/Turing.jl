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

function filecopy(src, dst)
    for item in readdir(src)
        cp(joinpath(src, item), joinpath(dst, item), force = true)
    end
end
