using Distributed
import Nanosoldier, GitHub

nodes = Dict(Any => addprocs())
@everywhere import Nanosoldier

ENV["GIT_COMMITTER_NAME"] = ENV["GIT_AUTHOR_NAME"] = "BayesBot"
ENV["GIT_COMMITTER_EMAIL"] = ENV["GIT_AUTHOR_EMAIL"] = "BayesBot@example.com"


auth = GitHub.authenticate(ENV["BENCHMARK_KEY"]) # ENV["GITHUB_AUTH"])
secret = get(ENV,"GITHUB_SECRET", "none")

config = Nanosoldier.Config(ENV["USER"], nodes, auth, secret;
                            workdir = joinpath(homedir(), "test_workdir"),
                            trackrepo = "TuringLang/Turing.jl",
                            reportrepo = "TuringLang/BenchmarkReports",
                            trigger =  r"\@BayesBot\s*`runbenchmarks\(.*?\)`",
                            admin = "BayesBot",
                            testmode = true)

server = Nanosoldier.Server(config)


using Sockets

EVENT_FILE = get(ENV, "GITHUB_EVENT_PATH", "")
if isempty(EVENT_FILE) # run as a webhook server
    run(server, IPv4(0,0,0,0), 8080)
else # run on GitHub Action
    Nanosoldier.run_as_github_action(server)
end
