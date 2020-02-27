FROM julia:1.3.1-buster
LABEL maintainer="The Turing team"

RUN set -eux; \
	julia -e 'using Pkg; Pkg.add(Pkg.PackageSpec(name="Turing", version="0.8.3"))'; \
	julia -e 'using Turing'; \
	mkdir -p ~/.julia/config; \
	echo 'using Turing' >> ~/.julia/config/startup.jl

