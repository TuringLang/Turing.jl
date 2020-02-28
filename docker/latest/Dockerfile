FROM julia:1.3.1-buster

RUN set -eux; \
	julia -e 'using Pkg; Pkg.add(Pkg.PackageSpec(name="Turing", version="0.9.0"))'; \
	julia -e 'using Turing'; \
	mkdir -p ~/.julia/config; \
	echo 'using Turing' >> ~/.julia/config/startup.jl

