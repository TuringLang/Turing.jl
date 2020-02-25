FROM julia:1.3.1-buster
LABEL maintainer="The Turing team"

RUN true \
    && julia -e 'using Pkg; Pkg.add("Turing");' \
    && julia -e 'using Turing;' \
    && mkdir -p /root/.julia/config \
    && echo "using Turing" > /root/.julia/config/startup.jl

WORKDIR /root

CMD ["julia"]
