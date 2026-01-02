FROM swamydev/geo-python:3.12-slim-bookworm
LABEL author=Martin Schobben
LABEL author_email=martin.schobben@tuwien.ac.at

RUN mkdir -p /etc/uv && mkdir && /etc/netrc && mkdir -p /root/.local/bin
ENV NETRC=/etc/netrc
ENV PATH=/root/.local/bin:$PATH
COPY ./ ./src
RUN --mount=type=secret,id=pypi,uid=84242,target=/etc/uv/uv.toml --mount=type=secret,id=netrc,uid=84242,target=/etc/netrc/.netrc \
    cd ./src && make uv

ENTRYPOINT ["./src/.venv/bin/xarray-rrepr"]
