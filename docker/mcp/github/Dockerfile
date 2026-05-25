FROM m.daocloud.io/docker.io/golang:1.25.4 AS builder

ENV GOPROXY=https://goproxy.cn,direct

WORKDIR /opt/mcp/github-mcp-server

COPY third_party/mcp/github-mcp-server/ /opt/mcp/github-mcp-server/

RUN go build -o /out/github-mcp-server ./cmd/github-mcp-server

FROM m.daocloud.io/docker.io/python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates nodejs npm \
    && npm config set registry https://registry.npmmirror.com \
    && npm install -g supergateway \
    && rm -rf /var/lib/apt/lists/* /root/.cache

COPY --from=builder /out/github-mcp-server /usr/local/bin/github-mcp-server

ENV GITHUB_TOOLS=search_repositories,search_code,get_file_contents
ENV GITHUB_READ_ONLY=1

EXPOSE 8767

CMD ["supergateway", "--stdio", "github-mcp-server stdio --tools search_repositories,search_code,get_file_contents --read-only", "--outputTransport", "streamableHttp", "--port", "8767", "--streamableHttpPath", "/mcp", "--healthEndpoint", "/healthz"]
