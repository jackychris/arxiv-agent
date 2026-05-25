FROM m.daocloud.io/docker.io/python:3.11-slim

WORKDIR /opt/mcp/tavily-mcp

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates nodejs npm \
    && npm config set registry https://registry.npmmirror.com \
    && npm install -g supergateway \
    && rm -rf /var/lib/apt/lists/* /root/.cache

COPY third_party/mcp/tavily-mcp/ /opt/mcp/tavily-mcp/

RUN npm ci \
    && npm run build \
    && npm prune --omit=dev \
    && rm -rf /root/.cache

EXPOSE 8768

CMD ["supergateway", "--stdio", "node /opt/mcp/tavily-mcp/build/index.js", "--outputTransport", "streamableHttp", "--port", "8768", "--streamableHttpPath", "/mcp", "--healthEndpoint", "/healthz"]
