FROM m.daocloud.io/docker.io/python:3.11-slim

ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /opt/mcp/paper-search-mcp

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates nodejs npm \
    && npm config set registry https://registry.npmmirror.com \
    && pip install --no-cache-dir uv \
    && npm install -g supergateway \
    && rm -rf /var/lib/apt/lists/* /root/.cache

COPY third_party/mcp/paper-search-mcp/ /opt/mcp/paper-search-mcp/

RUN uv sync --frozen --no-dev \
    && rm -rf /root/.cache

EXPOSE 8765

CMD ["supergateway", "--stdio", "uv run --directory /opt/mcp/paper-search-mcp -m paper_search_mcp.server", "--outputTransport", "streamableHttp", "--port", "8765", "--streamableHttpPath", "/mcp", "--healthEndpoint", "/healthz"]
