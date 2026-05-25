# arxiv-agent

`arxiv-agent` 是一个面向 AI/CS 研究场景的图驱动多阶段研究智能体系统。系统会把用户问题改写、拆解成研究任务，异步调用论文、GitHub 和网页检索工具收集证据，再经过证据评估、答案草稿、批判性审查和最终答案生成，输出带 inline citations 和 `Sources Used` 的研究答案。

## 当前架构

系统总览：

```text
                       +----------------------+
                       |       Browser        |
                       |  Web UI + SSE stream |
                       +----------+-----------+
                                  |
                                  v
                       +----------------------+
                       |        Nginx         |
                       |   reverse proxy :80  |
                       +----------+-----------+
                                  |
                                  v
+----------------------+  HTTP/SSE  +----------------------+
|      Postgres        |<---------->|         app          |
| run history/checkpoint|           | FastAPI + LangGraph  |
+----------------------+           +----------+-----------+
                                             |
                                             | A2A task dispatch
                                             v
                                  +----------------------+
                                  |   research-agent     |
                                  | A2A researcher server|
                                  +----------+-----------+
                                             |
                                             | MCP async client
                                             v
          +-----------------------+----------+-----------------------+
          |                       |                                  |
          v                       v                                  v
+------------------+    +------------------+              +------------------+
|  academic-mcp    |    |    github-mcp    |              |     web-mcp      |
| arXiv/S2/OpenAlex|    | GitHub read-only |              | Tavily search/   |
| Crossref/DBLP    |    | repo/code/files  |              | extract          |
+------------------+    +------------------+              +------------------+
```

整体链路基于 LangGraph 编排：

```text
rewrite_query
  -> plan_tasks
  -> review_plan
  -> dispatch_tasks
  -> research_task
  -> collect_results
  -> synthesize_answer
  -> critique_answer
  -> review_critique
  -> finalize
```

核心设计：

- **图驱动流程**：每个阶段是独立 graph node，支持计划审核、答案审核、follow-up 研究和断点续跑。
- **异步研究任务**：planner 将问题拆成多个 research task，researcher 通过 A2A Agent 执行；单个 researcher 内部是 ReAct 风格工具调用循环。
- **多源证据收集**：通过 MCP async client 接入论文、GitHub 和网页服务，并在 client 层做工具 allowlist、超时、重试、缓存、限速和域名拦截。
- **答案自处理**：`synthesize_answer` 内部执行 `evidence evaluator -> draft answer -> critical review -> final answer`。
- **最终答案简易评估**：在最终答案生成后执行确定性评估，检查 citation 有效性、citation 覆盖、query focus、答案充实度、内部流程泄漏和过度自信表述。
- **实时交互**：FastAPI 提供 SSE 流式接口，前端实时展示任务规划、工具调用、研究进度、审核节点和最终答案。
- **持久化**：Postgres 保存 run history / checkpoint；`runs/` 和 `paper_content/` 用于运行轨迹和论文内容缓存目录。

## 服务组成

Docker Compose 管理以下服务：

| 服务 | 作用 |
|---|---|
| `postgres` | 运行状态、run history、checkpoint 持久化 |
| `academic-mcp` | 论文/学术检索 MCP，基于 vendored `openags/paper-search-mcp` |
| `github-mcp` | GitHub MCP，基于 vendored `github/github-mcp-server`，只读模式 |
| `web-mcp` | Tavily MCP，用于网页搜索和网页抽取 |
| `research-agent` | A2A researcher server，执行具体 research task |
| `app` | FastAPI 主服务，提供页面、API、SSE stream |
| `nginx` | 反向代理，对外暴露网页入口 |

主服务间调用关系：

```text
Browser
  -> nginx
  -> app
  -> research-agent
  -> MCPClient
  -> academic-mcp / github-mcp / web-mcp
```

## MCP 数据源与工具

MCP server 源码 vendored 在 `third_party/mcp/`，Docker 构建时只从本地 `COPY`，不在 build 阶段 `git clone`、`go install @latest` 或 `npx tavily-mcp@...` 拉业务源码。

当前 allowlist 来自 `mcp_config.json`。

### Academic

服务：`academic-mcp`  
地址：`http://127.0.0.1:8765/mcp`

可用工具：

- `search_arxiv`
- `search_semantic`
- `search_openalex`
- `search_crossref`
- `get_crossref_paper_by_doi`
- `search_dblp`
- `download_arxiv`
- `download_semantic`
- `read_arxiv_paper`
- `read_semantic_paper`
- `read_openalex_paper`
- `read_dblp_paper`

### GitHub

服务：`github-mcp`  
地址：`http://127.0.0.1:8767/mcp/`

运行策略：

- GitHub MCP 以 `--read-only` 启动。
- 只暴露只读检索相关工具。
- 建议使用 public-only fine-grained token。

可用工具：

- `search_repositories`
- `search_code`
- `get_file_contents`

### Web / Tavily

服务：`web-mcp`  
地址：`http://127.0.0.1:8768/mcp/`

可用工具：

- `tavily_search`
- `tavily_extract`

运行策略：

- `tavily_search` 用于通用网页检索。
- `tavily_extract` 用于非论文网页内容抽取。
- `tavily_extract` 命中配置的受限域名时会被 client 拦截，不发出实际 extract 请求。
- `tavily_extract` 如果目标 URL 是 `arxiv.org/abs/...`、`arxiv.org/pdf/...` 或 `export.arxiv.org/...`，会共享 arXiv limiter。

## 工具调用治理

`mcp_client.py` 负责统一工具调用治理：

- **allowlist**：只向 agent 暴露 `mcp_config.json` 中配置的工具。
- **fail fast**：启动时如果 allowlist 工具缺失，直接报错并输出 server 实际工具列表。
- **timeout**：工具调用统一受 `MCP_TOOL_TIMEOUT` 控制。
- **retry/backoff**：对 timeout、rate limit、recoverable tool error 做有限重试。
- **缓存**：只缓存非空成功结果，避免空搜索结果污染后续运行。
- **arXiv 限速**：`search_arxiv`、`download_arxiv`、`read_arxiv_paper` 强制使用全局 limiter，当前最小间隔为 5 秒。
- **Tavily arXiv URL 限速**：`tavily_extract` 目标为 arXiv URL 时共享 arXiv limiter。
- **Tavily 受限域名拦截**：`tavily_extract` 命中配置的 restricted domains 时直接返回不可恢复工具错误，避免浪费调用。

## 答案生成链路

`graph/synthesizer.py` 中的合成阶段当前分为四步：

```text
evidence evaluator
  -> draft answer
  -> critical review
  -> final answer
```

阶段说明：

- **evidence evaluator**：判断当前证据能支持什么、不能支持什么、有哪些缺口或冲突。
- **draft answer**：基于证据评估和研究 findings 写第一版答案。
- **critical review**：检查草稿是否有 unsupported claim、citation issue、遗漏 caveat 等。
- **final answer**：根据 review 修正草稿，生成最终答案。

最终答案会经过：

- inline citation sanitizer：移除不存在的 `[N]`。
- citation marker：标记实际被引用的 sources。
- `Sources Used` 自动追加：只展示被最终答案引用的来源。
- simple answer evaluation：对最终答案本身做轻量质量检查。

## 简易最终答案评估

当前评估器位于 `rag/evaluation.py`，不调用 LLM，只检查最终答案本身：

| 维度 | 说明 |
|---|---|
| `citation_grounding` | `[N]` 是否都是有效来源编号 |
| `citation_coverage` | 最终答案覆盖了多少可用来源 |
| `query_focus` | 答案和 query 的关键词贴合度 |
| `answer_substance` | 答案正文是否过短、过空 |
| `internal_leakage` | 是否泄漏内部流程词，如 `evidence evaluator`、`critical review`、`tool call` |
| `overconfidence` | 是否出现未限定的绝对化表述，如 `always`、`never`、`proves`、`best`、`SOTA` |

评估结果会写入 `synthesize_answer` 的 node result，并在前端展示：

- `Simple quality score`
- `Quality flags`

## 配置与环境变量

复制环境变量模板：

```bash
cp .env.example .env
```

必填或常用变量：

| 变量 | 说明 |
|---|---|
| `DEEPSEEK_API_KEY` | LLM API key |
| `MODEL_NAME` | 默认模型名 |
| `POSTGRES_PASSWORD` | Postgres 密码 |
| `GITHUB_PERSONAL_ACCESS_TOKEN` | GitHub MCP token，建议 public-only / read-only |
| `TAVILY_API_KEY` | Tavily MCP API key |
| `PAPER_SEARCH_MCP_SEMANTIC_SCHOLAR_API_KEY` | Semantic Scholar key，可选 |
| `MCP_TOOL_TIMEOUT` | MCP 工具调用超时 |
| `MCP_ARXIV_MIN_INTERVAL_SECONDS` | arXiv 工具最小间隔，代码强制不低于 5 秒 |

不要提交 `.env`。

## 启动

启动全部服务：

```bash
docker compose up -d
```

只启动 MCP 服务：

```bash
docker compose up -d academic-mcp github-mcp web-mcp
```

查看状态：

```bash
docker compose ps
```

验证 MCP health 和 allowlist：

```bash
conda run -n research-agent python scripts/check_mcp_stack.py
```

停止服务：

```bash
docker compose stop
```
