# researcher/server.py
import sys

from a2a.server.request_handlers import DefaultRequestHandlerV2
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types.a2a_pb2 import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from a2a.utils.constants import TransportProtocol
from starlette.applications import Starlette

from config import PORT_RESEARCH_AGENT, RESEARCH_AGENT_URL
from researcher.executor import ResearchAgentExecutor

AGENT_CARD_PATH = "/.well-known/agent-card.json"
_agent_card = AgentCard(
    name="Research Agent",
    description="Searches arxiv, GitHub, and the web to answer research questions.",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=True),
    supported_interfaces=[
        AgentInterface(
            protocol_binding=TransportProtocol.JSONRPC,
            protocol_version="1.0",
            url=f"{RESEARCH_AGENT_URL}/",
        )
    ],
    skills=[
        AgentSkill(
            id="research",
            name="Research",
            description="Given a research mission, searches for papers, repos, and web resources.",
        )
    ],
)


def get_agent_card() -> AgentCard:
    return _agent_card


def build_app() -> Starlette:
    task_store = InMemoryTaskStore()
    executor = ResearchAgentExecutor()
    handler = DefaultRequestHandlerV2(
        agent_executor=executor,
        task_store=task_store,
        agent_card=_agent_card,
    )
    routes = create_agent_card_routes(_agent_card, card_url=AGENT_CARD_PATH) + create_jsonrpc_routes(handler, rpc_url="/")
    return Starlette(routes=routes)


def run_http(port: int = PORT_RESEARCH_AGENT):
    import uvicorn

    uvicorn.run(build_app(), host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    run_http()
