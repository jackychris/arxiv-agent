# researcher/agent.py
import json

import llm
import memory.long_term as lt
from config import MAX_STEPS
from mcp_client import MCPClient
from memory.short_term import ShortTermMemory
from prompts import TOOL_REFLECT_PROMPT, build_memory_hint, build_system_prompt
from researcher.reflection import build_reflection_history


def parse_response(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


class ResearchAgent:
    async def run(self, query: str, verbose: bool = False) -> str:
        async with MCPClient() as client:
            return await self._run(query, client, verbose)

    async def _run(self, query: str, client: MCPClient, verbose: bool) -> str:
        memories = await lt.get_all()
        tool_memories = {k: v for k, v in memories.items() if k != "orchestrator"}
        memory_hint = build_memory_hint(tool_memories)

        memory = ShortTermMemory()
        memory.add("system", build_system_prompt(client.get_tools_description()))
        if memory_hint:
            memory.add("system", f"Past tool experience:\n{memory_hint}")
        memory.add("user", query)

        for step in range(MAX_STEPS):
            remaining = MAX_STEPS - step
            if remaining <= 2:
                step_hint = f"{remaining} step(s) remaining. Output {{\"done\": true}} if you have enough information, otherwise use your last tool call."
            else:
                step_hint = f"{remaining} steps remaining."
            temp = [*memory.get(), {"role": "system", "content": step_hint}]

            response = await llm.chat(temp)
            memory.add("assistant", response)

            data = parse_response(response)
            if data is None:
                return f"Failed to parse model output:\n{response}"

            if verbose:
                print(f"\n--- Step {step + 1} ---")
                print(f"Thought: {data.get('thought', '')}")

            if data.get("done"):
                break

            action: str = str(data.get("action") or "")
            action_input = data.get("action_input", {})
            if verbose:
                print(f"Action: {action}")
                print(f"Action Input: {action_input}")

            observation = await client.execute_tool(action, **action_input)
            if verbose:
                print(f"Observation: {observation[:200]}...")
            memory.add("user", f"Observation: {observation}")

        await self._reflect(query, memory)
        return build_reflection_history(memory)

    async def _reflect(self, mission: str, memory: ShortTermMemory) -> None:
        history = build_reflection_history(memory)
        prompt = TOOL_REFLECT_PROMPT.format(mission=mission, history=history)
        try:
            response = await llm.chat([{"role": "user", "content": prompt}])
            data = json.loads(response)
            for tool, reflection in data.items():
                if reflection and reflection != "null":
                    await lt.add(tool, reflection)
        except Exception:
            pass
