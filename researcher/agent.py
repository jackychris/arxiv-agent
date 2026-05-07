# researcher/agent.py
import json
import llm
import memory.long_term as lt
from config import MAX_STEPS
from memory.short_term import ShortTermMemory
from prompts import build_system_prompt, build_memory_hint, TOOL_REFLECT_PROMPT
from mcp_client import MCPClient
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
        memories = lt.get_all()
        tool_memories = {k: v for k, v in memories.items() if k != "orchestrator"}
        memory_hint = build_memory_hint(tool_memories)

        memory = ShortTermMemory()
        memory.add("system", build_system_prompt(client.get_tools_description()))
        if memory_hint:
            memory.add("system", f"Past tool experience:\n{memory_hint}")
        memory.add("user", query)

        for step in range(MAX_STEPS):
            remaining = MAX_STEPS - step
            if remaining == 1:
                step_hint = "1 step remaining. You must now give your final_answer. Do not call any tools."
            elif remaining == 2:
                step_hint = "2 steps remaining. This is your last chance to call a tool. Use it wisely."
            else:
                step_hint = f"{remaining} steps remaining."
            temp = memory.get() + [{"role": "system", "content": step_hint}]

            response = await llm.chat(temp)
            memory.add("assistant", response)

            data = parse_response(response)
            if data is None:
                return f"Failed to parse model output:\n{response}"

            if verbose:
                print(f"\n--- Step {step + 1} ---")
                print(f"Thought: {data.get('thought', '')}")

            if "final_answer" in data:
                if verbose:
                    print(f"Final Answer: {data['final_answer']}")
                await self._reflect(query, memory)
                return data["final_answer"]

            action = data.get("action")
            action_input = data.get("action_input", {})
            if verbose:
                print(f"Action: {action}")
                print(f"Action Input: {action_input}")

            observation = await client.execute_tool(action, **action_input)
            if verbose:
                print(f"Observation: {observation[:200]}...")
            memory.add("user", f"Observation: {observation}")

        memory.add("system", "You have reached the maximum number of steps. Based on what you have found so far, give your best final_answer now. Do not call any tools.")
        response = await llm.chat(memory.get())
        await self._reflect(query, memory)
        data = parse_response(response)
        if data and "final_answer" in data:
            return f"[Max steps reached] {data['final_answer']}"
        return response

    async def _reflect(self, mission: str, memory: ShortTermMemory) -> None:
        history = build_reflection_history(memory)
        prompt = TOOL_REFLECT_PROMPT.format(mission=mission, history=history)
        try:
            response = await llm.chat([{"role": "user", "content": prompt}])
            data = json.loads(response)
            for tool, reflection in data.items():
                if reflection and reflection != "null":
                    lt.add(tool, reflection)
        except Exception:
            pass
