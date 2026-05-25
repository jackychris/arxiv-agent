#!/usr/bin/env python3
"""Validate the local MCP stack health and tool allowlists."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "mcp_config.json"
DEFAULT_TIMEOUT = 20.0


@dataclass
class CheckResult:
    server: str
    ok: bool
    message: str


def _health_url(mcp_url: str) -> str:
    parts = urlsplit(mcp_url)
    path = parts.path.rstrip("/")
    if path.endswith("/mcp"):
        path = path[: -len("/mcp")]
    return urlunsplit((parts.scheme, parts.netloc, f"{path}/healthz", "", ""))


def _load_servers(selected: set[str] | None) -> dict:
    with CONFIG_PATH.open() as f:
        servers = json.load(f)["servers"]
    if selected:
        unknown = selected - set(servers)
        if unknown:
            raise SystemExit(f"Unknown MCP server(s): {', '.join(sorted(unknown))}")
        servers = {name: cfg for name, cfg in servers.items() if name in selected}
    return servers


async def _check_health(server: str, url: str, timeout: float) -> CheckResult:
    health_url = _health_url(url)
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=timeout) as client:
            response = await client.get(health_url)
        if response.is_success:
            return CheckResult(server, True, f"health ok: {health_url}")
        return CheckResult(
            server,
            False,
            f"health failed: {health_url} returned HTTP {response.status_code}",
        )
    except Exception as exc:
        return CheckResult(server, False, f"health failed: {health_url}: {exc}")


async def _list_tools(url: str, timeout: float) -> list[str]:
    async with httpx.AsyncClient(trust_env=False, timeout=timeout) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), timeout=timeout)
                result = await asyncio.wait_for(session.list_tools(), timeout=timeout)
                return sorted(tool.name for tool in result.tools)


async def _check_tools(server: str, cfg: dict, timeout: float, exact: bool) -> CheckResult:
    url = cfg["url"]
    expected = sorted(cfg.get("allowed_tools") or [])
    try:
        actual = await _list_tools(url, timeout)
    except Exception as exc:
        return CheckResult(server, False, f"list_tools failed: {url}: {exc}")

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing:
        return CheckResult(
            server,
            False,
            f"missing required tools: {missing}; actual tools: {actual}",
        )
    if exact and extra:
        return CheckResult(
            server,
            False,
            f"unexpected tools in exact mode: {extra}; actual tools: {actual}",
        )
    return CheckResult(server, True, f"tools ok: {actual}")


async def _run(args: argparse.Namespace) -> int:
    selected = set(args.server or []) or None
    servers = _load_servers(selected)
    failures = 0

    for name, cfg in servers.items():
        if not args.skip_health:
            result = await _check_health(name, cfg["url"], args.timeout)
            print(("OK " if result.ok else "ERR ") + f"{name}: {result.message}")
            failures += 0 if result.ok else 1

        result = await _check_tools(name, cfg, args.timeout, args.exact)
        print(("OK " if result.ok else "ERR ") + f"{name}: {result.message}")
        failures += 0 if result.ok else 1

    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server",
        action="append",
        choices=["academic", "github", "web"],
        help="Check only this server. Repeat to check multiple servers.",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--skip-health", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Deprecated alias kept for old commands; allowlist checks are always strict.",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Fail if the raw upstream server exposes tools beyond the configured allowlist.",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
