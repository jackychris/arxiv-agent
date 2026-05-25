# Legacy helper tools

The runtime MCP path no longer starts the in-repo `tools.*.server` modules.
Production MCP services are vendored under `third_party/mcp/` and managed by
Docker Compose.

These helper modules are kept for tests, historical scripts, and possible local
debugging. Do not add new runtime dependencies on `python -m tools.*.server`
without first deciding whether the vendored MCP service should own that behavior.
