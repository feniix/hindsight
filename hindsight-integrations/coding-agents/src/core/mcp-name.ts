/**
 * The name this package registers its MCP server under in every host's config (`mcpServers.<name>`,
 * `[mcp_servers.<name>]`, `claude mcp add <name>` ...).
 *
 * It defaults to "hindsight", which is also the name a user is most likely to have given a
 * Hindsight server of their OWN — e.g. one pointed at an application-support bank. Hosts key MCP
 * servers by name, so the two cannot coexist under the same one: the installer either refuses to
 * overwrite the foreign entry or (for CLI-managed hosts) replaces it. `mcpServerName` in
 * ~/.hindsight/coding-agent.json, set once with `install --mcp-name <name>`, moves ours aside.
 *
 * Machine-level on purpose: only the top-level config file is read, never a `harnesses.*`,
 * `banks.*` or `paths.*` layer — the registration is written once per host at install time, and a
 * per-repo name would need a per-repo registration that no host other than Trae has.
 */
import { readFileSync } from "node:fs";
import { CONFIG_PATH } from "./config";

export const DEFAULT_MCP_SERVER_NAME = "hindsight";

/** Safe as a bare TOML key, a JSON key, a `claude mcp add` argument, a Claude Code tool prefix
 *  (`mcp__<name>__...`) and inside the single-quoted SQL literal Trae's switch is keyed by. */
const MCP_SERVER_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;

export function isValidMcpServerName(value: unknown): value is string {
  return typeof value === "string" && MCP_SERVER_NAME_RE.test(value);
}

/** The name the config file asks for, or the default when unset or invalid. Never throws. */
export function configuredMcpServerName(configPath: string = CONFIG_PATH): string {
  try {
    const raw = JSON.parse(readFileSync(configPath, "utf8")) as { mcpServerName?: unknown };
    return isValidMcpServerName(raw.mcpServerName) ? raw.mcpServerName : DEFAULT_MCP_SERVER_NAME;
  } catch {
    return DEFAULT_MCP_SERVER_NAME;
  }
}
