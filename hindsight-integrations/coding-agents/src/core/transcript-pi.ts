/**
 * pi live-transcript normalizer — shared by pi and its fork Prime Agent.
 *
 * These hosts hand an extension the completed exchange as an in-memory message list on the
 * `agent_end` event (not a JSONL file like Claude/Codex), so this is a pure function over that list,
 * mirroring transcript-opencode.ts. It produces the same rich `TransportTurn[]` shape (prose turns +
 * compact `role:"action"` tool turns) and reuses the shared `stripInjectedMemory`/`actionLine`
 * helpers so a retain never feeds injected memory back into recall and tool noise stays out of the
 * bank.
 */
import type { TransportTurn } from "./chat";
import { actionLine, stripInjectedMemory } from "./transcript-util";

/** Structural subset of a pi message content block (TextContent | ToolCall | dropped). */
export interface PiBlock {
  type?: string;
  text?: string; // text block
  name?: string; // toolCall block: the tool name
  arguments?: unknown; // toolCall block: the call input
}

/** Structural subset of a pi message ({ role, content }). */
export interface PiMessage {
  role?: string;
  content?: unknown; // string | PiBlock[]
}

/**
 * Render one pi message into turns. Text (string content or text blocks) joins into one
 * prose turn (injected-memory stripped); each `toolCall` block becomes its own compact
 * `role:"action"` turn (tool name + primary target via `actionLine` — no args, no output). Other
 * block types and non-conversational roles are dropped.
 */
function renderMessage(m: PiMessage): TransportTurn[] {
  if (!m || typeof m !== "object") return [];
  const role = m.role;
  if (role !== "user" && role !== "assistant") return [];

  const texts: string[] = [];
  const actions: TransportTurn[] = [];

  if (typeof m.content === "string") {
    const t = stripInjectedMemory(m.content).trim();
    if (t) texts.push(t);
  } else if (Array.isArray(m.content)) {
    for (const part of m.content) {
      if (!part || typeof part !== "object") continue;
      const block = part as PiBlock;
      if (block.type === "text" && typeof block.text === "string") {
        const t = stripInjectedMemory(block.text).trim();
        if (t) texts.push(t);
      } else if (block.type === "toolCall" && typeof block.name === "string") {
        actions.push({ role: "action", content: actionLine(block.name, block.arguments) });
      }
    }
  }

  const out: TransportTurn[] = [];
  const joined = texts.join("\n").trim();
  if (joined) out.push({ role, content: joined });
  out.push(...actions);
  return out;
}

/**
 * Normalize a pi `agent_end` message list into transcript turns (user/assistant prose plus
 * compact action turns for tool calls). Never throws on malformed entries.
 */
export function readPiMessages(messages: readonly PiMessage[]): TransportTurn[] {
  return (messages || []).flatMap((m) => renderMessage(m));
}
