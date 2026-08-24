import { describe, expect, it } from "vitest";
import { type PiMessage, readPiMessages } from "./transcript-pi";

describe("readPiMessages", () => {
  it("keeps user/assistant text + compact action turns; drops other roles/blocks and tool args", () => {
    const messages: PiMessage[] = [
      // non-conversational role: dropped
      { role: "system", content: [{ type: "text", text: "you are pi" }] },
      // string content is accepted as a prose turn
      { role: "user", content: "add retry backoff to the uploader" },
      // assistant message: text + a toolCall (args NOT retained) + a dropped reasoning block
      {
        role: "assistant",
        content: [
          { type: "reasoning", text: "thinking…" },
          { type: "text", text: "I'll add exponential backoff." },
          { type: "toolCall", name: "bash", arguments: { command: "npm test" } },
        ],
      },
      // assistant message with only a toolCall: just the compact action line
      {
        role: "assistant",
        content: [{ type: "toolCall", name: "read", arguments: { path: "nope.ts" } }],
      },
    ];

    expect(readPiMessages(messages)).toEqual([
      { role: "user", content: "add retry backoff to the uploader" },
      { role: "assistant", content: "I'll add exponential backoff." },
      { role: "action", content: "bash npm test" },
      { role: "action", content: "read nope.ts" },
    ]);
  });

  it("strips injected memory that leaks into a kept message", () => {
    const messages: PiMessage[] = [
      {
        role: "user",
        content: [
          { type: "text", text: "<hindsight_memories>\nleak\n</hindsight_memories>\nWhy retry?" },
        ],
      },
    ];
    expect(readPiMessages(messages)).toEqual([{ role: "user", content: "Why retry?" }]);
  });

  it("never throws on malformed entries", () => {
    const messages = [
      null,
      {},
      { role: "user" },
      { role: "assistant", content: [null, 3, "x"] },
    ] as unknown as PiMessage[];
    expect(() => readPiMessages(messages)).not.toThrow();
    expect(readPiMessages(messages)).toEqual([]);
  });
});
