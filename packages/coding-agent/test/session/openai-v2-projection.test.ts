import { expect, test } from "bun:test";
import { Agent, type AgentMessage } from "@oh-my-pi/pi-agent-core";
import {
	compact,
	createCompactionSummaryMessage,
	createFileOps,
	DEFAULT_COMPACTION_SETTINGS,
	defaultConvertToLlm,
} from "@oh-my-pi/pi-agent-core/compaction";
import type { Model } from "@oh-my-pi/pi-ai";
import { AssistantMessageEventStream } from "@oh-my-pi/pi-ai/utils/event-stream";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { SessionProviderBoundary } from "@oh-my-pi/pi-coding-agent/session/session-provider-boundary";
import { isRecord, TempDir } from "@oh-my-pi/pi-utils";
import { createAssistantMessage } from "../helpers/agent-session-setup";

/**
 * Agent plus provider boundary sharing one stateful `context` transform that counts
 * its invocations, and a provider transform that rewrites images as the
 * production image pipeline does.
 */
async function createProjectionFixture(tempDir: TempDir, model: Model, messages: AgentMessage[]) {
	const authStorage = await AuthStorage.create(tempDir.join("auth.db"));
	const state = { calls: 0 };
	const transformContext = async (input: AgentMessage[]) => {
		state.calls++;
		const injected: AgentMessage[] = [
			{ role: "user", content: `date/cwd reminder ${state.calls}`, timestamp: 0 },
			{ role: "user", content: `extension insertion ${state.calls}`, timestamp: 0 },
		];
		return [...injected, ...input.slice().reverse()];
	};
	const agent = new Agent({
		initialState: { model, systemPrompt: ["session instructions"], messages, tools: [] },
		transformContext,
		convertToLlm: defaultConvertToLlm,
		transformProviderContext: context => ({
			...context,
			messages: context.messages.map(message =>
				message.role === "toolResult"
					? {
							...message,
							content: message.content.map(part =>
								part.type === "image" ? { type: "text" as const, text: "[provider-rewritten image]" } : part,
							),
						}
					: message,
			),
		}),
		streamFn: () => {
			const stream = new AssistantMessageEventStream();
			queueMicrotask(() => stream.push({ type: "done", reason: "stop", message: createAssistantMessage("reply") }));
			return stream;
		},
	});
	const boundary = new SessionProviderBoundary({
		agent,
		sessionManager: SessionManager.inMemory(),
		settings: Settings.isolated({ "compaction.enabled": false }),
		modelRegistry: new ModelRegistry(authStorage, tempDir.join("models.yml")),
		model: () => model,
		sessionId: () => "projection-test",
		localProtocolOptions: () => ({}),
		transformContext,
		convertToLlm: defaultConvertToLlm,
		onPayload: undefined,
		onResponse: undefined,
		onSseEvent: undefined,
		obfuscator: () => undefined,
	});
	return { agent, boundary, state };
}

test("native V2 reuses the pinned stateful context projection after another turn", async () => {
	using tempDir = TempDir.createSync("@omp-v2-projection-");
	const model = getBundledModel("openai", "gpt-4o-mini");
	const { agent, boundary, state } = await createProjectionFixture(tempDir, model, [
		{ role: "user", content: "older", timestamp: 1 },
	]);
	await agent.prompt("current");
	const pinned = agent.lastPreparedProviderCall;
	if (!pinned) throw new Error("ordinary request did not complete preparation");
	const source = pinned.source.slice(0, pinned.sourceLength);
	const firstProjection = pinned.context.messages;
	await agent.prompt("newer ordinary turn");
	const afterNewTurn = state.calls;
	const call = createAssistantMessage("reading");
	call.content = [{ type: "toolCall", id: "call-read", name: "read", arguments: { path: "file.txt" } }];
	const tail: AgentMessage = {
		role: "toolResult",
		toolCallId: "call-read",
		toolName: "read",
		content: [
			{ type: "text", text: "file contents" },
			{ type: "image", data: "iVBORw0KGgo=", mimeType: "image/png" },
		],
		isError: false,
		timestamp: 10,
	};
	const prepared = await boundary.buildOpenAiV2Context([...source, call, tail], model, pinned);
	expect(state.calls).toBe(afterNewTurn);
	expect(prepared?.systemPrompt).toEqual(pinned.context.systemPrompt);
	expect(prepared?.messages.slice(0, -2)).toEqual(firstProjection);
	expect(prepared?.messages.at(-2)).toMatchObject({ content: call.content });
	expect(prepared?.messages.at(-1)).toMatchObject({
		role: "toolResult",
		content: [
			{ type: "text", text: "file contents" },
			{ type: "text", text: "[provider-rewritten image]" },
		],
	});
	expect(prepared?.messages[0]).toMatchObject({ content: "date/cwd reminder 1" });
	expect(prepared?.messages[1]).toMatchObject({ content: "extension insertion 1" });
});

test.each([
	{ name: "pinned projection", pin: true, expectedCalls: 1, insertion: "extension insertion 1" },
	{ name: "re-prepared fallback", pin: false, expectedCalls: 2, insertion: "extension insertion 2" },
])(
	"a second native V2 compaction replays the native summary once ($name)",
	async ({ pin, expectedCalls, insertion }) => {
		using tempDir = TempDir.createSync("@omp-v2-projection-native-");
		const model: Model = {
			...getBundledModel("openai", "gpt-4o-mini"),
			remoteCompaction: { enabled: true, v2StreamingEnabled: true },
		};
		const replacementHistory = [
			{ type: "message", role: "user", content: [{ type: "input_text", text: "retained user" }] },
			{ type: "compaction", id: "cmp_first", encrypted_content: "first-native-compaction" },
		];
		const summaryTimestamp = "2026-10-08T00:00:00.000Z";
		const nativeSummary = createCompactionSummaryMessage("first compaction", 100, summaryTimestamp, {
			providerPayload: { type: "openaiResponsesHistory", provider: model.provider, items: replacementHistory },
		});
		const { agent, boundary, state } = await createProjectionFixture(tempDir, model, [
			nativeSummary,
			{ role: "user", content: "after native compaction", timestamp: 2 },
		]);
		await agent.prompt("ordinary turn after compaction");
		const pinned = agent.lastPreparedProviderCall;
		if (!pinned) throw new Error("ordinary request did not complete preparation");
		const projection = pin ? pinned : undefined;

		let body: { input: unknown[] } | undefined;
		await compact(
			{
				firstKeptEntryId: "kept-1",
				messagesToSummarize: agent.state.messages.slice(1),
				turnPrefixMessages: [],
				recentMessages: [],
				isSplitTurn: false,
				tokensBefore: 1_000,
				previousSummary: nativeSummary.summary,
				previousSummaryTimestamp: summaryTimestamp,
				previousPreserveData: {
					openaiRemoteCompaction: { version: "v2", provider: model.provider, replacementHistory },
				},
				fileOps: createFileOps(),
				settings: { ...DEFAULT_COMPACTION_SETTINGS, remoteStreamingV2Enabled: true },
			},
			model,
			"test-key",
			undefined,
			undefined,
			{
				buildOpenAiV2Context: (messages, requestModel, signal) =>
					boundary.buildOpenAiV2Context(messages, requestModel, projection, signal),
				fetch: async (_input, init) => {
					body = JSON.parse(String(init?.body));
					const events = [
						{
							type: "response.output_item.done",
							output_index: 0,
							item: { type: "compaction", encrypted_content: "second-native-compaction" },
						},
						{ type: "response.completed", response: { usage: { input_tokens: 10 } } },
					];
					return new Response(
						events.map(event => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join(""),
						{
							headers: { "content-type": "text/event-stream" },
						},
					);
				},
			},
		);

		const input = (body?.input ?? []).filter(isRecord);
		const text = JSON.stringify(input);
		expect(state.calls).toBe(expectedCalls);
		expect(text).toContain(insertion);
		expect(text).toContain("reply");
		expect(input.filter(item => item.encrypted_content === "first-native-compaction")).toEqual([
			{ type: "compaction", encrypted_content: "first-native-compaction" },
		]);
		expect(input.at(-1)).toEqual({ type: "compaction_trigger" });
	},
);
