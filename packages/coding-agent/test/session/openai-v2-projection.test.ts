import { expect, test } from "bun:test";
import { Agent, type AgentMessage } from "@oh-my-pi/pi-agent-core";
import {
	compact,
	createCompactionSummaryMessage,
	createFileOps,
	DEFAULT_COMPACTION_SETTINGS,
	defaultConvertToLlm,
} from "@oh-my-pi/pi-agent-core/compaction";
import { getCompactionV2PreserveData } from "@oh-my-pi/pi-agent-core/compaction/openai";
import type { Context, ImageContent, Model, TextContent } from "@oh-my-pi/pi-ai";
import { AssistantMessageEventStream } from "@oh-my-pi/pi-ai/utils/event-stream";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import { DateCwdReminderInjector, renderDateCwdReminder } from "@oh-my-pi/pi-coding-agent/session/date-cwd-reminder";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { SessionProviderBoundary } from "@oh-my-pi/pi-coding-agent/session/session-provider-boundary";
import { isRecord, TempDir } from "@oh-my-pi/pi-utils";
import { createAssistantMessage } from "../helpers/agent-session-setup";

/** The reminder as it appears inside serialized messages. */
const REMINDER = JSON.stringify(renderDateCwdReminder("2026-10-08", "/repo")).slice(1, -1);
const IMAGE: ImageContent = { type: "image", data: "iVBORw0KGgo=", mimeType: "image/png" };
const REWRITTEN: TextContent = { type: "text", text: "[provider-rewritten image]" };

/** Stands in for the production image pipeline: every image becomes a marker. */
function rewriteImages(context: Context): Context {
	return {
		...context,
		messages: context.messages.map(message =>
			(message.role === "toolResult" || message.role === "user") && Array.isArray(message.content)
				? { ...message, content: message.content.map(part => (part.type === "image" ? REWRITTEN : part)) }
				: message,
		),
	};
}

/**
 * Agent plus provider boundary sharing one stateful `context` transform that
 * reorders history and counts its invocations, and the production-shaped
 * provider transform chain: image rewriting, then the date/cwd reminder.
 */
async function createProjectionFixture(tempDir: TempDir, model: Model, messages: AgentMessage[]) {
	const authStorage = await AuthStorage.create(tempDir.join("auth.db"));
	const state = { calls: 0 };
	const transformContext = async (input: AgentMessage[]) => {
		state.calls++;
		const inserted: AgentMessage = { role: "user", content: `extension insertion ${state.calls}`, timestamp: 0 };
		return [...input.slice().reverse(), inserted];
	};
	const reminder = new DateCwdReminderInjector();
	const agent = new Agent({
		initialState: { model, systemPrompt: ["session instructions"], messages, tools: [] },
		transformContext,
		convertToLlm: defaultConvertToLlm,
		transformProviderContext: context => reminder.transform(rewriteImages(context), "2026-10-08", "/repo"),
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

test("native V2 reuses the pinned projection and provider-prepares a grown tail", async () => {
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
		content: [{ type: "text", text: "file contents" }, IMAGE],
		isError: false,
		timestamp: 10,
	};
	const prepared = await boundary.prepareOpenAiV2Request([...source, call, tail], [], model, pinned);
	expect(state.calls).toBe(afterNewTurn);
	expect(prepared?.context.systemPrompt).toEqual(pinned.context.systemPrompt);
	expect(prepared?.context.messages.slice(0, -2)).toEqual(firstProjection);
	expect(prepared?.context.messages.at(-2)).toMatchObject({ content: call.content });
	expect(prepared?.context.messages.at(-1)).toMatchObject({
		role: "toolResult",
		content: [{ type: "text", text: "file contents" }, REWRITTEN],
	});
	expect(JSON.stringify(firstProjection)).toContain("extension insertion 1");
});

test("retained user turns come from the prepared request, without the date/cwd reminder", async () => {
	using tempDir = TempDir.createSync("@omp-v2-projection-retained-");
	const model = getBundledModel("openai", "gpt-4o-mini");
	const withImage: AgentMessage = {
		role: "user",
		content: [{ type: "text", text: "look at this" }, IMAGE],
		timestamp: 1,
	};
	const { agent, boundary } = await createProjectionFixture(tempDir, model, [withImage]);
	await agent.prompt("current");
	const pinned = agent.lastPreparedProviderCall;
	if (!pinned) throw new Error("ordinary request did not complete preparation");
	const current = pinned.source[1];
	if (!current) throw new Error("prompt was not recorded");
	// The reordering handler makes "current" the first user turn, so it carries the reminder on the wire.
	expect(JSON.stringify(pinned.context.messages[0])).toContain(REMINDER);

	const prepared = await boundary.prepareOpenAiV2Request(
		pinned.source.slice(0, pinned.sourceLength),
		[withImage, current],
		model,
		pinned,
	);
	expect(prepared?.retained).toHaveLength(2);
	expect(prepared?.retained[0]).toMatchObject({ content: [{ type: "text", text: "look at this" }, REWRITTEN] });
	expect(JSON.stringify(prepared?.retained[1])).toContain("current");
	expect(JSON.stringify(prepared?.retained)).not.toContain(REMINDER);
	expect(JSON.stringify(prepared?.retained)).not.toContain(IMAGE.data);
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
			{ role: "user", content: [{ type: "text", text: "after native compaction" }, IMAGE], timestamp: 2 },
		]);
		await agent.prompt("ordinary turn after compaction");
		const pinned = agent.lastPreparedProviderCall;
		if (!pinned) throw new Error("ordinary request did not complete preparation");
		const projection = pin ? pinned : undefined;

		let body: { input: unknown[] } | undefined;
		const result = await compact(
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
				prepareOpenAiV2Request: (messages, retained, requestModel, signal) =>
					boundary.prepareOpenAiV2Request(messages, retained, requestModel, projection, signal),
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
		expect(text).not.toContain(IMAGE.data);
		expect(input.filter(item => item.encrypted_content === "first-native-compaction")).toEqual([
			{ type: "compaction", encrypted_content: "first-native-compaction" },
		]);
		expect(input.at(-1)).toEqual({ type: "compaction_trigger" });
		// Replacement history replays on the next ordinary turn: the retained user
		// turn is the provider-prepared one, not the raw image.
		const replacement = JSON.stringify(getCompactionV2PreserveData(result.preserveData)?.replacementHistory);
		expect(replacement).toContain("after native compaction");
		expect(replacement).toContain(REWRITTEN.text);
		expect(replacement).not.toContain(IMAGE.data);
		expect(replacement).not.toContain(REMINDER);
	},
);
