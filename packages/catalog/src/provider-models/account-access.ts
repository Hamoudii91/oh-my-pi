import { resolveWireModelId } from "../model-thinking";
import type { Api, Model, ModelSpec } from "../types";

/**
 * Merge per-account discovery catalogs into one list, deduped by model id.
 *
 * The first catalog to list an id supplies its spec; {@link ModelSpec.accountAccess}
 * is merged from every catalog listing that id, so credential selection can
 * route account-gated models to the accounts that serve them. Used by
 * multi-account discovery for Codex and Antigravity, whose rosters differ per
 * account.
 */
export function unionAccountCatalogs<TApi extends Api>(
	catalogs: readonly (readonly ModelSpec<TApi>[])[],
): ModelSpec<TApi>[] {
	const byId = new Map<string, ModelSpec<TApi>>();
	for (const catalog of catalogs) {
		for (const model of catalog) {
			const existing = byId.get(model.id);
			if (!existing) {
				byId.set(model.id, model);
			} else if (model.accountAccess) {
				byId.set(model.id, {
					...existing,
					accountAccess: { ...existing.accountAccess, ...model.accountAccess },
				});
			}
		}
	}
	return [...byId.values()];
}

/**
 * Accounts serving the selected upstream wire ID. Codex catalogs identify
 * accounts by logical model alone; Antigravity catalogs also identify the
 * individual effort-tier members, which must be enforced rather than tried
 * on a sibling that cannot serve them.
 */
export function modelAccountRouting(
	model: Model<Api>,
	wireModelId?: string,
): { accountIds: string[]; strict: boolean } | undefined {
	const access = model.accountAccess;
	if (access === undefined) return undefined;
	let strict = false;
	for (const accountId in access) {
		if (access[accountId]?.wireModelIds !== undefined) {
			strict = true;
			break;
		}
	}
	if (!strict) return { accountIds: Object.keys(access), strict: false };
	const target = wireModelId ?? resolveWireModelId(model, undefined);
	const accountIds: string[] = [];
	for (const accountId in access) {
		if (access[accountId]?.wireModelIds?.includes(target)) accountIds.push(accountId);
	}
	return { accountIds, strict: true };
}
