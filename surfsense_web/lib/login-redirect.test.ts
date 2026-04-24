import { describe, expect, it } from "vitest";
import { resolveLoginRedirect } from "./login-redirect";

describe("resolveLoginRedirect", () => {
	it("returns null when user is not authenticated", () => {
		expect(
			resolveLoginRedirect({ hasToken: false, returnUrl: null, storedRedirect: null })
		).toBeNull();
	});

	it("returns null when not authenticated even with a returnUrl", () => {
		expect(
			resolveLoginRedirect({
				hasToken: false,
				returnUrl: "/dashboard/spaces",
				storedRedirect: "/dashboard/other",
			})
		).toBeNull();
	});

	it("prefers returnUrl over storedRedirect and fallback", () => {
		expect(
			resolveLoginRedirect({
				hasToken: true,
				returnUrl: "/dashboard/spaces/42",
				storedRedirect: "/dashboard/other",
			})
		).toBe("/dashboard/spaces/42");
	});

	it("decodes URL-encoded returnUrl", () => {
		expect(
			resolveLoginRedirect({
				hasToken: true,
				returnUrl: "%2Fdashboard%2Fspaces%2F42%3Ftab%3Dchat",
				storedRedirect: null,
			})
		).toBe("/dashboard/spaces/42?tab=chat");
	});

	it("falls through to storedRedirect when returnUrl is malformed", () => {
		// "%" alone is not a valid URI-encoded sequence — decodeURIComponent throws.
		expect(
			resolveLoginRedirect({
				hasToken: true,
				returnUrl: "%E0%A4%A",
				storedRedirect: "/dashboard/other",
			})
		).toBe("/dashboard/other");
	});

	it("falls back to /dashboard when neither returnUrl nor storedRedirect is set", () => {
		expect(resolveLoginRedirect({ hasToken: true, returnUrl: null, storedRedirect: null })).toBe(
			"/dashboard"
		);
	});

	it("uses storedRedirect when returnUrl is empty string", () => {
		expect(
			resolveLoginRedirect({
				hasToken: true,
				returnUrl: "",
				storedRedirect: "/dashboard/saved",
			})
		).toBe("/dashboard/saved");
	});

	it("uses /dashboard fallback when returnUrl and storedRedirect are empty", () => {
		expect(resolveLoginRedirect({ hasToken: true, returnUrl: "", storedRedirect: "" })).toBe(
			"/dashboard"
		);
	});

	it("preserves already-decoded paths with no URL encoding", () => {
		expect(
			resolveLoginRedirect({
				hasToken: true,
				returnUrl: "/dashboard/space/1",
				storedRedirect: null,
			})
		).toBe("/dashboard/space/1");
	});
});
