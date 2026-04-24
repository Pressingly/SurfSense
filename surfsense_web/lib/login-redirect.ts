/**
 * Pure decision helper for the /login page.
 *
 * Given an authenticated user and the URL/storage state, returns the path the
 * user should be redirected to, or null if no redirect should happen.
 *
 * Precedence: `?returnUrl=` param → stored `surfsense_redirect_path` → `/dashboard`.
 */
export interface LoginRedirectInput {
	hasToken: boolean;
	returnUrl: string | null;
	storedRedirect: string | null;
}

export function resolveLoginRedirect(input: LoginRedirectInput): string | null {
	if (!input.hasToken) return null;

	if (input.returnUrl) {
		try {
			return decodeURIComponent(input.returnUrl);
		} catch {
			// Malformed URI component — fall through to stored/fallback.
		}
	}

	return input.storedRedirect || "/dashboard";
}
