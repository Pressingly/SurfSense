"use client";

import { useEffect } from "react";

/**
 * SSO-only register route.
 *
 * This fork is configured for mPass/Cognito SSO via oauth2-proxy. The local
 * email/password registration form is dead UI in this deployment — user
 * provisioning happens at Cognito, not at the app, and the SurfSense backend
 * JIT-creates a local user record on first SSO login (see proxy_login in
 * surfsense_backend/app/routes/auth_routes.py).
 *
 * Anything that links to /register lands here and gets bounced into the
 * OIDC flow at the dedicated auth subdomain — same pattern as /login.
 *
 * If/when this fork needs to support local registration again, restore the
 * full RegisterPage component from git history.
 */
export default function RegisterPage() {
	useEffect(() => {
		if (typeof window === "undefined") return;

		// Bounce straight to oauth2-proxy /oauth2/sign_in. The dedicated auth
		// subdomain handles the OIDC dance with Cognito and returns the user
		// to `rd=` on success.
		const oauthProxyUrl = process.env.NEXT_PUBLIC_OAUTH2_PROXY_URL || window.location.origin;
		// On a /register visit there's no useful "return to here" URL — drop
		// the user back at / so the home-route splash + cookie handoff
		// finishes the login normally.
		const rd = `${window.location.origin}/`;
		window.location.replace(`${oauthProxyUrl}/oauth2/sign_in?rd=${encodeURIComponent(rd)}`);
	}, []);

	// Splash — neutral background, no UI flash during the redirect.
	return <div className="min-h-screen bg-gray-50 dark:bg-black" />;
}
