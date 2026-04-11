"use client";

import { useEffect } from "react";

/**
 * SSO-only login route.
 *
 * This fork is configured for mPass/Cognito SSO via oauth2-proxy. The local
 * login form, Google OAuth button, and registration flows that the upstream
 * /login page renders are all dead UI in this deployment — every visitor
 * either has a session or needs to start one through oauth2-proxy. Anything
 * that links to /login (the navbar Sign-in button, hero section CTAs, the
 * legacy `handleUnauthorized` fallback, or a user manually typing /login in
 * the address bar) lands here and gets bounced into the OIDC flow at the
 * dedicated auth subdomain.
 *
 * If/when this fork needs to support local or Google auth again, restore
 * the full LoginContent component from git history.
 */
export default function LoginPage() {
	useEffect(() => {
		if (typeof window === "undefined") return;

		// Bounce straight to oauth2-proxy /oauth2/sign_in. The dedicated auth
		// subdomain handles the OIDC dance with Cognito and returns the user
		// to `rd=` on success — same pattern as handleUnauthorized() in
		// lib/auth-utils.ts, kept consistent on purpose.
		const oauthProxyUrl = process.env.NEXT_PUBLIC_OAUTH2_PROXY_URL || window.location.origin;
		// On a /login visit there's no useful "return to here" URL — drop the
		// user back at / so the home-route splash + cookie handoff finishes
		// the login normally.
		const rd = `${window.location.origin}/`;
		window.location.replace(`${oauthProxyUrl}/oauth2/sign_in?rd=${encodeURIComponent(rd)}`);
	}, []);

	// Splash — neutral background, no UI flash during the redirect.
	return <div className="min-h-screen bg-gray-50 dark:bg-black" />;
}
