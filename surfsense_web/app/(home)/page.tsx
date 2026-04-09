"use client";

import dynamic from "next/dynamic";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { HeroSection } from "@/components/homepage/hero-section";
import { getBearerToken, getSSOCookieTokens, clearSSOCookies, setBearerToken, setRefreshToken } from "@/lib/auth-utils";

const FeaturesCards = dynamic(
	() => import("@/components/homepage/features-card").then((m) => ({ default: m.FeaturesCards })),
	{ ssr: false }
);

const FeaturesBentoGrid = dynamic(
	() =>
		import("@/components/homepage/features-bento-grid").then((m) => ({
			default: m.FeaturesBentoGrid,
		})),
	{ ssr: false }
);

const ExternalIntegrations = dynamic(() => import("@/components/homepage/integrations"), {
	ssr: false,
});

const CTAHomepage = dynamic(
	() => import("@/components/homepage/cta").then((m) => ({ default: m.CTAHomepage })),
	{ ssr: false }
);

export default function HomePage() {
	const router = useRouter();

	useEffect(() => {
		if (getBearerToken()) {
			router.replace("/dashboard");
			return;
		}

		// Check for SSO handoff cookies set by /auth/jwt/proxy-login after Cognito login.
		// The backend sets short-lived cookies (60s TTL) and redirects here instead of
		// to /auth/callback, avoiding any Traefik path-split between frontend and backend.
		const { token, refreshToken } = getSSOCookieTokens();
		if (token) {
			setBearerToken(token);
			if (refreshToken) setRefreshToken(refreshToken);
			clearSSOCookies();
			router.replace("/dashboard");
			return;
		}

		// No JWT anywhere — trigger SSO flow.
		window.location.href = `${process.env.NEXT_PUBLIC_FASTAPI_BACKEND_URL}/auth/jwt/proxy-login`;
	}, [router]);

	return (
		<main className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 text-gray-900 dark:from-black dark:to-gray-900 dark:text-white">
			<HeroSection />
			<FeaturesCards />
			<FeaturesBentoGrid />
			<ExternalIntegrations />
			<CTAHomepage />
		</main>
	);
}
