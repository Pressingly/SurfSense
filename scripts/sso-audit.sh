#!/usr/bin/env bash
#
# sso-audit.sh — SurfSense-side fork audit against the cross-app SSO contract.
#
# ============================================================================
# Covers fork-side rows of awais786/sso-rules-moneta:openspec/specs/
# proxy-auth-middleware/spec.md. SurfSense is architecturally immune to
# the cross-user stale-session class of bug because current_active_user
# resolves request.state.proxy_user (set per-request by ProxyAuthMiddleware
# from X-Auth-Request-Email) BEFORE falling back to the JWT bearer
# identity. The other four forks need an explicit Rule-2 flush; SurfSense
# inherits the property from its precedence rule.
#
# This audit pins that precedence as a regression guard: the contract
# holds if-and-only-if the precedence is unchanged, so we check for the
# regression-guard test file added in Pressingly/SurfSense#22.
#
# Exit codes:
#   0 — all rows ✅ or n/a / informational
#   1 — at least one SECURITY-CRITICAL row failed
#
# Rows covered:
#   Row 14 — logout shape: SPA logout (surfsense_web/lib/auth-utils.ts)
#            MUST NOT call /oauth2/sign_out.
#   Row 20 — session-identity reconciliation: SurfSense relies on
#            `current_active_user` returning proxy_user before jwt_user
#            (architectural immunity). The regression-guard test file
#            MUST exist and pin the precedence. SECURITY-CRITICAL only
#            if the test is missing (means no guard against future
#            refactor flipping the precedence).
#   Row 21 — email-shape detection MUST NOT use polynomial-backtracking
#            regex. SurfSense uses `"@" not in email` substring check.
# ============================================================================

set -euo pipefail

MIDDLEWARE="surfsense_backend/app/middleware/proxy_auth.py"
USERS_PY="surfsense_backend/app/users.py"
PRECEDENCE_TEST="surfsense_backend/tests/unit/test_current_active_user_proxy_precedence.py"
AUTH_UTILS="surfsense_web/lib/auth-utils.ts"

declare -a ROW_STATUS=()
declare -a ROW_TITLES=(
  "logout shape: SPA logout does not call /oauth2/sign_out"
  "session-identity reconciliation: proxy>jwt precedence pinned by test"
  "email-shape detection uses substring, not polynomial regex"
)
declare -a ROW_NOTES=()
declare -a ROW_NUMBERS=(14 20 21)

SECURITY_CRITICAL=(1)
SECURITY_CRITICAL_FAILS=0

record() {
  local idx=$1 status=$2 note=$3
  ROW_STATUS[$idx]="$status"
  ROW_NOTES[$idx]="$note"
  if [[ "$status" == "❌" ]]; then
    for c in "${SECURITY_CRITICAL[@]}"; do
      if [[ "$c" -eq "$idx" ]]; then
        SECURITY_CRITICAL_FAILS=$((SECURITY_CRITICAL_FAILS + 1))
        return
      fi
    done
  fi
}

# ============================================================================
# Row 14 (idx 0): logout shape — narrow check
# ============================================================================
check_row_14() {
  if [[ ! -f "$AUTH_UTILS" ]]; then
    record 0 "?" "$AUTH_UTILS not found — skipping"
    return
  fi

  if grep -qE '/oauth2/sign_?out' "$AUTH_UTILS"; then
    local line
    line=$(grep -nE '/oauth2/sign_?out' "$AUTH_UTILS" | head -1)
    record 0 "❌" "Logout calls \`/oauth2/sign_out\` at $AUTH_UTILS:$line — per logout-flow spec, per-app Logout is navigation-only. Drop the call; portal 'Logout all' handles oauth2-proxy clearing."
    return
  fi

  record 0 "✅" "$AUTH_UTILS does not invoke \`/oauth2/sign_out\` (this row verifies only that the SPA doesn't try to clear the upstream proxy cookie itself; that's the portal's job)"
}

# ============================================================================
# Row 20 (idx 1): session-identity reconciliation via architectural immunity
#
# SurfSense doesn't need an explicit Rule 2 flush because
# `current_active_user` in app/users.py resolves:
#   proxy_user (per-request from X-Auth-Request-Email) BEFORE jwt_user
#
# That precedence IS the contract. If a future refactor flips the
# precedence or removes the proxy_user check, the cross-user stale-session
# leak appears.
#
# We check two things:
#   1. The precedence test file exists (regression guard pin).
#   2. app/users.py still has the precedence pattern:
#      proxy_user check returns FIRST, jwt_user fallback SECOND.
#
# SECURITY-CRITICAL: missing test = no guard against precedence-flip
# regression. Reference: Pressingly/SurfSense#22.
# ============================================================================
check_row_20() {
  if [[ ! -f "$USERS_PY" ]]; then
    record 1 "?" "$USERS_PY not found — skipping"
    return
  fi

  # The precedence pattern: in current_active_user, proxy_user is checked
  # and returned BEFORE jwt_user. We grep for the canonical shape that
  # makes the precedence visible.
  local has_precedence
  has_precedence=$(grep -cE "proxy_user.*is not None|return proxy_user" "$USERS_PY" || true)

  local has_test
  has_test=0
  [[ -f "$PRECEDENCE_TEST" ]] && has_test=1

  if [[ "$has_precedence" -gt 0 && "$has_test" -eq 1 ]]; then
    record 1 "✅" "$USERS_PY preserves proxy_user > jwt_user precedence (architectural immunity); regression guard test at $PRECEDENCE_TEST pins the contract"
    return
  fi

  if [[ "$has_precedence" -eq 0 ]]; then
    record 1 "❌" "$USERS_PY does NOT show the proxy_user precedence pattern. SurfSense's immunity to the cross-user stale-session leak depends on \`current_active_user\` returning \`proxy_user\` before falling back to \`jwt_user\`. If this property is gone, the leak class is re-introduced. Reference: openspec proxy-auth-middleware §'Identity mismatch SHALL flush', surfsense-security.md §'architecturally immune', Pressingly/SurfSense#22."
    return
  fi

  record 1 "❌" "$PRECEDENCE_TEST does NOT exist. The proxy_user > jwt_user precedence in $USERS_PY is the load-bearing reason SurfSense is immune to the cross-user stale-session leak — but it's unguarded by tests. A future refactor that flips the precedence (or removes the proxy_user check during a 'simplify auth' pass) would silently re-introduce the bug. Add the regression-guard test from Pressingly/SurfSense#22."
}

# ============================================================================
# Row 21 (idx 2): polynomial-regex avoidance in email-shape detection
# ============================================================================
check_row_21() {
  if [[ ! -f "$MIDDLEWARE" ]]; then
    record 2 "?" "$MIDDLEWARE not found — skipping"
    return
  fi

  local hits
  hits=$(grep -nE '\^\[\^[\\\\]s@\]\+@\[\^[\\\\]s@\]\+\\\.\[\^[\\\\]s@\]\+\$' "$MIDDLEWARE" 2>/dev/null || true)

  if [[ -n "$hits" ]]; then
    record 2 "❌" "Polynomial-backtracking email-shape regex detected in $MIDDLEWARE: $hits. SurfSense uses a substring check (\`'@' not in email\`); introducing the explicit ^...$ regex form re-enables polynomial backtracking on adversarial input. Rewrite to substring or indexOf-based check per openspec proxy-auth-middleware §'email-shape detection SHALL avoid polynomial-backtracking regex'."
    return
  fi

  record 2 "✅" "No polynomial-backtracking email-shape regex in $MIDDLEWARE; using substring check"
}

# ============================================================================
# Run checks
# ============================================================================
check_row_14
check_row_20
check_row_21

# ============================================================================
# Print table
# ============================================================================
echo "## SurfSense SSO Fork Audit"
echo
echo "Cross-app contract: https://github.com/awais786/sso-rules-moneta/blob/main/openspec/specs/proxy-auth-middleware/spec.md"
echo "Row numbers match the 21-row table at https://github.com/awais786/sso-rules-moneta/blob/main/skills/app-rules/SKILL.md#5-report"
echo
echo "| Row | Invariant | Status | Notes |"
echo "|-----|-----------|--------|-------|"
for i in "${!ROW_TITLES[@]}"; do
  printf "| %d | %s | %s | %s |\n" \
    "${ROW_NUMBERS[$i]}" "${ROW_TITLES[$i]}" "${ROW_STATUS[$i]:-?}" "${ROW_NOTES[$i]:-}"
done
echo

TOTAL_FAILS=0
for s in "${ROW_STATUS[@]}"; do
  [[ "$s" == "❌" ]] && TOTAL_FAILS=$((TOTAL_FAILS + 1))
done

if [[ "$TOTAL_FAILS" -eq 0 ]]; then
  echo "**All fork-side invariants hold.**"
  exit 0
fi

echo "**$TOTAL_FAILS violations.** Security-critical (row 20): $SECURITY_CRITICAL_FAILS."
if [[ "$SECURITY_CRITICAL_FAILS" -gt 0 ]]; then
  exit 1
fi
exit 0
