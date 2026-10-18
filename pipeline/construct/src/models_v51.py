"""Model routing for the v5.1 pipeline (wraps trajgen_models without editing it, so frozen runs keep their route).

k3-256k goes through LiteLLM (127.0.0.1:4001; three Kimi keys with failover), not the direct Kimi coding endpoint
that the older registry points to. Decision 2026-10-02: K3 is used with reasoning_effort=low only.
Everything else is delegated unchanged.
"""
import trajgen_models as tm

K3_MODEL = 'k3-256k'
K3_EFFORT = 'low'
K3_LITELLM = tm.Endpoint('http://127.0.0.1:4001', tm.REPO_ENV, 'LITELLM_API_KEY', fixed_temperature=1.0, fixed_top_p=0.95)
tm.MODELS[K3_MODEL] = K3_LITELLM
# 2026-10-02 13:20 (user): grok-4.7 is served by LiteLLM :4001 -> cursor/grok-4.7-xhigh by default, falling back to
# grok-4.7-build-fast on error or 45 s without data. The old CLIProxy xAI route was unstable (Cloudflare 5xx).
# Note: the Cursor deployment runs at xhigh; a client reasoning_effort=low is not honoured there.
tm.MODELS['grok-4.7'] = tm.LITELLM
# 2026-10-04 (user): gpt-5.6-sol. LiteLLM :4001 serves the GPT models as anthropic/ deployments that only work on
# /v1/messages (its /v1/chat/completions returns 500 'unexpected keyword argument thinking', gpt-6-luna too), so use the
# same backend directly: CLIProxy :8317 OpenAI route (smoke 200, returned model gpt-5.6-sol, reasoning_effort low).
tm.MODELS['gpt-5.6-sol'] = tm.CLIPROXY
# 2026-10-03 08:15 user: K2.8 Preview (`kimi-for-coding`, vision, effort low|high|max) as a second family because GLM is
# slow. KIMI_API_KEY0/1 have hit the monthly limit; only KEY2 answers, so the route is pinned to KEY2.
tm.MODELS['kimi-for-coding'] = tm.Endpoint('https://api.kimi.com/coding', tm.PROXY_ENV, 'KIMI_API_KEY2',
                                           fixed_temperature=1.0, fixed_top_p=0.95)


# 2026-10-05 (user): Grok has two sources. 'grok-4.7-cpa' = CLIProxy :8317 xAI route, upstream model 'grok-4.7'
# (image probe 200 in 1-2 s; returned model reports grok-4.7-build). Separate name -> separate run dir and receipts.
GROK_CPA = 'grok-4.7-cpa'
tm.MODELS[GROK_CPA] = tm.CLIPROXY
# 2026-10-05 14:20 (user): use Cursor for Grok as much as possible. 'grok-4.7-cur' = CLIProxy Cursor id
# cursor/grok-4.7-low (same model Grok 4.7, low effort as the protocol asks; the LiteLLM 'grok-4.7' entry pins
# cursor/grok-4.7-xhigh with a 45 s fallback to grok-4.7-build-fast).
GROK_CUR = 'grok-4.7-cur'
tm.MODELS[GROK_CUR] = tm.CLIPROXY
UPSTREAM = {GROK_CPA: 'grok-4.7'}  # GROK_CUR goes through cursor_anthropic (OpenAI route breaks on images)
GLM_DIRECT = 'glm-5.3-flash'  # 2026-10-03 user: GLM goes straight to Zhipu (KEY2 Max + KEY0 Pro), not via LiteLLM


def make_api(model: str, out, **kw):
    if model == GROK_CUR:
        from cursor_anthropic import CursorAnthropicAPI
        return CursorAnthropicAPI(out, model, rpm=kw.get('rpm', 120), max_tokens=kw.get('max_tokens', 8192))
    if model == GLM_DIRECT:
        from glm_direct import GLMDirectAPI
        return GLMDirectAPI(out, model, rpm=kw.get('rpm', 120), max_tokens=kw.get('max_tokens', 16384),
                            effort=kw.get('effort', 'high'))
    if model == K3_MODEL and kw.get('effort', tm.DEFAULT_EFFORT) != K3_EFFORT:
        raise ValueError(f"{K3_MODEL} is used with effort={K3_EFFORT!r} only (got {kw.get('effort', tm.DEFAULT_EFFORT)!r})")
    api = tm.make_api(model, out, **kw)
    if model in UPSTREAM:
        api.extra = {**api.extra, 'model': UPSTREAM[model]}  # RoleAPI payload spreads extra after 'model': upstream name
    return api
