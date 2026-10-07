# Halo

Halo is an open-source, autonomous AI pentesting tool. AI agents run your
code dynamically, probe it for vulnerabilities, and validate findings with
real proofs of concept inside a Docker sandbox.

On top of the classic runtime scan it ships a whitebox static analysis
pipeline driven by a code knowledge graph (tree-sitter → graphify) plus
semgrep/gitleaks/trivy, with fingerprint dedupe, a sanitizer gate, and an
LLM adversarial verifier.

## Layout

```
halo/
  agents/        agent graph + prompts (Jinja system prompt, skills)
  tools/         agent tools: shell, browser, proxy, sast/, kg/, reporting, ...
  sast/          static analysis pipeline (fingerprint, findings store,
                 semgrep/gitleaks/trivy runners, gates, graphify bridge)
  runtime/       Docker sandbox lifecycle
  config/        settings (env aliases), LiteLLM provider wiring
  core/          agent loop, sessions, budget hooks
  report/        findings state, SARIF, dedupe, usage/pricing
  skills/        internal markdown knowledge packs the agents load
  interface/     CLI, TUI (Bubble Tea), local web viewer
containers/      Kali-based sandbox image (semgrep, nuclei, nmap, ...)
tests/           pytest suite
```

## Prerequisites

- Python 3.12+, `uv`
- Docker running
- An LLM API key (any LiteLLM-compatible provider)
- Go (only for the interactive TUI; headless use does not need it)

## Install

```bash
make dev-install          # create the uv venv with dev deps

# sandbox image (the registry image is private; build locally)
docker build -t halo-sandbox:local -f containers/Dockerfile .
```

## Configure

Persisted in `~/.halo/cli-config.json` (env always wins over the file):

```json
{
  "env": {
    "HALO_IMAGE": "halo-sandbox:local",
    "HALO_LLM": "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b",
    "LLM_API_KEY": "<key>"
  }
}
```

Or export in your shell:

```bash
export HALO_LLM="openrouter/z-ai/glm-5.3"   # any LiteLLM model id
export LLM_API_KEY="<key>"
export LLM_API_BASE="<url>"                  # only for OpenAI-compatible endpoints
export HALO_IMAGE="halo-sandbox:local"
```

## Run

```bash
# interactive TUI (requires Go on PATH)
uv run halo -t <target>

# headless
uv run halo -n -t <target> --scan-mode quick --max-budget 10

# local source tree (whitebox: SAST + KG tools enabled)
uv run halo -n -t ./my-app --scan-mode quick --max-budget 10

# GitHub repo (cloned automatically, also whitebox)
uv run halo -n -t https://github.com/org/repo --scan-mode quick
```

Headless exit codes: `0` clean, `1` fatal error, `2` vulnerabilities found.
Artifacts in `halo_runs/<run>/` (report, `vulnerabilities.json`, SARIF,
`run.json`). Live web replay of a run: `uv run halo view`.

## SAST pipeline + knowledge graph

Enabled automatically for whitebox targets (`--target` on a local source
tree or a GitHub repo), gated by `HALO_SAST_ENABLED` (default on):

1. `kg_build(source_path=...)` — tree-sitter parse of the target into a
   graphify knowledge graph (`halo_runs/<run>/graphify-out/graph.json`).
2. `sast_scan(source_path=...)` — semgrep/gitleaks/trivy, normalized,
   fingerprinted (`sha1(file+sink+slice+taxonomy)`), deduped into
   `findings.jsonl`, and injected into the graph as finding nodes.
3. `finding_verify(finding_id=..., source_slice=...)` — syntactic sanitizer
   gate, then one adversarial LLM call; findings end up
   `PROPOSED / VERIFIED / REFUTED / UNVERIFIED`.
4. `finding_state(...)` — manual verdicts.
5. `kg_query` / `kg_path` / `kg_explain` — pull targeted code slices from
   the graph instead of reading whole files.

## Development

```bash
make check-all          # ruff + mypy + pyright + bandit
uv run pytest           # test suite
cd halo/interface/tui && PATH="$HOME/.local/go/bin:$PATH" go build ./cmd/halo-tui   # TUI sidecar
```

## Use responsibly

Only scan targets you own or are explicitly authorized to assess.

## License

Apache 2.0 — see [LICENSE](LICENSE).
