Agent contract

Context
This repo contains two projects:
- ./My-ai-agent/ollama-turbo-cli (python)
- ./My-ai-agent/agent-desk-pro (node+tauri)

Authoritative commands
Python
install: pip install -r My-ai-agent/ollama-turbo-cli/requirements.txt
test:    pytest -q My-ai-agent/ollama-turbo-cli
run:     python -m src.cli --help

Node
install: cd My-ai-agent/agent-desk-pro && pnpm install
test:    cd My-ai-agent/agent-desk-pro && pnpm test
dev:     cd My-ai-agent/agent-desk-pro && pnpm dev

Rules
Prefer offline builds. Enable internet only for dependency docs or API changelogs.
Do not attempt native Tauri packaging in cloud; dev and tests only.
No edits under src/web/pipeline/ without matching tests.
Separate mechanical refactors and logic changes into different commits.
Open changes as a PR with summary, risk, and test results.

Secrets
Never print .env. Use placeholders in examples. Assume CI-like perms.
