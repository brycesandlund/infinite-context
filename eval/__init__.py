"""Multi-backend eval for the recursive-agent RULER suite.

Runs the SAME backend-agnostic agent loop (eval/agent.py) against any
ModelBackend — the Tinker-hosted policy (Qwen), Anthropic, or OpenAI — so the
comparison isn't confounded by harness differences. Training (train.py) keeps
its own token-level RL rollout; this package shares only the backend-agnostic
surface via harness.py.
"""
