# EmerKit — instructions for Claude Code

You are running the model-dependent part of the EmerKit experiments on the author's machine. Everything that runs without
language models is already implemented, executed and reported (`report/results_pilot.pdf`, `results/`). Read `README.md`,
`CLAUDE_CODE.md`, `DATA.md` and `config/protocol.yaml` before doing anything.

## Rules (non-negotiable)
- Results come only from real executions. Never write, estimate or "fill in" a number, a label or a model output by hand.
- Every model call goes through `emerkit.llm.LLMClient`, which logs it to `llm_log.jsonl`; keep that file.
- Do not change the method, the protocol values or the evaluation metrics without asking the author. Fix bugs, not results.
- Use at least three models for A5, A6, the judges and the audit review: one small open model (local, Ollama or vLLM), one
  medium model, and at least one model that is not from the Claude family.
- Synthetic data and model judgements are declared as such; nothing is presented as a human validation.
- Ask before any step whose cost could exceed the budget the author gives you; report the cost of each series.

## Order of work
1. Setup: `pip install -e ".[llm]"`, `python scripts/fetch_data.py`, `make test` (10/10 properties must hold).
2. Smoke test (cheap, one model): `run_llm.py probes 0 20`, `run_llm.py a6 0 5`, `run_llm.py c1 0` on 2 requests,
   `scripts/run_f1.py 0 3 llm`. Inspect outputs and logs; fix parsing problems; then ask the author to confirm the budget.
3. Toucan: `run_llm.py toucan 20000`, print the columns, check that `tools` are parsed correctly on 20 rows before going on;
   then `run_llm.py toucan-caps` and `run_llm.py a7 S`.
4. Full model series for each model: `capabilities`, `probes`, `a6`, `c1`, `judges`, `audit-review`; then `fit-probes`.
5. F1 with the LLM policy on 3 seeds; full factorial grids with `EMERKIT_GRID=full make grids` (several cores).
6. Extend `make_report*.py` and `report/results_pilot.tex` with the new series; every number must come from generated macros.

## Environment variables
`EMERKIT_PROVIDER` (anthropic | openai), `EMERKIT_MODEL`, `ANTHROPIC_API_KEY`, and for OpenAI-compatible servers
`EMERKIT_BASE_URL` and `OPENAI_API_KEY`.
