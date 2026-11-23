# Running the full campaign in Claude Code
1. `pip install -e ".[llm]"`; set `ANTHROPIC_API_KEY` and/or an OpenAI-compatible endpoint (`EMERKIT_PROVIDER=openai`,
   `EMERKIT_BASE_URL`, `OPENAI_API_KEY`); set `EMERKIT_MODEL`. Use at least three models, one small open model, at least one non-Claude.
2. `make all-local` (tests, dataset, pilot, all grids, F1 exhaustive, audit step 1, report) -- no model needed.
3. For each model: `run_llm.py capabilities S`, `probes S`, `a6 S`, `c1 S`, `judges`, `audit-review`; then `fit-probes`.
4. `run_llm.py toucan 20000` then `a7 S` (trajectories, co-usage; `Prepared.set_traces` feeds the composer).
5. F1 with an agent: `scripts/run_f1.py S 100 llm`.
6. Every call is in `llm_log.jsonl`; results come from these logs only.
