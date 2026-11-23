# EmerKit: Composing MCP Toolkits for LLM Agents in Learning Workflows

Implementation, datasets, experiments, and interactive portal of the article *EmerKit: Composing MCP Toolkits for LLM Agents in
Learning Workflows* (F. Ghedass, H. Mezni, A. Alsehaim, H. Elmannai, S. Alalmaei, M. Sliti).

EmerKit derives a learning workflow from a learner's request and equips each task with a small toolkit of MCP tools: it counts
each capability once, integrates quality of service, charges interference between similar tools and saturation of the context
window, checks every flow of learner data against education-specific norms with a calibrated risk bound, and replaces tasks that
cannot be equipped by substitutes that preserve their learning objective.

| | |
|---|---|
| **Portal online** | https://agai-research.github.io/EmerKit/ (after GitHub Pages is enabled; see below) |
| **Demo video** | [`docs/EmerKit_portal_demo.mp4`](docs/EmerKit_portal_demo.mp4), about three minutes, live run with Claude |
| **Portal on your computer** | `python3 portal/serve.py`, then open http://localhost:8000 ([`portal/README.md`](portal/README.md)) |
| **Experiments** | `pip install -r requirements.txt`, then `make test` and `make all-local` ([Reproduce](#reproduce) below) |
| **Licence** | Code MIT; generated data CC BY 4.0; third-party data under their own licences ([`THIRD_PARTY.md`](THIRD_PARTY.md), [`DATA.md`](DATA.md)) |

## Demo video
## Portal
The portal runs the request phase of EmerKit in the browser (decomposition by a language model or by rules, workflow validation
and repair, flow check, exact composition, revision) on the prepared registries, and explains the approach step by step.
Its composer was checked against the Python implementation on 550 tasks (`portal/tests/`): same feasibility, same optimal
objective, same minimal conflicts. A learner page shows the plan in plain language; an expert view shows every computation.

## Status
Language-model series were run with Claude models only (A0-LLM, expert panel, A5 on real Toucan definitions, audit review, C1,
F1 with an agent, panel sensitivity P1/P2; logs in `results/llm/llm_log.jsonl`). Results: `results/*.csv` holds the reference
series on the final data with 300 calibration tasks; `results/grid/` the grid series; `results/f1/` the live-enactment runs;
`results/extra/` the additional series X1–X6; `results/pilot_v1/` and `results/pilot_v2_ncal100/` are earlier pilots kept for
transparency, not final results. Audit labels of real tools come from static analysis and a model review (Claude), without human
verification. Full factorial designs: `EMERKIT_GRID=full make grids`.

## Contents
| Path | Role | Executed here |
|---|---|---|
| `emerkit/vocab.py` | Domains, capabilities, types, attributes, roles, norm bases, substitution catalogue (provisional) | yes |
| `emerkit/data.py` | Synthetic educational MCP servers with ground truth; MCP-tools loader; requests, workflows, archetypes | yes |
| `emerkit/prep.py` | Preparation: capabilities, weak supervision, conformal sets, interference index | yes |
| `emerkit/compose.py` | Flow check, exact CP-SAT composer, conflict explanation, Learn-then-Test, RAG-MCP and Zeng et al. | yes |
| `emerkit/e2e.py` | Algorithm 3: validator, conservative mode, backtracking, substitution; gap map | yes |
| `emerkit/llm.py` | Model client, Compositio Prompto (adapted), selection probes, LLM generator and capability extraction, Toucan loader | **no** (Claude Code) |
| `run_pilot.py`, `run_all.py` | Series A0–A4, B0–B4, C2–C4, D1–D2, E1–E2 | yes |
| `aggregate.py`, `analyze_grid.py`, `make_report*.py` | Tables, figures, LaTeX macros | yes |
| `report/results_pilot.tex/.pdf` | Pilot report (all numbers generated) | yes |
| `emerkit/sources.py`, `emerkit/concepts.py`, `emerkit/esco.py` | Toucan real catalogue; LectureBank, CS2023 and CEFR concept graphs; ESCO skills and demand weights for the gap map | yes |
| `emerkit/runtime/` | Executable MCP servers (ground-truth behaviour, canaries, sinks) and MCP gateway | yes |
| `emerkit/stats.py` | Bootstrap, Wilcoxon + Holm, Cliff's delta, mixed-effects probe model | yes |
| `emerkit/llm_agent.py`, `scripts/run_llm.py` | LLM agent for F1; runners of A0-LLM, A5, A6, A7, C1, judges, audit review | **no** (Claude Code) |
| `scripts/run_f1.py`, `scripts/audit_static.py` | Live enactment through the gateway; static audit of real servers | yes |
| `run_extra.py` | Additional series: catalogue size (X1), capabilities per task (X2), workflow length (X4), exact vs greedy composition (X5), calibration under distribution shift (X6) | yes |
| `make_figures.py`, `make_paper_tables.py` | Publication figures (vector PDF and 300-dpi PNG, colour-blind-safe) and comparative tables with best values in bold | yes |
| `tests/test_properties.py` | 10 property tests (CI) | yes |
| `scripts/run_probes_real.py` | A5 on real tool definitions (requests by one model, probes by others, logistic fit) | yes (Claude) |
| `analyze_judges.py`, `make_report3.py` | Expert-panel agreement; report part 3 (model-dependent series, panel sensitivity) | yes |
| `docs/`, `DATA.md`, `Makefile`, `.github/workflows/` | Expert review, data cards, build targets, CI | — |
| `data/mcp_tools_compact.json` | MCP-tools (308 servers, 2,797 tools), embeddings removed | — |
| `data/generated/` | Exported datasets: catalogues, tasks, requests, workflows, archetypes, norms, vocabulary (5 seeds) | — |
| `results/` | Raw results (CSV) of every executed series | — |
| `config/protocol.yaml` | Frozen parameters and grids | — |

## Reproduce
`pip install numpy scipy scikit-learn pandas matplotlib ortools` then
`python run_pilot.py 0 1 2 3 4`, `python run_all.py SERIES CONFIG...` (`python run_all.py list`), `python export_dataset.py`,
`python make_report.py && python make_report2.py`, and compile `report/results_pilot.tex`.

## Not executed here (requires model calls or external data) — see CLAUDE_CODE.md
A5–A6 selection probes, A7 Toucan trajectories, C1 with Compositio Prompto, LLM workflow generator and capability extraction, F1 live enactment.

## Method names
In the paper and figures: QoS-IP = QoS-aware composition by integer programming (`Zeng` in the result files), `+F` = filter on declared annotations (`+filter`), CP = Compositio Prompto, EmerKit-NS / -NI / -DA / -LLM = without substitution / without interaction terms / with declared annotations / behind an LLM workflow generator.
