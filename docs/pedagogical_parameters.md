# Pedagogical parameters: expert review

These parameters were reviewed by Claude acting as a panel of pedagogical experts, at the authors' request, because no human
panel was available. **They are model judgements, not a validation by instructors, and must be declared as such.** The script
`scripts/run_llm.py judges` repeats the review with several independent models and reports their agreement; the authors should
check a sample by hand. Values live in `emerkit/vocab.py`.

## Default data attributes per pedagogical type (`DEFAULT_D`)
| Type | Attributes | Rationale |
|---|---|---|
| acquisition | PROF | Explaining or retrieving material uses the learner's level and preferences, not results or identity. |
| practice | CONT | Exercises handle the learner's answers and code; practice is ungraded. |
| formative | ID, PERF, CONT | Feedback attaches results to an identified learner. |
| summative | ID, PERF, CONT | Same as formative, with certification stakes. |
| discussion | ID, CONT | Posts are authored by identified learners. |
| production | ID, CONT | Submitted work is authored and identified. |

## Norm base, standard profile (`norm_base("standard")`)
Institution: every flow allowed. Contracted processor (bound by a data processing agreement, GDPR art. 28): allowed except
performance and identity in assessment tasks, which stay inside the institution. Third party and public web: only profile and
content in acquisition tasks (a level or a query is needed to search or explain); everything else denied. This follows the
principles of purpose limitation and data minimisation (GDPR art. 5) and the FERPA restriction on disclosing education records.
Permissive profile: content may also reach third parties in practice, production and discussion. Strict profile: behaviour and
profile never reach processors; profile never reaches third parties.

## Fit matrix (`PHI`)
Explanation fits acquisition best and assessment least; exercise and assessment fit practice and assessment; recording fits
formative and summative tasks. A tool that hands out full solutions fits acquisition better than practice (assistance dilemma,
Koedinger & Aleven 2007).

## Substitution catalogue (`SUB`)
Allowed: acquisition -> discussion (same concepts, higher Bloom level); practice -> production (applying -> creating);
discussion -> acquisition is listed but rejected by the preservation relation because the cognitive level would drop;
production -> practice likewise. Formative assessment is substituted only by formative assessment, so that evidence of learning is
kept (constructive alignment, Biggs 1996). Summative tasks are never substituted automatically.

## Type-capability table (`TYPE_CAPS`) and core capabilities (`CORE_CAP`)
Each type lists the capabilities an instructor would expect; each domain has one core capability per assessed type (for example
grade_code for programming assessment, pronunciation_eval for summative English assessment).

## Changes made during the review
None of the provisional values was changed; two cells were flagged for the human check: the PRO cell for behaviour data in
practice tasks (allowed) and the acquisition -> discussion substitution (allowed).
