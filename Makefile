PY ?= python
test:            ; $(PY) tests/test_properties.py
dataset:         ; $(PY) export_dataset.py && for s in 0 1 2 3 4; do $(PY) scripts/run_llm.py probes-design $$s; done
pilot:           ; for s in 0 1 2 3 4; do $(PY) run_pilot.py $$s; done
grids:           ; for s in A2 A3 A4 B2 B3 D2 C3 C4 C2 E1 E2; do n=$$($(PY) run_all.py list | awk -v k=$$s '$$1==k{print $$2}'); $(PY) run_all.py $$s $$(seq 0 $$((n-1))); done
f1:              ; for s in 0 1 2; do $(PY) scripts/run_f1.py $$s 100 exhaustive; done
audit:           ; $(PY) scripts/audit_static.py 40
report:          ; $(PY) make_report.py && $(PY) make_report2.py && $(PY) make_report3.py && $(PY) make_report4.py && $(PY) make_figures.py && $(PY) make_paper_tables.py && cd report && pdflatex results_full.tex && pdflatex results_full.tex
extra:           ; for s in X1 X2 X4 X5 X6; do n=$$($(PY) run_extra.py list | tr -d "{}'," | tr ' ' '\n' | grep -A1 "$$s:" | tail -1); $(PY) run_extra.py $$s $$(seq 0 $$((n-1))); done
grids-full:      ; EMERKIT_GRID=full $(MAKE) grids
all-local: test dataset pilot grids f1 audit report
