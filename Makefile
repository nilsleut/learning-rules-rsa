# Paper v3 (arXiv:2604.16875): numbers -> PDF -> number check.
# check_numbers.py reads the compiled PDF and .aux, so `check` compiles first.
# Without make: python scripts/paper_v3/build_and_check.py (same steps).

PYTHON ?= python
PAPER  := paper/arxiv_upload_learning_rules_v3
TEX    := learning_rules_rsa_paper_v3

.PHONY: macros pdf check clean

macros:
	$(PYTHON) scripts/paper_v3/make_macros.py

pdf: macros
	cd $(PAPER) && pdflatex -interaction=nonstopmode -halt-on-error $(TEX).tex > /dev/null
	cd $(PAPER) && pdflatex -interaction=nonstopmode -halt-on-error $(TEX).tex > /dev/null
	cd $(PAPER) && pdflatex -interaction=nonstopmode -halt-on-error $(TEX).tex > /dev/null

check: pdf
	$(PYTHON) scripts/paper_v3/check_numbers.py

clean:
	cd $(PAPER) && rm -f *.aux *.log *.out *.synctex.gz
