# Makefile for EMNLP 2026 paper

PAPER = emnlp2026
PYTHON = python3

.PHONY: all clean figures benchmark view

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex figures/
	latexmk -pdf $(PAPER).tex

figures: generate_figures.py
	$(PYTHON) generate_figures.py

benchmark: benchmark.py
	$(PYTHON) benchmark.py

view: $(PAPER).pdf
	xdg-open $(PAPER).pdf

clean:
	rm -f $(PAPER).pdf $(PAPER).aux $(PAPER).log $(PAPER).bbl $(PAPER).blg
	rm -rf figures/

setup:
	@echo "Installing LaTeX (if needed)..."
	@which pdflatex > /dev/null 2>&1 || sudo apt-get install -y texlive-full
	@echo "Installing matplotlib..."
	pip install matplotlib numpy
