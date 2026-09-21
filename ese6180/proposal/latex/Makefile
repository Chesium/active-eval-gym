# Build the ESE 6180 proposal PDF.
#
# Requires pdflatex + bibtex with the JMLR/PMLR class chain available
# (jmlr.cls, algorithm2e, placeins, xcolor, ...). See README.md if
# `kpsewhich jmlr.cls` comes back empty.

TEX := proposal

.PHONY: all clean distclean

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex ../references.bib figures/cartpole_recovery_regions.png
	pdflatex -interaction=nonstopmode $(TEX).tex
	bibtex $(TEX)
	pdflatex -interaction=nonstopmode $(TEX).tex
	pdflatex -interaction=nonstopmode $(TEX).tex

# Regenerate the preliminary-results figure from the stored evaluation summary.
# The script resolves repository paths itself, so it runs from anywhere.
figure:
	../../../.venv/bin/python figures/make_recovery_regions.py

clean:
	rm -f $(TEX).aux $(TEX).bbl $(TEX).blg $(TEX).log $(TEX).out

distclean: clean
	rm -f $(TEX).pdf
