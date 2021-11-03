DOCNAME=proposal
TEXVER=xelatex

all: proposal

.PHONY: clean

proposal:
	$(TEXVER) $(DOCNAME).tex
	bibtex $(DOCNAME).aux
	$(TEXVER) $(DOCNAME).tex
	$(TEXVER) $(DOCNAME).tex

view: proposal
	open $(DOCNAME).pdf

clean:
	rm $(DOCNAME).blg $(DOCNAME).bbl $(DOCNAME).aux $(DOCNAME).log $(DOCNAME).run.xml $(DOCNAME)-blx.bib
