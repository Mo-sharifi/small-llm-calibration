.PHONY: all clean

all: report.pdf

report.pdf: report.tex refs.bib $(wildcard figures/*.pdf)
	pdflatex -interaction=nonstopmode report.tex
	bibtex report
	pdflatex -interaction=nonstopmode report.tex
	pdflatex -interaction=nonstopmode report.tex

clean:
	rm -f report.aux report.bbl report.blg report.log report.out
