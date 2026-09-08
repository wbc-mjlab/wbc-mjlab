# Build the technical report PDF (local only; paper/ is gitignored for now).
.PHONY: pdf clean

pdf:
	latexmk -pdf -interaction=nonstopmode main.tex

clean:
	latexmk -C
	rm -f main.bbl main.blg
