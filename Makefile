all: build

pdf:
	./dist/build/make-pdf/make-pdf > sample.tex
	pdflatex sample.tex

include common.mk
