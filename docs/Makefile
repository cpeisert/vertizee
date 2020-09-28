# Makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS)

.PHONY: help clean html dirhtml pickle json latex latexpdf changes linkcheck doctest epub

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html      to make standalone HTML files"
	@echo "  dirhtml   to make HTML files named index.html in directories"
	@echo "  pickle    to make pickle files"
	@echo "  epub      to make an epub"
	@echo "  json      to make JSON files"
	@echo "  latex     to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  latexpdf  to make LaTeX files and run them through pdflatex"
	@echo "  changes   to make an overview of all changed/added/deprecated items"
	@echo "  linkcheck to check all external links for integrity"
	@echo "  doctest   to run all doctests embedded in the documentation (if enabled)"


clean:
	-rm -rf _build/*
	-rm -rf ghpages_build
	-rm -rf auto_examples modules

dist: html
	test -d build/latex || make latex
	make -C build/latex all-pdf
	-rm -rf build/dist
	(cd build/html; cp -r . ../../build/dist)
	(cd build/dist && tar czf ../dist.tar.gz .)

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html $(ALLSPHINXOPTS)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html"

dirhtml:
	$(SPHINXBUILD) -b dirhtml $(SOURCEDIR) $(BUILDDIR)/dirhtml $(ALLSPHINXOPTS)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/dirhtml."

pickle:
	$(SPHINXBUILD) -b pickle $(SOURCEDIR) $(BUILDDIR)/pickle $(ALLSPHINXOPTS)
	@echo
	@echo "Build finished; now you can process the pickle files."

json:
	$(SPHINXBUILD) -b json $(SOURCEDIR) $(BUILDDIR)/json $(ALLSPHINXOPTS)
	@echo
	@echo "Build finished; now you can process the JSON files."

epub:
	$(SPHINXBUILD) -b epub $(SOURCEDIR) $(BUILDDIR)/epub $(ALLSPHINXOPTS)
	@echo
	@echo "Build finished. The epub file is in $(BUILDDIR)/epub."

latex:
	$(SPHINXBUILD) -b latex $(SOURCEDIR) $(BUILDDIR)/latex $(ALLSPHINXOPTS)
	@echo
	@echo "Build finished; the LaTeX files are in $(BUILDDIR)/latex."
	@echo "Run \`make' in that directory to run these through (pdf)latex" \
	      "(use \`make latexpdf' here to do that automatically)."

latexpdf:
	$(SPHINXBUILD) -b latex $(SOURCEDIR) $(BUILDDIR)/latex $(ALLSPHINXOPTS)
	@echo "Running LaTeX files through pdflatex..."
	make -C $(BUILDDIR)/latex all-pdf
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."

changes:
	$(SPHINXBUILD) -b changes $(SOURCEDIR) $(BUILDDIR)/changes $(ALLSPHINXOPTS)
	@echo
	@echo "The overview file is in $(BUILDDIR)/changes."

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(SOURCEDIR) $(BUILDDIR)/linkcheck $(ALLSPHINXOPTS)
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

doctest:
	$(SPHINXBUILD) -b doctest $(SOURCEDIR) $(BUILDDIR)/doctest $(ALLSPHINXOPTS)
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in $(BUILDDIR)/doctest/output.txt."

docs: clean html latexpdf
	cp $(BUILDDIR)/latex/vertizee_reference.pdf $(BUILDDIR)/html/_downloads/.
