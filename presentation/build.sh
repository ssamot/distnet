#!/bin/bash
source=beliefs
pandoc --slide-level 2 -V theme=bjeldbak --template=custom.beamer --toc -t beamer $source.md -o $source.pdf
pdfnup $source.pdf --nup 2x3 --no-landscape --keepinfo --paper A4 --frame true --scale 0.9 --suffix "nup"