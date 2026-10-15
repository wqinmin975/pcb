#!/usr/bin/env bash
# Regenerate the track-changes version of the manuscript.
#
#   baseline : solderbench_softwarex_draft_v16_base.tex  (original submission, commit 0035d61)
#   current  : solderbench_softwarex_draft.tex           (revised manuscript)
#   output   : solderbench_softwarex_draft_diff.tex  ->  solderbench_trackchanges.pdf
#
# latexdiff is shipped with MiKTeX but needs a perl interpreter. On Windows the perl
# that comes with "Git for Windows" works, provided that
#   (a) its core module directory is put on PERL5LIB (msys-style /c/... paths, ":"-separated),
#   (b) the CPAN module Algorithm::Diff is available, e.g.
#         mkdir -p /tmp/perllib && curl -sL "$(curl -s https://fastapi.metacpan.org/v1/download_url/Algorithm::Diff | grep -o '"download_url" : "[^"]*"' | sed 's/.*: "//;s/"$//')" \
#           | tar xz -C /tmp/perllib && cp /tmp/perllib/*/lib/Algorithm/Diff.pm /tmp/perllib/Algorithm/Diff.pm
#         (adjust /tmp/perllib to a directory you keep)
set -e
cd "$(dirname "$0")"

PERL="/c/Program Files/Git/usr/bin/perl.exe"
LDIFF="/c/Users/Administrator/AppData/Local/Programs/MiKTeX/scripts/latexdiff/latexdiff"
export PERL5LIB="/c/Users/Administrator/AppData/Local/Temp/perllib:/c/Program Files/Git/usr/share/perl5/core_perl:/c/Program Files/Git/usr/lib/perl5/core_perl"

"$PERL" "$LDIFF" --encoding=utf8 --allow-spaces \
  solderbench_softwarex_draft_v16_base.tex \
  solderbench_softwarex_draft.tex \
  > solderbench_softwarex_draft_diff.tex

pdflatex -interaction=nonstopmode -halt-on-error -jobname=solderbench_trackchanges solderbench_softwarex_draft_diff.tex > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error -jobname=solderbench_trackchanges solderbench_softwarex_draft_diff.tex
cp solderbench_trackchanges.pdf solderbench_softwarex_draft_v20_trackchanges.pdf

# clean version
pdflatex -interaction=nonstopmode -halt-on-error -jobname=solderbench_r20 solderbench_softwarex_draft.tex > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error -jobname=solderbench_r20 solderbench_softwarex_draft.tex
cp solderbench_r20.pdf solderbench_softwarex_draft_v20_clean.pdf

echo "done: solderbench_softwarex_draft_v20_trackchanges.pdf / solderbench_softwarex_draft_v20_clean.pdf"
