#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
docx2tex.py  —  Convert solderbench_softwarex_draft.docx -> LaTeX (.tex) + PDF.

Strategy (pandoc is unavailable; MiKTeX/elsarticle is):
  python-docx extracts the structured content, we emit a clean elsarticle
  manuscript and compile with pdflatex. Handles: frontmatter (title/author/
  affil/corresponding-email), abstract + keywords, numbered headings, bullet
  lists, verbatim code block, booktabs tables (with captions), figures, and a
  thebibliography built from the inline [n] reference list (in-text [n] -> cite).
"""
import os, re, sys
import docx
from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.table import Table

SRC = r"D:\ace\PCB\paper\solderbench_softwarex_draft.docx"
OUTDIR = r"D:\ace\PCB\paper"
FIGDIR = os.path.join(OUTDIR, "figures")
TEX = os.path.join(OUTDIR, "solderbench_softwarex_draft.tex")
os.makedirs(FIGDIR, exist_ok=True)

d = Document(SRC)

# ---------------------------------------------------------------- helpers
def latex_escape(s):
    for a, b in [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'),
                 ('$', r'\$'), ('#', r'\#'), ('_', r'\_'),
                 ('{', r'\{'), ('}', r'\}'), ('~', r'\textasciitilde{}'),
                 ('^', r'\textasciicircum{}')]:
        s = s.replace(a, b)
    return s

def runs_to_latex(para):
    parts = []
    for r in para.runs:
        txt = r.text
        if not txt:
            continue
        txt = latex_escape(txt)
        if r.bold and r.italic:
            txt = r'\textbf{\textit{%s}}' % txt
        elif r.bold:
            txt = r'\textbf{%s}' % txt
        elif r.italic:
            txt = r'\textit{%s}' % txt
        parts.append(txt)
    return ''.join(parts)

def apply_citations(s):
    def repl(m):
        keys = ','.join('ref' + x.strip() for x in m.group(1).split(','))
        return r'\cite{%s}' % keys
    return re.sub(r'\[(\d+(?:\s*,\s*\d+)*)\]', repl, s)

def strip_number(h):
    return re.sub(r'^\d+(\.\d+)*\.?\s*', '', h).strip()

def strip_tab(cap):
    return re.sub(r'^Table\s+\d+\.\s*', '', cap, flags=re.I).strip()

def strip_fig(cap):
    return re.sub(r'^Figure\s+\d+\.\s*', '', cap, flags=re.I).strip()

def looks_like_code(t):
    s = t.strip()
    if not s or len(s) > 200:
        return False
    if re.search(r'[.!?]$', s):          # sentence -> prose
        return False
    if re.match(r'^(from\s|import\s|print\()', s):
        return True
    if ('=' in s) and ('(' in s):
        return True
    if s.endswith(')') or s.endswith(',') or s.endswith('('):
        return True
    return False

def cell_text(cell):
    parts = []
    for p in cell.paragraphs:
        t = runs_to_latex(p)
        if t:
            parts.append(t)
    return ' '.join(parts).strip()

def table_to_latex(tbl, caption):
    rows = tbl.rows
    ncols = max(len(r.cells) for r in rows)
    spec = 'll' if ncols == 2 else 'l' + 'c' * (ncols - 1)
    L = []
    L.append(r'\begin{table}[htbp]')
    L.append(r'\centering')
    L.append(r'\caption{%s}' % latex_escape(strip_tab(caption) if caption else ''))
    L.append(r'\begin{tabular}{%s}' % spec)
    L.append(r'\toprule')
    for ri, row in enumerate(rows):
        cells = []
        for ci in range(ncols):
            c = row.cells[ci].text.strip()
            txt = cell_text(row.cells[ci]) if c else ''
            if ri == 0 and txt:
                txt = r'\textbf{%s}' % txt
            cells.append(txt)
        L.append(' & '.join(cells) + r' \\')
        if ri == 0:
            L.append(r'\midrule')
    L.append(r'\bottomrule')
    L.append(r'\end{tabular}')
    L.append(r'\end{table}')
    return '\n'.join(L)

def has_drawing(para):
    return para._p.find('.//' + qn('a:blip')) is not None

def extract_image(para, idx):
    blip = para._p.find('.//' + qn('a:blip'))
    rId = blip.get(qn('r:embed'))
    part = d.part.related_parts[rId]
    path = os.path.join(FIGDIR, 'fig%d.png' % idx)
    with open(path, 'wb') as f:
        f.write(part.blob)
    return path

# ---------------------------------------------------------------- frontmatter
front = {'title': '', 'author': '', 'affil': '', 'email': ''}
# pre-scan preamble paragraphs (before Abstract)
for p in d.paragraphs:
    st = p.style.name if p.style else ''
    t = p.text.strip()
    if st == 'Title':
        front['title'] = latex_escape(p.text.strip())
    elif '[Author Name]' in t:
        front['author'] = t
    elif '[Affiliation]' in t:
        front['affil'] = re.sub(r'^[a-z]\s+', '', t).strip()
    elif 'Corresponding author' in t:
        front['email'] = t

# ---------------------------------------------------------------- body loop
children = list(d.element.body.iterchildren())
out = []
out.append(r'\documentclass[3p]{elsarticle}')
out.append(r'\usepackage{times}')
out.append(r'\usepackage{graphicx}')
out.append(r'\graphicspath{{figures/}}')
out.append(r'\usepackage{booktabs}')
out.append(r'\usepackage[pdfencoding=auto]{hyperref}')
out.append(r'\hypersetup{colorlinks=true,urlcolor=blue}')
out.append(r'\begin{document}')
out.append(r'\begin{frontmatter}')
out.append(r'\title{%s}' % front['title'])
# ---- Authors (real metadata provided by user) ----
# [1] Guangdong University of Science and Technology (GDUST), Dongguan
# [2] Tianjin Normal University (Tianjin); emails as provided
# [3] Guangdong University of Technology (GDUT), Guangzhou -- corresponding
out.append(r'\author[1]{Qinmin Wang\fnref{fn1}}')
out.append(r'\author[2]{Zixin Wang\fnref{fn2}}')
out.append(r'\author[1]{Hongyuan Wang}')
out.append(r'\author[3]{Zongzhong Wang\corref{cor1}}')
out.append(r'\address[1]{School of Computer Science and Artificial Intelligence, '
           r'Guangdong University of Science and Technology, Dongguan, China}')
out.append(r'\address[2]{School of Electronic and Communication Engineering, '
           r'Tianjin Normal University, Tianjin, China}')
out.append(r'\address[3]{School of Electromechanical Engineering, '
           r'Guangdong University of Technology, Guangzhou, China}')
out.append(r'\cortext[cor1]{Corresponding author at School of Electromechanical '
           r'Engineering, Guangdong University of Technology, Guangzhou, China. '
           r'E-mail: wangzz@gdut.edu.cn}')
out.append(r'\fntext[fn1]{E-mail: wangqinmin@gdust.edu.cn}')
out.append(r'\fntext[fn2]{E-mail: 396514839@qq.com; wanzhongsun@tju.edu.cn}')
out.append(r'\end{frontmatter}')

mode = 'front'            # front -> abstract -> body
abstract_lines = []
keywords = ''
seen_abstract = False
in_refs = False
pending_caption = None
pending_fig = None
code_buf = []
fig_counter = 1
refs = {}

def flush_code():
    global code_buf
    if code_buf:
        out.append(r'\begin{verbatim}')
        out.append('\n'.join(code_buf))
        out.append(r'\end{verbatim}')
        code_buf = []

for child in children:
    if child.tag == qn('w:tbl'):
        flush_code()
        tbl = Table(child, d)
        cap = pending_caption or ''
        out.append(table_to_latex(tbl, cap))
        pending_caption = None
        continue

    if child.tag != qn('w:p'):      # skip sectPr / other non-paragraph nodes
        continue

    # paragraph
    para = Paragraph(child, d)
    st = para.style.name if para.style else ''
    text = para.text.strip()

    if not seen_abstract:
        if st == 'Heading 1' and text.lower().startswith('abstract'):
            out.append(r'\begin{abstract}')
            seen_abstract = True
            mode = 'abstract'
        continue  # other preamble text skipped (already captured)

    if st.startswith('Heading'):
        flush_code()
        if mode == 'abstract':
            if abstract_lines:
                out.append(' '.join(abstract_lines))
            out.append(r'\end{abstract}')
            if keywords:
                out.append(r'\noindent\textbf{Keywords:} %s\par' % keywords)
            mode = 'body'
        try:
            level = int(''.join(ch for ch in st if ch.isdigit()) or 1)
        except ValueError:
            level = 1
        h = strip_number(text)
        if h.lower() == 'references':
            in_refs = True
            continue
        cmd = {1: 'section', 2: 'subsection', 3: 'subsubsection'}.get(level, 'section')
        out.append(r'\%s{%s}' % (cmd, h))
        continue

    # image?
    if has_drawing(para):
        pending_fig = extract_image(para, fig_counter)
        fig_counter += 1
        continue

    # figure caption?
    if pending_fig and re.match(r'^Figure\s+\d+\.', text, re.I):
        cap = strip_fig(text)
        out.append(r'\begin{figure}[htbp]')
        out.append(r'\centering')
        out.append(r'\includegraphics[width=0.7\textwidth]{%s}' % os.path.basename(pending_fig))
        out.append(r'\caption{%s}' % latex_escape(cap))
        out.append(r'\end{figure}')
        pending_fig = None
        continue

    # table caption?
    if re.match(r'^Table\s+\d+\.', text, re.I):
        pending_caption = text
        continue

    if not text:
        continue

    # code block?
    if looks_like_code(text):
        code_buf.append(text)
        continue

    flush_code()

    if mode == 'abstract':
        if text.lower().startswith('keywords:'):
            keywords = latex_escape(text[len('keywords:'):].strip())
        else:
            abstract_lines.append(apply_citations(runs_to_latex(para)))
        continue

    if in_refs:
        m = re.match(r'^\[(\d+)\]\s*(.*)', text)
        if m:
            refs[int(m.group(1))] = m.group(2)
        continue

    # normal body paragraph
    out.append(apply_citations(runs_to_latex(para)))

# close open abstract if still open
if mode == 'abstract':
    if abstract_lines:
        out.append(' '.join(abstract_lines))
    out.append(r'\end{abstract}')
    if keywords:
        out.append(r'\noindent\textbf{Keywords:} %s\par' % keywords)

# bibliography
if refs:
    out.append(r'\begin{thebibliography}{%d}' % max(refs))
    for k in sorted(refs):
        out.append(r'\bibitem{ref%d}%s' % (k, latex_escape(refs[k])))
    out.append(r'\end{thebibliography}')

out.append(r'\end{document}')

with open(TEX, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out) + '\n')

print('Wrote', TEX)
print('refs found:', len(refs), '| figures:', fig_counter - 1)
