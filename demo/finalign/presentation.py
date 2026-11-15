"""Render corpus text as escaped content, never as executable markup."""

import html
import math
import re

from .core import CHUNK_WORDS, metrics

PREVIEW_WORDS = 80
SHOWN = 10
WORD = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = frozenset(
    "a about an and any are as at be been but by can could did do does for from had has have how i if in "
    "into is it its me my not of on or our should so than that the their them then there these they this "
    "to up was we were what when where which who why will with would you your".split())
STAGES = {"keywords": "Qwen keywords", "hyde": "Qwen hypothetical answer",
          "lexical": "BM25 fusion", "rerank": "monoT5 rerank"}
VARIANT_LABELS = {"keywords": "Keyword expansion · Qwen", "hyde": "Hypothetical answer · Qwen"}


def esc(value):
    return html.escape(str(value), quote=True)


def score_name(options):
    """Name the scale of the last FinAlign stage, so scores are never read as accuracy."""
    options = options or {}
    if options.get("rerank"):
        return "monoT5"
    if options.get("lexical"):
        return "Fused"
    return "BGE"


def pipeline_chain(options):
    """The stages of this run, in execution order."""
    options = options or {}
    expansions = [STAGES[k] for k in ("keywords", "hyde") if options.get(k)]
    chain = ([" + ".join(expansions)] if expansions else []) + ["BGE passage search"]
    chain += [STAGES[k] for k in ("lexical", "rerank") if options.get(k)]
    return chain


def query_terms(query):
    return {w.lower() for w in WORD.findall(query or "") if len(w) > 2 and w.lower() not in STOPWORDS}


def highlight(text, terms):
    """Escape `text`, marking words shared with the question. Only alphanumeric runs are wrapped."""
    if not terms:
        return esc(text)
    out, last = [], 0
    for match in WORD.finditer(text):
        word = match.group().lower()
        if word in terms or (word.endswith("s") and word[:-1] in terms) or word + "s" in terms:
            out += [esc(text[last:match.start()]), "<mark>", esc(match.group()), "</mark>"]
            last = match.end()
    out.append(esc(text[last:]))
    return "".join(out)


def rank_map(ranking):
    return {docno: position for position, (docno, _) in enumerate((ranking or [])[:SHOWN], 1)}


def fmt_ms(ms):
    return f"{ms / 1000:.2f} s" if ms >= 1000 else f"{ms:,.0f} ms"


# ----------------------------------------------------------------- result cells

def cells(index, ranking, passages=None, known=None, *, method="bm25", scale="BM25",
          other=None, other_name=None, terms=None):
    """One <article> per result. `other` maps docno -> rank in the opposite list, when available."""
    passages = passages or {}
    out = []
    for docno, score in (ranking or [])[:SHOWN]:
        text = index.documents[docno]
        words = index.snippet(docno, passages.get(docno)).split()
        preview = " ".join(words[:PREVIEW_WORDS]) + (" …" if len(words) > PREVIEW_WORDS else "")
        total_words = len(text.split())
        badge = ""
        if known and known["qrels"].get(docno, 0) > 0:
            badge = '<span class="fa-badge" title="Judged relevant in FiQA">Relevant</span>'
        if docno in passages:
            number = int(passages[docno].rsplit("%", 1)[1]) + 1
            count = max(1, math.ceil(total_words / CHUNK_WORDS))
            source = f"Passage {number} of {count}"
        else:
            source = "Document opening"
        xref = ""
        if other is not None and other_name:
            if docno in other:
                xref = (f'<span class="fa-xref fa-xref-hit"><i aria-hidden="true"></i>'
                        f'{esc(other_name)} #{other[docno]}</span>')
            else:
                xref = f'<span class="fa-xref">Not in {esc(other_name)} top {SHOWN}</span>'
        out.append(
            f'<article class="fa-cell fa-m-{method}"><header class="fa-cell-top">'
            f'<span class="fa-doc">FiQA {esc(docno)}</span>{badge}'
            f'<span class="fa-score" title="{esc(scale)} score. It orders this list only: scores are not '
            f'comparable between methods and are not probabilities.">'
            f'<span>{esc(scale)}</span> {float(score):.3f}</span></header>'
            f'<p class="fa-snippet">{highlight(preview, terms)}</p>'
            f'<footer class="fa-cell-foot">{xref}<span class="fa-src">{source}</span></footer>'
            f'<details><summary>Full document <span>{total_words:,} words</span></summary>'
            f'<div class="fa-document">{esc(text)}</div></details></article>')
    return out


def results_html(index, ranking, passages=None, known=None, **kwargs):
    """A single list, for callers that do not need the side-by-side board."""
    items = cells(index, ranking, passages, known, **kwargs)
    if not items:
        return notice_html("empty", "No documents matched this question.")
    return '<ol class="fa fa-results">' + "".join(f"<li>{item}</li>" for item in items) + "</ol>"


def notice_html(kind, title, body=""):
    role = "alert" if kind in ("error", "quota") else "status"
    body = f"<p>{body}</p>" if body else ""
    return f'<div class="fa fa-notice fa-notice-{kind}" role="{role}"><p class="fa-notice-title">{title}</p>{body}</div>'


# ----------------------------------------------------------------- comparison board

def _head(method, name, kind, description, state_text, state="done"):
    return (f'<label class="fa-head fa-m-{method}" for="fa-tab-{method}"><span class="fa-head-top">'
            f'<span class="fa-swatch" aria-hidden="true"></span><span class="fa-head-name">{name}</span>'
            f'<span class="fa-head-kind">{kind}</span>'
            f'<span class="fa-head-state fa-state-{state}">{state_text}</span></span>'
            f'<span class="fa-head-desc">{description}</span></label>')


def board_html(index, query="", lexical=None, neural=None, passages=None, known=None, options=None, *,
               state="idle", lexical_ms=None, neural_ms=None, notice=None):
    """Both top-10 lists in one grid, so rank n of one method sits beside rank n of the other.

    state: idle | waiting (BM25 shown, FinAlign pending) | done | failed (notice replaces FinAlign).
    """
    chain = pipeline_chain(options)
    if len(chain) == 1:
        fin_desc = "BGE passage search: each document is ranked by its best 250-word passage."
    else:
        fin_desc = '<span class="fa-chain">' + "".join(f"<span>{esc(stage)}</span>" for stage in chain) + "</span>"
    bm25_state = fmt_ms(lexical_ms) if lexical_ms is not None else ""
    fin_state, fin_kind = "", "done"
    if state == "waiting":
        fin_state, fin_kind = "waiting for GPU", "waiting"
    elif state == "failed":
        fin_state, fin_kind = "not run", "failed"
    elif neural_ms is not None:
        fin_state = fmt_ms(neural_ms)
    heads = ('<div class="fa-heads"><span class="fa-gutter" aria-hidden="true"></span>'
             + _head("bm25", "BM25", "Lexical baseline",
                     "Terrier BM25 over whole documents: ranks by term overlap with the question.", bm25_state)
             + _head("fin", "FinAlign", "Neural pipeline", fin_desc, fin_state, fin_kind) + '</div>')
    tabs = ('<input class="fa-tab" type="radio" name="fa-tab" id="fa-tab-bm25" checked aria-label="Show BM25 results">'
            '<input class="fa-tab" type="radio" name="fa-tab" id="fa-tab-fin" aria-label="Show FinAlign results">')
    if state == "idle":
        body = ('<div class="fa-idle"><p>Both top-10 lists appear here, rank beside rank.</p>'
                '<p>BM25 answers first. FinAlign follows when a ZeroGPU slot is free.</p></div>')
        return f'<section class="fa fa-board fa-board-idle" aria-label="Ranking comparison">{tabs}{heads}{body}</section>'

    terms = query_terms(query)
    left = cells(index, lexical, known=known, method="bm25", scale="BM25", terms=terms,
                 other=rank_map(neural) if neural is not None else None, other_name="FinAlign")
    right = []
    if state == "done":
        right = cells(index, neural, passages, known, method="fin", scale=score_name(options), terms=terms,
                      other=rank_map(lexical), other_name="BM25")
    skeleton = ('<div class="fa-cell fa-m-fin fa-skeleton" aria-hidden="true"><span></span><span></span>'
                '<span></span><span></span></div>')
    empty = '<div class="fa-cell fa-m-{} fa-cell-empty" aria-hidden="true"></div>'
    count = SHOWN if state == "waiting" else max(len(left), len(right))
    rows = []
    for i in range(count):
        a = left[i] if i < len(left) else empty.format("bm25")
        b = skeleton if state == "waiting" else (right[i] if i < len(right) else empty.format("fin"))
        rows.append(f'<li class="fa-row"><span class="fa-rank" aria-label="Rank {i + 1}">{i + 1}</span>{a}{b}</li>')
    extra = ""
    if not left:
        extra += ('<div class="fa-row fa-row-note"><span class="fa-rank"></span><div class="fa-cell fa-m-bm25">'
                  + notice_html("empty", "BM25 found no document sharing a term with this question.")
                  + '</div><div class="fa-cell fa-m-fin"></div></div>')
    if state == "failed" and notice:
        # Wide screens show the notice beside the first BM25 result; narrow ones above both tabs.
        narrow = " fa-narrow-only" if rows else ""
        extra += (f'<div class="fa-row fa-row-note{narrow}"><span class="fa-rank"></span>'
                  '<div class="fa-cell fa-m-bm25"></div><div class="fa-cell fa-m-fin">'
                  + notice_html(*notice) + '</div></div>')
        if rows:
            rows[0] = rows[0].replace(empty.format("fin"), '<div class="fa-cell fa-m-fin fa-wide-only">'
                                      + notice_html(*notice) + '</div>', 1)
    live = {"waiting": "BM25 results are shown. FinAlign is waiting for a GPU.",
            "done": "BM25 and FinAlign results are shown.", "failed": "FinAlign did not run."}[state]
    return (f'<section class="fa fa-board fa-board-{state}" aria-label="Ranking comparison">{tabs}{heads}'
            f'<p class="fa-sr" role="status">{live}</p>{extra}<ol class="fa-rows">{"".join(rows)}</ol></section>')


# ----------------------------------------------------------------- summary strip

def rank_map_svg(lexical, neural):
    """A small slope chart: BM25 ranks on the left, FinAlign on the right, lines join shared documents."""
    left = [d for d, _ in (lexical or [])[:SHOWN]]
    right = [d for d, _ in (neural or [])[:SHOWN]]
    width, height, x1, x2, top, step = 150, 108, 15, 135, 9, 10

    def y(i):
        return top + i * step

    parts = []
    shared = set(left) & set(right)
    for i, docno in enumerate(left):
        if docno in shared:
            parts.append(f'<line x1="{x1}" y1="{y(i)}" x2="{x2}" y2="{y(right.index(docno))}"/>')
    for i in range(SHOWN):
        if i < len(left):
            parts.append(f'<circle class="bm25{" on" if left[i] in shared else ""}" cx="{x1}" cy="{y(i)}" r="2.6"/>')
        if neural is None:
            parts.append(f'<circle class="ghost" cx="{x2}" cy="{y(i)}" r="2.6"/>')
        elif i < len(right):
            parts.append(f'<circle class="fin{" on" if right[i] in shared else ""}" cx="{x2}" cy="{y(i)}" r="2.6"/>')
    label = (f"Rank map. {len(shared)} documents appear in both lists." if neural is not None
             else "Rank map, waiting for FinAlign.")
    return (f'<svg class="fa-rankmap" viewBox="0 0 {width} {height}" role="img" aria-label="{label}">'
            + "".join(parts) + '</svg>')


def _metric_tile(known, lexical, neural):
    if not known:
        return ('<div class="fa-tile"><h3>Relevance</h3><p class="fa-tile-lead">No judgments</p>'
                '<p class="fa-tile-note">A custom question has no FiQA relevance labels, so no quality '
                'metric is shown.</p></div>')
    a = metrics([d for d, _ in lexical], known["qrels"])
    b = metrics([d for d, _ in neural], known["qrels"]) if neural is not None else None
    if not a:
        return ('<div class="fa-tile"><h3>Relevance</h3><p class="fa-tile-lead">No judgments</p>'
                '<p class="fa-tile-note">This FiQA question has no positive judgments in the retained corpus.</p></div>')
    rows = []
    for name in ("nDCG@10", "Recall@10"):
        left, right = a[name], (b[name] if b else None)
        best_l = ' class="fa-best"' if right is not None and left > right else ""
        best_r = ' class="fa-best"' if right is not None and right > left else ""
        shown = f"{right:.3f}" if right is not None else "…"
        rows.append(f'<tr><th scope="row">{name}</th><td{best_l}>{left:.3f}</td><td{best_r}>{shown}</td></tr>')
    return ('<div class="fa-tile"><h3>Relevance · this question only</h3><table class="fa-metrics"><thead><tr>'
            '<td></td><th scope="col"><i class="fa-m-bm25"></i>BM25</th>'
            '<th scope="col"><i class="fa-m-fin"></i>FinAlign</th></tr></thead><tbody>' + "".join(rows)
            + f'</tbody></table><p class="fa-tile-note">FiQA {esc(known["split"])} query {esc(known["qid"])}. '
            'Unlabelled documents are unjudged, not necessarily irrelevant.</p></div>')


def summary_html(query, known=None, lexical=None, neural=None, *, state="waiting", lexical_ms=None,
                 timings=None, server_s=None):
    """The question, then three tiles: overlap, per-question metrics (FiQA questions only) and time."""
    lexical = lexical or []
    if neural is not None:
        shared = len(set(rank_map(lexical)) & set(rank_map(neural)))
        overlap = (f'<p class="fa-big">{shared}<small>/ {SHOWN}</small></p>'
                   '<p class="fa-tile-note">documents in both top-10 lists</p>')
    else:
        pending = "FinAlign did not run" if state == "failed" else "waiting for FinAlign"
        overlap = f'<p class="fa-big fa-pending">–<small>/ {SHOWN}</small></p><p class="fa-tile-note">{pending}</p>'
    tile_overlap = (f'<div class="fa-tile fa-tile-overlap"><div><h3>Shared documents</h3>{overlap}</div>'
                    f'{rank_map_svg(lexical, neural)}</div>')

    neural_ms = sum(timings.values()) if timings else None
    extra = ""
    if neural_ms is not None:
        total = neural_ms or 1
        bar = "".join(f'<i style="flex:{ms / total:.4f}" title="{esc(stage)}: {fmt_ms(ms)}"></i>'
                      for stage, ms in timings.items())
        fin_time = f"<dd>{fmt_ms(neural_ms)}</dd>"
        extra = (f'<div class="fa-stagebar" role="img" aria-label="FinAlign time by stage">{bar}</div>'
                 f'<p class="fa-tile-note">{server_s:.2f} s on the server, including GPU wait.</p>')
    elif state == "failed":
        fin_time = '<dd class="fa-pending">not run</dd>'
    else:
        fin_time = '<dd class="fa-pending fa-pulse">waiting for GPU</dd>'
        extra = '<p class="fa-tile-note">FinAlign runs on a shared ZeroGPU and may wait in a queue.</p>'
    tile_time = ('<div class="fa-tile"><h3>Time</h3><dl class="fa-times"><div><dt><i class="fa-m-bm25"></i>BM25</dt>'
                 f'<dd>{fmt_ms(lexical_ms or 0)}</dd></div><div><dt><i class="fa-m-fin"></i>FinAlign</dt>'
                 f'{fin_time}</div></dl>{extra}</div>')

    return ('<section class="fa fa-summary" aria-label="Summary of this search">'
            f'<p class="fa-q"><span>Results for</span> “{esc(query)}”</p>'
            f'<div class="fa-tiles">{tile_overlap}{_metric_tile(known, lexical, neural)}{tile_time}</div></section>')


def details_html(timings=None, variants=None, processing_s=None, server_s=None):
    if not timings:
        return '<div class="fa"><p class="fa-fine">Run a search to see stage timings and the generated query variants.</p></div>'
    longest = max(timings.values()) or 1
    rows = "".join(f'<tr><th scope="row">{esc(stage)}</th><td class="fa-bar"><i style="width:{100 * ms / longest:.1f}%">'
                   f'</i></td><td>{ms:,.1f} ms</td></tr>' for stage, ms in timings.items())
    out = ['<div class="fa fa-details"><section><h3>Latency by stage</h3>'
           f'<table class="fa-timings"><tbody>{rows}</tbody><tfoot><tr><th scope="row">Processing</th><td></td>'
           f'<td>{processing_s:.2f} s</td></tr><tr><th scope="row">Server, incl. GPU wait</th><td></td>'
           f'<td>{server_s:.2f} s</td></tr></tfoot></table>'
           '<p class="fa-fine">Measured on this run over the full FiQA corpus. Browser transfer is excluded.</p>'
           '</section><section><h3>Query variants</h3>']
    generated = {k: v for k, v in (variants or {}).items() if k != "original"}
    if generated:
        for name, text in generated.items():
            out.append(f'<div class="fa-variant"><h4>{esc(VARIANT_LABELS.get(name, name.title()))}</h4>'
                       f'<p>{esc(text)}</p></div>')
        out.append('<p class="fa-fine">Generated text is only a retrieval input. It is not an answer '
                   'and not financial guidance.</p>')
    else:
        out.append('<p class="fa-fine">Query expansion is off. BGE encodes the original question '
                   'with its retrieval instruction.</p>')
    out.append('</section></div>')
    return "".join(out)


# ----------------------------------------------------------------- styles

CSS = """
/* Tokens. Blue and orange tell the two methods apart; green marks relevance labels; the rest is slate. */
.gradio-container {
  --fa-bm25: #c2410c; --fa-fin: #2563eb; --fa-fin-soft: rgba(37, 99, 235, .07);
  --fa-ink: var(--body-text-color); --fa-muted: var(--body-text-color-subdued);
  --fa-line: var(--border-color-primary); --fa-line-strong: var(--neutral-300);
  --fa-surface: var(--background-fill-primary); --fa-sunken: var(--background-fill-secondary);
  --fa-mark: #fdf0a6;
  --fa-ok-fg: #166534; --fa-ok-bg: #dcfce7;
  --fa-warn-fg: #854d0e; --fa-warn-bg: #fef9c3; --fa-warn-line: #ca8a04;
  --fa-err-fg: #991b1b; --fa-err-bg: #fee2e2; --fa-err-line: #dc2626;
  --fa-radius: 10px;
  width: 100% !important; max-width: 1180px !important; margin: 0 auto !important;
}
.dark .gradio-container, .gradio-container.dark {
  --fa-bm25: #fb923c; --fa-fin: #60a5fa; --fa-fin-soft: rgba(96, 165, 250, .10);
  --fa-line-strong: var(--neutral-600); --fa-surface: var(--neutral-900); --fa-sunken: var(--neutral-800);
  --fa-mark: rgba(250, 204, 21, .24);
  --fa-ok-fg: #86efac; --fa-ok-bg: rgba(34, 197, 94, .15);
  --fa-warn-fg: #fde68a; --fa-warn-bg: rgba(234, 179, 8, .11);
  --fa-err-fg: #fecaca; --fa-err-bg: rgba(239, 68, 68, .13);
}

/* Custom HTML lives inside Gradio's .prose; spacing is owned by containers (gap), not by text elements. */
.gradio-container .html-container {padding: 0 !important;}
.gradio-container .fa {color: var(--fa-ink); font-size: 15px; line-height: 1.5;}
.gradio-container .fa * {color: inherit; box-sizing: border-box;}
.gradio-container .fa :is(p, h1, h2, h3, h4, ul, ol, li, dl, dt, dd, table) {margin: 0 !important; padding: 0 !important;}
.gradio-container .fa :is(ul, ol) {list-style: none;}
.fa .fa-sr, .fa.fa-sr {position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap;}
.fa .fa-label {font-weight: 600; font-size: 12.5px; line-height: 1.3; color: var(--fa-muted);}
.fa .fa-fine {font-size: 13.5px; line-height: 1.5; color: var(--fa-muted);}
.fa i {font-style: normal;}

/* Keyboard focus is always visible. */
.gradio-container :is(a, button, summary, input, textarea, label, [tabindex]):focus-visible {
  outline: 2px solid var(--fa-fin) !important; outline-offset: 2px;
}

/* ZeroGPU's "waiting / acquired a GPU" toasts repeat what the board already shows; errors stay. */
.toast-wrap .toast-body.info, .toast-wrap .toast-body.success {display: none !important;}

/* ---------- Masthead */
#fa-masthead {display: grid; gap: 12px; padding: 28px 0 14px;}
.fa .fa-top {display: flex; justify-content: space-between; align-items: center; gap: 12px 24px; flex-wrap: wrap;}
.fa .fa-kicker {font-weight: 600; font-size: 11px; line-height: 1.2; letter-spacing: .16em; text-transform: uppercase; color: var(--fa-muted);}
.fa .fa-top nav {display: flex; gap: 22px; font-size: 14px;}
.fa .fa-top nav a {color: var(--fa-ink); text-decoration: none; padding-bottom: 2px; border-bottom: 1px solid var(--fa-line-strong);}
.fa .fa-top nav a:hover {border-bottom-color: var(--fa-ink);}
.fa h1 {font-weight: 700; font-size: clamp(36px, 5vw, 52px); line-height: 1.05; letter-spacing: -.04em;}
.fa .fa-lede {font-size: 16px; line-height: 1.6; max-width: 70ch;}
.fa .fa-meta {display: flex; flex-wrap: wrap; gap: 4px 20px; font-size: 13px; color: var(--fa-muted); font-variant-numeric: tabular-nums;}
.fa .fa-meta b {color: var(--fa-ink); font-weight: 600;}

/* ---------- Search bar: one surface holding the question and the primary action */
#fa-app {gap: 12px;}
#fa-searchbar {align-items: center; gap: 10px; flex-wrap: nowrap; padding: 6px 6px 6px 18px; background: var(--fa-surface);
  border: 1px solid var(--fa-line-strong); border-radius: var(--fa-radius); transition: border-color .15s;}
#fa-searchbar:focus-within {border-color: var(--fa-fin); box-shadow: 0 0 0 3px var(--fa-fin-soft);}
#fa-query {flex: 1 1 auto; min-width: 0 !important;}
#fa-query, #fa-query * {border: 0 !important; box-shadow: none !important; background: transparent !important;}
#fa-query :is(textarea, input) {background: transparent !important; border: 0 !important; box-shadow: none !important;
  outline: 0 !important; padding: 11px 0 !important; font-size: 17px; line-height: 1.4; color: var(--fa-ink); resize: none;}
#fa-query :is(textarea, input)::placeholder {color: var(--fa-muted); opacity: .75;}
#fa-submit {flex: 0 0 auto; height: 46px; padding: 0 22px; border-radius: 7px; font-size: 15px; font-weight: 600;}

/* ---------- Example questions */
#fa-examples {align-items: center; gap: 8px; flex-wrap: wrap; padding: 2px 2px 0;}
#fa-examples > * {flex: 0 1 auto !important; min-width: 0 !important; width: auto !important;}
#fa-examples button {padding: 6px 11px; border: 1px solid var(--fa-line-strong); border-radius: 6px; background: transparent;
  box-shadow: none; font-size: 13.5px; font-weight: 450; line-height: 1.3; color: var(--fa-ink); white-space: normal; text-align: left;}
#fa-examples button:hover {border-color: var(--fa-fin); color: var(--fa-fin);}
#fa-notice-slot:empty, #fa-summary-slot:empty {display: none;}

/* ---------- Pipeline: five stages in execution order, four of them switchable */
#fa-pipe {gap: 10px; margin-top: 18px;}
#fa-pipe-head {align-items: center; gap: 6px; flex-wrap: wrap; padding: 0 2px;}
#fa-pipe-head > .fa-inline {flex: 1 1 320px !important; min-width: 0 !important;}
.fa .fa-pipe-hint {margin-left: 14px; font-size: 13.5px; color: var(--fa-muted);}
#fa-pipe-head button {flex: 0 0 auto; padding: 5px 10px; border: 1px solid transparent; border-radius: 6px; background: transparent;
  box-shadow: none; font-size: 13px; font-weight: 500; color: var(--fa-muted);}
#fa-pipe-head button:hover {color: var(--fa-ink); border-color: var(--fa-line-strong);}

#fa-stages {display: grid; grid-template-columns: 2fr 1fr 2fr; gap: 0; align-items: stretch; background: var(--fa-surface);
  border: 1px solid var(--fa-line); border-radius: var(--fa-radius); overflow: hidden;}
#fa-stages > * {min-width: 0 !important; margin: 0 !important;}
#fa-stages .form {display: grid; grid-template-columns: 1fr 1fr; gap: 0; border: 0; border-radius: 0; background: none; box-shadow: none; overflow: visible;}
#fa-stages .block {position: relative; display: grid; gap: 5px; align-content: start; padding: 14px 18px 15px !important; margin: 0 !important;
  border: 0 !important; border-radius: 0 !important; background: transparent !important; box-shadow: none !important;
  min-width: 0 !important; transition: background .15s;}
#fa-stages .form + .block, #fa-stages .block + .form, #fa-stages .form > .block + .block {border-left: 1px solid var(--fa-line) !important;}
#fa-stages > .block::before, #fa-stages .form > .block::before {font-weight: 600; font-size: 12px; line-height: 1; color: var(--fa-muted);}
#fa-stages .form:first-child > .block::before {content: "Expand";}
#fa-stages .form:last-child > .block:first-child::before {content: "Fuse";}
#fa-stages .form:last-child > .block:last-child::before {content: "Rerank";}
#fa-stages > .fa-stage-core::before {content: "Retrieve · always on";}
#fa-stages .checkbox-container {display: flex; flex-direction: row-reverse; justify-content: space-between; align-items: center; gap: 10px;
  cursor: pointer; margin: 0;}
#fa-stages .checkbox-container::after {content: ""; position: absolute; inset: 0;}
#fa-stages .label-text {font-size: 15px; font-weight: 600; line-height: 1.25; margin: 0; color: var(--fa-ink);}
#fa-stages .info-text {font-size: 13px; line-height: 1.4; color: var(--fa-muted); margin: 0;}
#fa-stages input[type=checkbox] {width: 18px; height: 18px; border-radius: 5px; flex: none; margin: 0; border-color: var(--fa-line-strong);}
#fa-stages .block:has(input:checked) {background: var(--fa-fin-soft) !important;}
#fa-stages .block:has(input:checked)::before {color: var(--fa-fin);}
#fa-stages .block:has(input:disabled) {opacity: .5;}
#fa-stages .block.fa-stage:hover {background: var(--fa-sunken) !important;}
#fa-stages .block.fa-stage:has(input:checked):hover {background: var(--fa-fin-soft) !important;}
#fa-stages > .fa-stage-core {background: var(--fa-fin-soft) !important;}
#fa-stages > .fa-stage-core::before {color: var(--fa-fin);}
.fa.fa-stage-fixed {display: grid; gap: 5px;}
.fa .fa-stage-title {display: flex; justify-content: space-between; align-items: center; gap: 10px; font-size: 15px; font-weight: 600; line-height: 1.25;}
.fa .fa-stage-title i {font-weight: 600; font-size: 11.5px; line-height: 1; color: var(--fa-fin); white-space: nowrap;}
.fa .fa-stage-info {font-size: 13px; line-height: 1.4; color: var(--fa-muted);}

/* ---------- Summary strip */
.fa.fa-summary {display: grid; gap: 14px; padding-top: 22px;}
.fa .fa-q {font-weight: 600; font-size: 20px; line-height: 1.35; letter-spacing: -.01em; overflow-wrap: anywhere; padding: 0 2px !important;}
.fa .fa-q span {font-weight: 600; font-size: 12.5px; line-height: 1.3; color: var(--fa-muted); display: block; padding-bottom: 4px; letter-spacing: 0;}
.fa .fa-tiles {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); background: var(--fa-surface);
  border: 1px solid var(--fa-line); border-radius: var(--fa-radius);}
.fa .fa-tile {display: grid; gap: 8px; align-content: start; padding: 18px 22px 18px; min-width: 0;}
.fa .fa-tile + .fa-tile {border-left: 1px solid var(--fa-line);}
.fa .fa-tile h3 {font-weight: 600; font-size: 12.5px; line-height: 1.3; color: var(--fa-muted);}
.fa .fa-tile-note {font-size: 13px; line-height: 1.45; color: var(--fa-muted);}
.fa .fa-tile-lead {font-weight: 600; font-size: 18px; line-height: 1.3;}
.fa .fa-tile-overlap {grid-template-columns: minmax(0, 1fr) auto; align-items: center; gap: 12px;}
.fa .fa-tile-overlap > div {display: grid; gap: 8px;}
.fa .fa-big {font-weight: 700; font-size: 38px; line-height: 1; letter-spacing: -.03em; font-variant-numeric: tabular-nums;}
.fa .fa-big small {font-weight: 500; font-size: 17px; line-height: 1; color: var(--fa-muted); margin-left: 5px; letter-spacing: 0;}
.fa .fa-pending {color: var(--fa-muted);}
.fa .fa-rankmap {width: 150px; height: 108px; flex: none;}
.fa .fa-rankmap line {stroke: var(--fa-muted); stroke-width: 1.2; opacity: .7;}
.fa .fa-rankmap circle {fill: var(--fa-surface); stroke-width: 1.3;}
.fa .fa-rankmap .bm25 {stroke: var(--fa-bm25);}
.fa .fa-rankmap .fin {stroke: var(--fa-fin);}
.fa .fa-rankmap .bm25.on {fill: var(--fa-bm25);}
.fa .fa-rankmap .fin.on {fill: var(--fa-fin);}
.fa .fa-rankmap .ghost {stroke: var(--fa-line-strong); stroke-dasharray: 2 2;}
.gradio-container .fa .fa-metrics {width: 100% !important; border-collapse: collapse; border: 0 !important; background: none !important;
  font-variant-numeric: tabular-nums;}
.gradio-container .fa .fa-metrics tr {border: 0 !important; background: none !important;}
.gradio-container .fa .fa-metrics :is(th, td) {padding: 5px 0 !important; border: 0 !important; background: none !important;
  text-align: right; font-weight: 500; font-size: 16px; line-height: 1.2;}
.gradio-container .fa .fa-metrics tbody tr + tr :is(th, td) {border-top: 1px solid var(--fa-line) !important;}
.gradio-container .fa .fa-metrics tbody th {text-align: left; font-weight: 400; font-size: 14px; line-height: 1.2; color: var(--fa-ink);}
.gradio-container .fa .fa-metrics thead th {font-weight: 500; font-size: 12.5px; line-height: 1.2; color: var(--fa-muted); padding-bottom: 2px !important;}
.gradio-container .fa .fa-metrics .fa-best {font-weight: 700;}
.fa i.fa-m-bm25, .fa i.fa-m-fin {display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 6px; background: var(--fa-bm25);}
.fa i.fa-m-fin {background: var(--fa-fin);}
.fa .fa-times {display: grid; gap: 4px;}
.fa .fa-times > div {display: flex; align-items: baseline; justify-content: space-between; gap: 12px;}
.fa .fa-times dt {font-size: 14px;}
.fa .fa-times dd {font-weight: 500; font-size: 16px; line-height: 1.3; font-variant-numeric: tabular-nums;}
.fa .fa-times dd.fa-pending {font-weight: 400; font-size: 14px; line-height: 1.6;}
.fa .fa-stagebar {display: flex; gap: 2px; height: 6px; border-radius: 3px; overflow: hidden;}
.fa .fa-stagebar i {min-width: 2px; background: var(--fa-fin);}
.fa .fa-stagebar i:nth-child(even) {opacity: .5;}
.fa .fa-pulse {animation: fa-pulse 1.3s ease-in-out infinite;}

/* ---------- Board: both lists in one grid, rank beside rank */
.fa.fa-board {--fa-cols: 48px minmax(0, 1fr) minmax(0, 1fr); background: var(--fa-surface); border: 1px solid var(--fa-line);
  border-radius: var(--fa-radius);}
.fa .fa-tab {display: none;}
.fa .fa-heads {display: grid; grid-template-columns: var(--fa-cols); border-bottom: 1px solid var(--fa-line);
  position: sticky; top: 0; z-index: 3; background: var(--fa-surface); border-radius: var(--fa-radius) var(--fa-radius) 0 0;}
.fa .fa-gutter {display: none;}
.fa .fa-head.fa-m-bm25 {grid-column: 1 / 3; padding-left: 68px; border-top-left-radius: var(--fa-radius);}
.fa .fa-head {display: grid; gap: 6px; align-content: start; padding: 16px 20px 14px; min-width: 0; --m: var(--fa-bm25);
  box-shadow: inset 0 3px 0 var(--m); cursor: default;}
.fa .fa-head.fa-m-fin {--m: var(--fa-fin); border-left: 1px solid var(--fa-line); border-top-right-radius: var(--fa-radius);}
.fa .fa-head-top {display: flex; align-items: center; gap: 10px; flex-wrap: wrap;}
.fa .fa-swatch {width: 10px; height: 10px; border-radius: 2px; background: var(--m);}
.fa .fa-head-name {font-weight: 700; font-size: 20px; line-height: 1.1; letter-spacing: -.02em;}
.fa .fa-head-kind {font-weight: 500; font-size: 12.5px; line-height: 1; color: var(--fa-muted); padding: 3px 7px; border: 1px solid var(--fa-line); border-radius: 999px;}
.fa .fa-head-state {margin-left: auto; font-weight: 500; font-size: 13px; line-height: 1; color: var(--fa-muted); white-space: nowrap; font-variant-numeric: tabular-nums;}
.fa .fa-head-state:empty {display: none;}
.fa .fa-state-waiting {color: var(--fa-fin); animation: fa-pulse 1.3s ease-in-out infinite;}
.fa .fa-state-failed {color: var(--fa-err-fg);}
.fa .fa-head-desc {font-size: 13.5px; line-height: 1.45; color: var(--fa-muted);}
.fa .fa-chain {display: flex; flex-wrap: wrap; gap: 4px 0; align-items: center;}
.fa .fa-chain span {color: var(--fa-ink);}
.fa .fa-chain span + span::before {content: "→"; color: var(--fa-muted); padding: 0 7px;}
.fa .fa-rows {display: grid;}
.fa .fa-row {display: grid; grid-template-columns: var(--fa-cols);}
.fa .fa-row + .fa-row {border-top: 1px solid var(--fa-line);}
.fa .fa-row-note {border-bottom: 1px solid var(--fa-line);}
.fa .fa-narrow-only {display: none;}
.fa .fa-rank {padding-top: 17px; text-align: center; font-weight: 700; font-size: 15px; line-height: 1.2; color: var(--fa-ink); font-variant-numeric: tabular-nums;}
.fa .fa-cell {display: grid; gap: 8px; align-content: start; padding: 16px 20px 15px; min-width: 0; --m: var(--fa-bm25); --o: var(--fa-fin);}
.fa .fa-cell.fa-m-fin {--m: var(--fa-fin); --o: var(--fa-bm25); border-left: 1px solid var(--fa-line);}
.fa .fa-cell-top {display: flex; align-items: center; flex-wrap: wrap; gap: 6px 10px; font-size: 13px; line-height: 1.2; color: var(--fa-muted); font-variant-numeric: tabular-nums;}
.fa .fa-badge {font-weight: 600; font-size: 11.5px; line-height: 1; padding: 3px 7px; border-radius: 5px; color: var(--fa-ok-fg); background: var(--fa-ok-bg);}
.fa .fa-score {margin-left: auto; white-space: nowrap; cursor: help; color: var(--fa-ink); font-variant-numeric: tabular-nums;}
.fa .fa-score span {color: var(--fa-muted); margin-right: 3px;}
.fa .fa-snippet {font-size: 15px; line-height: 1.6; overflow-wrap: anywhere;
  display: -webkit-box; -webkit-line-clamp: 5; -webkit-box-orient: vertical; overflow: hidden;}
.fa mark {background: var(--fa-mark); color: inherit; padding: 0 2px; border-radius: 3px;}
.fa .fa-cell-foot {display: flex; flex-wrap: wrap; align-items: center; gap: 4px 14px; font-size: 13px; color: var(--fa-muted);}
.fa .fa-xref {display: inline-flex; align-items: center; gap: 6px;}
.fa .fa-xref-hit {color: var(--fa-ink); font-weight: 500;}
.fa .fa-xref i {width: 8px; height: 8px; border-radius: 2px; background: var(--o);}
.fa details summary {display: inline-flex; align-items: center; gap: 7px; list-style: none; cursor: pointer; font-size: 13px;
  font-weight: 500; color: var(--fa-ink); padding: 2px 0;}
.fa details summary::-webkit-details-marker {display: none;}
.fa details summary::before {content: ""; width: 6px; height: 6px; border: solid currentColor; border-width: 0 1.5px 1.5px 0;
  transform: rotate(-45deg); transition: transform .15s; margin-right: 1px;}
.fa details[open] summary::before {transform: rotate(45deg);}
.fa details summary span {font-weight: 400; color: var(--fa-muted);}
.fa details summary:hover {text-decoration: underline; text-underline-offset: 3px;}
.fa .fa-document {margin-top: 10px; padding: 14px 16px; border-radius: 10px; background: var(--fa-sunken); white-space: pre-wrap;
  overflow-wrap: anywhere; font-size: 14px; line-height: 1.65; max-height: 440px; overflow-y: auto;}
.fa .fa-skeleton span {display: block; height: 11px; border-radius: 5px; background: var(--fa-sunken); animation: fa-pulse 1.4s ease-in-out infinite;}
.fa .fa-skeleton span:nth-child(1) {width: 34%;}
.fa .fa-skeleton span:nth-child(4) {width: 62%;}
.fa .fa-idle {display: grid; gap: 4px; justify-items: center; text-align: center; padding: 48px 24px 56px; color: var(--fa-muted);}
.fa .fa-idle p:first-child {font-weight: 600; font-size: 16px; line-height: 1.3; color: var(--fa-ink);}
.fa.fa-results {display: grid;}
.fa.fa-results li + li {border-top: 1px solid var(--fa-line);}

/* ---------- Notices: validation, GPU quota, failures */
.fa.fa-notice, .fa .fa-notice {display: grid; gap: 4px; padding: 14px 16px; border-radius: 10px; border-left: 3px solid var(--fa-line-strong);
  background: var(--fa-sunken); font-size: 14px; line-height: 1.5;}
.fa .fa-notice-title, .fa.fa-notice .fa-notice-title {font-weight: 600;}
.fa.fa-notice-error, .fa .fa-notice-error {background: var(--fa-err-bg); border-left-color: var(--fa-err-line);}
.fa.fa-notice-error .fa-notice-title, .fa .fa-notice-error .fa-notice-title {color: var(--fa-err-fg);}
.fa.fa-notice-quota, .fa .fa-notice-quota {background: var(--fa-warn-bg); border-left-color: var(--fa-warn-line);}
.fa.fa-notice-quota .fa-notice-title, .fa .fa-notice-quota .fa-notice-title {color: var(--fa-warn-fg);}
.fa .fa-notice small, .fa.fa-notice small {display: block; padding-top: 6px; font-size: 12.5px; color: var(--fa-muted);}

/* ---------- Run details, accordions, footer */
.fa.fa-details {display: grid; grid-template-columns: minmax(260px, 1fr) minmax(280px, 1.4fr); gap: 32px; padding: 6px 2px 4px;}
.fa.fa-details section {display: grid; gap: 10px; align-content: start;}
.fa.fa-details h3 {font-weight: 600; font-size: 12.5px; line-height: 1.3; color: var(--fa-muted);}
.fa.fa-details h4 {font-size: 13px; font-weight: 600;}
.fa .fa-variant {display: grid; gap: 4px;}
.fa .fa-variant p {font-size: 14.5px; line-height: 1.6; overflow-wrap: anywhere;}
.gradio-container .fa .fa-timings {width: 100% !important; border-collapse: collapse; border: 0 !important; font-size: 14px; font-variant-numeric: tabular-nums;}
.gradio-container .fa .fa-timings tr {border: 0 !important; background: none !important;}
.gradio-container .fa .fa-timings :is(th, td) {padding: 6px 0 !important; border: 0 !important; border-bottom: 1px solid var(--fa-line) !important;
  background: none !important; font-weight: 400; text-align: left;}
.gradio-container .fa .fa-timings td:last-child {text-align: right; white-space: nowrap;}
.gradio-container .fa .fa-timings td.fa-bar {width: 38%; padding: 0 14px !important;}
.fa .fa-bar i {display: block; height: 6px; min-width: 2px; border-radius: 3px; background: var(--fa-fin); opacity: .75;}
.gradio-container .fa .fa-timings tfoot :is(th, td) {font-weight: 600;}
.fa-accordion {border-radius: var(--fa-radius) !important; border-color: var(--fa-line) !important;}
.fa-accordion .label-wrap {text-align: left; font-weight: 500;}
#fa-footer {padding-top: 18px; margin-top: 8px; border-top: 1px solid var(--fa-line); font-size: 13px;}
#fa-footer, #fa-footer p {color: var(--fa-muted);}
#fa-footer a {color: var(--fa-ink);}

@keyframes fa-pulse {50% {opacity: .45;}}
@media (prefers-reduced-motion: reduce) {
  .fa .fa-skeleton span, .fa .fa-pulse, .fa .fa-state-waiting {animation: none;}
}

/* ---------- Narrow screens: the two lists become tabs, so only one top 10 is on screen at a time */
@media (max-width: 820px) {
  .main.fillable {padding-left: 16px !important; padding-right: 16px !important;}
  #fa-masthead {padding: 22px 0 8px; gap: 12px;}
  .fa .fa-lede {font-size: 16px;}
  #fa-searchbar {flex-wrap: wrap; padding: 6px 6px 6px 14px;}
  #fa-query {flex-basis: 100%;}
  #fa-query :is(textarea, input) {font-size: 17px; padding: 10px 0 6px !important;}
  #fa-submit {flex: 1 1 100%; width: 100%; margin-left: -8px; height: 46px;}
  #fa-examples {padding-left: 2px;}
  .fa .fa-pipe-hint {display: block; margin: 6px 0 0;}
  .fa .fa-q {font-size: 18px;}
  .fa .fa-tiles {grid-template-columns: 1fr;}
  .fa .fa-tile + .fa-tile {border-left: 0; border-top: 1px solid var(--fa-line);}
  .fa.fa-board {--fa-cols: 34px minmax(0, 1fr);}
  .fa .fa-tab {display: block; position: absolute; opacity: 0; pointer-events: none;}
  .fa .fa-heads {grid-template-columns: 1fr 1fr;}
  .fa .fa-head.fa-m-bm25 {grid-column: auto; padding-left: 14px;}
  .fa .fa-head-top {gap: 4px 8px;}
  #fa-stages {grid-template-columns: 1fr;}
  #fa-stages .form {grid-template-columns: 1fr;}
  #fa-stages .form + .block, #fa-stages .block + .form, #fa-stages .form > .block + .block {border-left: 0 !important; border-top: 1px solid var(--fa-line) !important;}
  .fa .fa-row-note {grid-template-columns: 1fr;}
  .fa .fa-narrow-only {display: grid;}
  .fa .fa-rows .fa-wide-only {display: none;}
  .fa .fa-row-note .fa-cell:empty, .fa .fa-row-note .fa-rank {display: none;}
  .fa .fa-row-note .fa-cell, .fa .fa-row-note .fa-cell.fa-m-fin {padding: 12px;}
  .fa .fa-head {padding: 14px 14px 12px; cursor: pointer; box-shadow: none; border-bottom: 3px solid transparent; opacity: .62;}
  .fa .fa-head.fa-m-fin {border-left: 0;}
  .fa .fa-head-name {font-size: 20px;}
  .fa .fa-head-kind, .fa .fa-head-desc {display: none;}
  .fa #fa-tab-bm25:checked ~ .fa-heads .fa-m-bm25, .fa #fa-tab-fin:checked ~ .fa-heads .fa-m-fin {opacity: 1; border-bottom-color: var(--m);}
  .fa #fa-tab-bm25:focus-visible ~ .fa-heads .fa-m-bm25, .fa #fa-tab-fin:focus-visible ~ .fa-heads .fa-m-fin {outline: 2px solid var(--fa-fin); outline-offset: -2px;}
  .fa #fa-tab-bm25:checked ~ .fa-rows .fa-m-fin, .fa #fa-tab-fin:checked ~ .fa-rows .fa-m-bm25 {display: none;}
  .fa.fa-board-failed #fa-tab-fin:checked ~ .fa-rows {display: none;}
  .fa .fa-cell, .fa .fa-cell.fa-m-fin {padding: 16px 14px 14px 4px; border-left: 0;}
  .fa .fa-rank {padding-top: 16px;}
  .fa .fa-snippet {-webkit-line-clamp: 4;}
  .fa.fa-details {grid-template-columns: 1fr; gap: 22px;}
}
"""
