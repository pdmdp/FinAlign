"""Render corpus text as escaped content, never as executable markup."""

import html

from .core import metrics


def results_html(index, ranking, passages=None, known=None):
    passages = passages or {}
    if not ranking:
        return '<div class="empty-result">No documents matched this query.</div>'
    cards = []
    for position, (docno, score) in enumerate(ranking[:10], 1):
        snippet = index.snippet(docno, passages.get(docno))
        words = snippet.split()
        preview = " ".join(words[:80]) + (" …" if len(words) > 80 else "")
        badge = ""
        if known:
            relevant = known["qrels"].get(docno, 0) > 0
            label = "Judged relevant" if relevant else "Not listed in qrels"
            badge = f'<span class="judgment {"relevant" if relevant else "unjudged"}">{label}</span>'
        source = "BGE passage" if docno in passages else "Document opening"
        cards.append(
            f'<article class="result"><div class="result-meta"><span class="rank">{position:02}</span>'
            f'<span>FiQA · {html.escape(docno)}</span>{badge}'
            f'<span class="score">{float(score):.4f}</span></div>'
            f'<p>{html.escape(preview)}</p>'
            f'<details><summary>Read document <span>· {source} shown above</span></summary>'
            f'<div class="document-text">{html.escape(index.documents[docno])}</div></details></article>')
    return '<div class="results">' + "".join(cards) + '</div>'


def query_metrics(known, lexical, neural):
    if not known:
        return ("This is a custom query. FiQA relevance judgments are unavailable, so no "
                "quality score is shown. Retrieval scores are not probabilities.")
    rows = []
    for name, ranking in (("BM25", lexical), ("FinAlign", neural)):
        m = metrics([d for d, _ in ranking], known["qrels"])
        if m:
            rows.append(f"| {name} | {m['nDCG@10']:.3f} | {m['Recall@10']:.3f} |")
    if not rows:
        return "This query has no positive judgments in the retained corpus."
    return (f"**FiQA {known['split']} · query {known['qid']}** — metrics for this question only.\n\n"
            "| Ranking | nDCG@10 | Recall@10 |\n| :--- | ---: | ---: |\n" + "\n".join(rows)
            + "\n\nA document absent from the judgments is not necessarily irrelevant.")


CSS = """
.gradio-container {max-width: 1240px !important; margin: auto;}
#masthead {padding: 24px 0 12px; border-bottom: 1px solid var(--border-color-primary); margin-bottom: 16px;}
#masthead .eyebrow {letter-spacing: .16em; text-transform: uppercase; font-size: 11px; color: #7b8391; font-weight: 600;}
#masthead h1 {font-size: clamp(36px, 5vw, 52px); line-height: 1.05; margin: 10px 0 12px; letter-spacing: -.04em;}
#masthead p {font-size: 16px; max-width: 700px; line-height: 1.6;}
.collection {font-size: 12px; color: #7b8391; margin-top: 14px;}
.results {margin-top: 4px;}
.result {padding: 19px 2px; border-bottom: 1px solid var(--border-color-primary);}
.result-meta {display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 11px; color: #7b8391;}
.rank {color: var(--body-text-color); font-size: 15px; font-weight: 650; min-width: 24px;}
.score {margin-left: auto; font-variant-numeric: tabular-nums;}
.result p {line-height: 1.65; font-size: 14px; margin: 11px 0; overflow-wrap: anywhere;}
.result summary {font-size: 12px; cursor: pointer; color: #5683bf; padding: 4px 0;}
.result summary span {color: #7b8391;}
.document-text {white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.65; font-size: 13px; margin-top: 12px;}
.judgment {font-size: 10px; padding: 2px 6px; border-radius: 3px;}
.judgment.relevant {color: #167658; background: #d9f2e8;}
.judgment.unjudged {color: #7b8391;}
.empty-result {padding: 30px 4px; color: #7b8391; font-size: 14px;}
#footer {border-top: 1px solid var(--border-color-primary); padding-top: 16px; margin-top: 24px; font-size: 12px;}
"""
