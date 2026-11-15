"""Gradio entry point. Models are loaded before ZeroGPU allocates a worker."""

# ruff: noqa: I001

import json
import logging
import os
import time
from pathlib import Path

# Install ZeroGPU's CUDA emulation before libraries that may import torch.
import spaces
import gradio as gr
from finalign.core import Options, ranked, validate_query
from finalign.index import Index, LexicalSearch, unpack
from finalign.neural import NeuralSearch
from finalign.preparation import finish_index, prepare_batch
from finalign.presentation import (
    CSS,
    board_html,
    details_html,
    esc,
    notice_html,
    summary_html,
)

ROOT = Path(__file__).resolve().parent
index_dir = os.environ.get("FINALIGN_INDEX_DIR")
if index_dir is None:
    archive = ROOT / "data" / "fiqa-index.zip"
    if not archive.exists():
        archive = ROOT / "data" / "fiqa-corpus.zip"
    if not archive.exists():
        raise RuntimeError("Upload data/fiqa-corpus.zip alongside the application files. See DEPLOY.md.")
    index_dir = unpack(archive, ROOT / ".index")
INDEX = Index(index_dir)
LEXICAL = LexicalSearch(INDEX)
DEVICE = os.environ.get("FINALIGN_DEVICE", "cuda" if os.environ.get("SPACE_ID") else "cpu")
ENCODER_ONLY = os.environ.get("FINALIGN_ENCODER_ONLY", "0") == "1"
NEURAL = NeuralSearch(INDEX, device=DEVICE, encoder_only=ENCODER_ONLY)


@spaces.GPU(duration=60)
def build_passages(profile: gr.OAuthProfile | None):
    owner = os.environ.get("SPACE_ID", "pdmdp/finalign").split("/")[0]
    if os.environ.get("SPACE_ID") and (profile is None or profile.username != owner):
        raise gr.Error("Sign in as the Space owner to prepare the index.")
    return prepare_batch(INDEX, NEURAL.encoder, device=DEVICE)


def complete_preparation(progress):
    if not progress["done"]:
        message = (f"**{progress['completed']:,} / {progress['total']:,} passages prepared.** "
                   "Click Prepare / continue index again. Progress is saved between calls. "
                   "If the free GPU quota is exhausted, resume when it resets; no paid hardware is needed.")
        return message, gr.update(), gr.update(visible=False), gr.update()
    archive = finish_index(INDEX, ROOT / "data" / "fiqa-index.zip")
    return (("**The index is ready.** Download fiqa-index.zip below and upload it to the Space’s "
             "data folder to preserve it across rebuilds. You can now use the search interface."),
            gr.update(value=archive, visible=True), gr.update(visible=True), gr.update(visible=False))


def refresh_page():
    INDEX.refresh_dense()
    archive = ROOT / "data" / "fiqa-index.zip"
    if INDEX.ready:
        return (("**The index is ready.** Save the completed ZIP in the Space’s data folder if it "
                 "has not been uploaded yet."),
                gr.update(value=str(archive) if archive.exists() else None, visible=archive.exists()),
                gr.update(visible=True), gr.update(visible=False))
    progress_path = INDEX.path / "indexing-progress.json"
    count = json.loads(progress_path.read_text())["completed"] if progress_path.exists() else 0
    return ((f"**{count:,} / {len(INDEX.passage_ids):,} passages prepared.** "
             "Sign in as the Space owner and prepare the next batch."),
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=True))


def theme():
    """The Space's original palette (slate and blue), with contrast raised to AA in both modes."""
    return gr.themes.Base(primary_hue="blue", neutral_hue="slate").set(
        body_text_color_subdued="*neutral_600", body_text_color_subdued_dark="*neutral_400",
        button_primary_background_fill="*primary_600", button_primary_background_fill_dark="*primary_600",
        button_primary_background_fill_hover="*primary_700", button_primary_background_fill_hover_dark="*primary_700",
        button_primary_text_color="white", button_primary_text_color_dark="white",
        button_secondary_background_fill="transparent", button_secondary_background_fill_dark="transparent",
        button_secondary_background_fill_hover="transparent", button_secondary_background_fill_hover_dark="transparent",
        checkbox_background_color_selected="*primary_600", checkbox_background_color_selected_dark="*primary_500",
        checkbox_border_color_selected="*primary_600", checkbox_border_color_selected_dark="*primary_500")


def gpu_failure(error):
    """Turn a ZeroGPU or model failure into an inline notice; BM25 results stay visible."""
    message = str(getattr(error, "message", "") or error).strip()[:300]
    lowered = message.lower()
    if any(word in lowered for word in ("zerogpu", "quota", "exceeded", "runs limit")):
        return ("quota", "GPU quota reached",
                "The BM25 ranking is complete. FinAlign needs a ZeroGPU slot, and the free allowance "
                "for this visitor is used up. Signed-in Hugging Face users receive a larger daily "
                f"allowance; otherwise, try again after it resets.<small>{esc(message)}</small>")
    return ("error", "FinAlign could not finish this search",
            "The BM25 ranking is complete. Try again in a moment; the Space logs contain details."
            + (f"<small>{esc(message)}</small>" if message else ""))


def prepare(query, keywords, hyde, lexical, rerank):
    started = time.perf_counter()
    skip = gr.update()
    INDEX.refresh_dense()
    if not INDEX.ready:
        return (None, notice_html("error", "The search index is not ready yet.",
                                  "The Space owner is still preparing the passage embeddings."), skip, skip, skip)
    try:
        query = validate_query(query)
    except ValueError as error:
        return (None, notice_html("error", esc(error)), skip, skip, skip)
    scores, elapsed = LEXICAL.search(query)
    options = Options(keywords, hyde, lexical, rerank)
    known = INDEX.known_query(query)
    state = {"query": query, "lexical": scores, "lexical_ms": elapsed, "options": options.to_dict(),
             "started": started}
    ranking = ranked(scores)
    return (state, "",
            summary_html(query, known, ranking, state="waiting", lexical_ms=elapsed),
            board_html(INDEX, query, ranking, known=known, options=options.to_dict(), state="waiting",
                       lexical_ms=elapsed),
            details_html())


@spaces.GPU(duration=60)
def neural_search(state):
    return NEURAL.search(state["query"], state["lexical"], state["options"])


def search(state):
    if not state:
        return (gr.update(),) * 3
    query, options = state["query"], state["options"]
    known = INDEX.known_query(query)
    lexical = ranked(state["lexical"])
    try:
        result = neural_search(state)
    except Exception as error:  # Shown inline instead of a toast and per-component error badges.
        logging.exception("FinAlign search failed")
        return (summary_html(query, known, lexical, state="failed", lexical_ms=state["lexical_ms"]),
                board_html(INDEX, query, lexical, known=known, options=options, state="failed",
                           lexical_ms=state["lexical_ms"], notice=gpu_failure(error)),
                gr.update())
    neural = result["ranking"]
    timings = {"BM25": state["lexical_ms"], **result["timings_ms"]}
    elapsed = sum(timings.values()) / 1000
    server_elapsed = time.perf_counter() - state["started"]
    return (summary_html(query, known, lexical, neural, state="done", lexical_ms=state["lexical_ms"],
                         timings=result["timings_ms"], server_s=server_elapsed),
            board_html(INDEX, query, lexical, neural, result["passages"], known, options, state="done",
                       lexical_ms=state["lexical_ms"], neural_ms=sum(result["timings_ms"].values())),
            details_html(timings, result["variants"], elapsed, server_elapsed))


def make_app():
    with gr.Blocks(title="FinAlign · Financial document retrieval", analytics_enabled=False) as app:
        gr.HTML('<header id="fa-masthead" class="fa"><div class="fa-top"><span class="fa-kicker">Financial document '
                'retrieval</span><nav aria-label="Project links">'
                '<a href="https://github.com/pdmdp/FinAlign">Code &amp; report</a>'
                '<a href="https://huggingface.co/datasets/BeIR/fiqa">Dataset</a></nav></div>'
                '<h1>FinAlign</h1>'
                '<p class="fa-lede">Ask a financial question in English and compare two top-10 rankings of '
                'FiQA documents: BM25 keyword search and FinAlign, a BGE passage retriever with optional '
                'query expansion, score fusion and reranking.</p>'
                f'<ul class="fa-meta"><li><b>{INDEX.manifest["documents"]:,}</b> documents</li>'
                f'<li><b>{INDEX.manifest["passages"]:,}</b> passages</li><li>FiQA / BEIR</li>'
                '<li>English</li></ul></header>', padding=False)
        with gr.Group(visible=not INDEX.ready):
            gr.Markdown("### Prepare the passage index")
            setup_status = gr.Markdown(
                "The corpus and BM25 index are ready. The Space owner can prepare BGE embeddings "
                "using the free GPU in short, resumable batches. This is needed only once.")
            if os.environ.get("SPACE_ID"):
                gr.LoginButton()
            build_button = gr.Button("Prepare / continue index", variant="primary")
            archive_download = gr.File(label="Save the completed index", visible=False, interactive=False)
            build_state = gr.State()
        with gr.Column(visible=INDEX.ready, elem_id="fa-app") as search_panel:
            with gr.Row(elem_id="fa-searchbar"):
                query = gr.Textbox(label="Question", show_label=False, container=False,
                                   placeholder="Why does a bond’s price fall when interest rates rise?",
                                   lines=1, max_lines=4, max_length=600, elem_id="fa-query", scale=1)
                submit = gr.Button("Compare rankings", variant="primary", elem_id="fa-submit", scale=0,
                                   min_width=176)
            with gr.Row(elem_id="fa-examples"):
                gr.HTML('<div class="fa"><span class="fa-label">Try a FiQA question</span></div>', padding=False,
                        elem_classes="fa-inline")
                # Fixed questions from the dev split; selected for readable wording, not scores.
                example_ids = {"1952", "1718", "469"}
                example_texts = [r["query"] for r in INDEX.queries["dev"] if r["qid"] in example_ids]
                example_buttons = [(gr.Button(text, size="sm", variant="secondary"), text)
                                   for text in example_texts]
            notice = gr.HTML("", padding=False, elem_id="fa-notice-slot")
            with gr.Column(elem_id="fa-pipe"):
                with gr.Row(elem_id="fa-pipe-head"):
                    gr.HTML('<div class="fa"><span class="fa-label">FinAlign pipeline</span><span class="fa-pipe-hint">'
                            'Switch stages on to see what each one changes. BM25 is not affected.</span></div>',
                            padding=False, elem_classes="fa-inline")
                    bge_only = gr.Button("BGE only", size="sm", variant="secondary", scale=0, min_width=0)
                    all_four = gr.Button("All stages", size="sm", variant="secondary", scale=0, min_width=0)
                with gr.Row(elem_id="fa-stages"):
                    keywords = gr.Checkbox(label="Keyword expansion", value=False,
                                           info="Qwen adds financial terms as an extra query.",
                                           interactive=not ENCODER_ONLY, elem_classes="fa-stage")
                    hyde = gr.Checkbox(label="Hypothetical answer", value=False,
                                       info="Qwen drafts an answer, searched as an extra query.",
                                       interactive=not ENCODER_ONLY, elem_classes="fa-stage")
                    gr.HTML('<div class="fa fa-stage-fixed"><span class="fa-stage-title">BGE passage search'
                            '</span><span class="fa-stage-info">Ranks each document by its '
                            'best 250-word passage.</span></div>', padding=False, elem_classes="fa-stage-core")
                    lexical = gr.Checkbox(label="BM25 fusion", value=False,
                                          info="Adds normalized BM25 and BGE scores, equal weight.",
                                          elem_classes="fa-stage")
                    rerank = gr.Checkbox(label="monoT5 rerank", value=False,
                                         info="Rescores the first 50 candidates.",
                                         interactive=not ENCODER_ONLY, elem_classes="fa-stage")
            summary = gr.HTML("", padding=False, elem_id="fa-summary-slot")
            board = gr.HTML(board_html(INDEX), padding=False, elem_id="fa-board-slot")
            with gr.Accordion("Run details · latency by stage and query variants", open=False,
                              elem_classes="fa-accordion"):
                details = gr.HTML(details_html(), padding=False)
        with gr.Accordion("About the experiment", open=False, elem_classes="fa-accordion"):
            gr.Markdown(
                "BGE retrieves up to 400 non-overlapping, 250-word passages per query variant. "
                "Variant scores are normalized and combined (original 0.70, keywords 0.15, "
                "hypothetical 0.15); the best passage represents each document. Disabled variants "
                "contribute zero. BM25 and dense document scores receive equal weight when blended. "
                "monoT5 reranks the first 50 candidates.\n\n"
                "This interactive version follows the course notebook’s method and fixes model revisions "
                "and the expansion seed. Its timings and rankings are new runs. The recorded course "
                "results and their limitations remain in the [GitHub repository](https://github.com/pdmdp/FinAlign). "
                "A single query is not an ablation study across a benchmark.\n\n"
                "FiQA is a historical research collection. Its posts and generated query expansions "
                "are not current financial guidance.")
            archive = ROOT / "data" / "fiqa-index.zip"
            if archive.exists():
                gr.File(value=str(archive), label="Prepared index · CC BY-SA 4.0", interactive=False)
        gr.Markdown(
            "[Code & report](https://github.com/pdmdp/FinAlign) · "
            "[FiQA / BEIR dataset](https://huggingface.co/datasets/BeIR/fiqa) · "
            "[Data license: CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)\n\n"
            "Original project: Matteo Pietro Di Pilato, Lorenzo Cinquemani and Lorenzo Marchioni. "
            "Interactive adaptation: Matteo Pietro Di Pilato.", elem_id="fa-footer")
        build_button.click(build_passages, None, build_state, concurrency_limit=1,
                           concurrency_id="search", api_visibility="private").success(
            complete_preparation, build_state, [setup_status, archive_download, search_panel, build_button],
            concurrency_limit=1, concurrency_id="search", api_visibility="private")
        app.load(refresh_page, None, [setup_status, archive_download, search_panel, build_button],
                 queue=False, api_visibility="private")

        options = [keywords, hyde, lexical, rerank]
        bge_only.click(lambda: [False] * 4, None, options, queue=False, api_visibility="private")
        full = not ENCODER_ONLY
        all_four.click(lambda: [full, full, True, full], None, options, queue=False, api_visibility="private")

        state = gr.State()
        inputs = [query, *options]
        prep_outputs = [state, notice, summary, board, details]
        search_outputs = [summary, board, details]
        # The board's own skeleton and status replace Gradio's overlay while FinAlign waits for a GPU.
        shared = {"concurrency_limit": 1, "concurrency_id": "search", "api_visibility": "private",
                  "show_progress": "hidden"}
        gr.on([submit.click, query.submit], prepare, inputs, prep_outputs, **shared).success(
            search, state, search_outputs, **shared)
        # An example fills the question and runs the comparison in one click.
        for button, text in example_buttons:
            button.click(lambda text=text: text, None, query, queue=False, api_visibility="private").then(
                prepare, inputs, prep_outputs, **shared).success(search, state, search_outputs, **shared)
    return app


if __name__ == "__main__":
    make_app().queue(max_size=12).launch(theme=theme(), css=CSS, show_error=False, ssr_mode=False)
