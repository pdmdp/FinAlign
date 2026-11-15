"""Gradio entry point. Models are loaded before ZeroGPU allocates a worker."""

# ruff: noqa: I001

import json
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
from finalign.presentation import CSS, query_metrics, results_html

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


def prepare(query, keywords, hyde, lexical, rerank):
    started = time.perf_counter()
    INDEX.refresh_dense()
    if not INDEX.ready:
        raise gr.Error("The Space owner is still preparing the passage index.")
    try:
        query = validate_query(query)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    scores, elapsed = LEXICAL.search(query)
    options = Options(keywords, hyde, lexical, rerank)
    known = INDEX.known_query(query)
    state = {"query": query, "lexical": scores, "lexical_ms": elapsed, "options": options.to_dict(),
             "started": started}
    return (state, results_html(INDEX, ranked(scores), known=known),
            '<div class="empty-result">Retrieving passages…</div>',
            "BM25 is ready. FinAlign will run when a GPU is available.", [], "", "")


@spaces.GPU(duration=60)
def search(state):
    if not state:
        raise gr.Error("Enter a question before starting a search.")
    result = NEURAL.search(state["query"], state["lexical"], state["options"])
    known = INDEX.known_query(state["query"])
    timings = {"BM25": state["lexical_ms"], **result["timings_ms"]}
    rows = [[stage, round(ms, 1)] for stage, ms in timings.items()]
    elapsed = sum(timings.values()) / 1000
    variants = "\n\n".join(f"{name.title()}\n{text}"
                             for name, text in result["variants"].items() if name != "original")
    if not variants:
        variants = "Query expansion is off. BGE encodes the original question with its retrieval instruction."
    server_elapsed = time.perf_counter() - state["started"]
    detail = (f"**{elapsed:.2f} s processing** · {server_elapsed:.2f} s elapsed on the server, including GPU wait. "
              "Full FiQA corpus · 10 results shown. Browser transfer is excluded.")
    return (results_html(INDEX, result["ranking"], result["passages"], known), detail,
            rows, variants, query_metrics(known, ranked(state["lexical"]), result["ranking"]))


def make_app():
    with gr.Blocks(title="FinAlign · Financial document retrieval", analytics_enabled=False) as app:
        gr.HTML(f'<header id="masthead"><div class="eyebrow">Financial document retrieval</div>'
                '<h1>FinAlign</h1><p>Ask a financial question. Compare keyword search with neural retrieval, '
                'then see what query expansion and reranking change.</p>'
                f'<div class="collection">FiQA / BEIR · {INDEX.manifest["documents"]:,} documents · '
                f'{INDEX.manifest["passages"]:,} passages · English</div></header>')
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
        with gr.Group(visible=INDEX.ready) as search_panel:
            with gr.Row():
                query = gr.Textbox(label="Your question", placeholder="Why does a bond’s price fall when interest rates rise?",
                                   lines=2, max_lines=3, max_length=600, scale=5, min_width=280)
                submit = gr.Button("Compare search results", variant="primary", scale=1)
            # Fixed questions from the dev split; selected for readable wording, not scores.
            example_ids = {"1952", "1718", "469"}
            example_rows = [[r["query"]] for r in INDEX.queries["dev"] if r["qid"] in example_ids]
            gr.Examples(examples=example_rows, inputs=[query], label="Try a FiQA question", cache_examples=False)
            with gr.Accordion("Retrieval settings", open=True):
                gr.Markdown("Start with BGE, or turn on components to inspect their effect. "
                            "All four options together reproduce the notebook’s final pipeline design.")
                with gr.Row(elem_id="retrieval-options"):
                    keywords = gr.Checkbox(label="Technical keywords · Qwen", value=False, interactive=not ENCODER_ONLY)
                    hyde = gr.Checkbox(label="Hypothetical expansion · Qwen", value=False, interactive=not ENCODER_ONLY)
                    lexical = gr.Checkbox(label="Blend BM25 with BGE", value=False)
                    rerank = gr.Checkbox(label="Rerank top 50 · monoT5", value=False, interactive=not ENCODER_ONLY)
            status = gr.Markdown("BGE is selected. Turn on extra components above to compare their contribution.")
            with gr.Row(equal_height=False):
                with gr.Column():
                    gr.Markdown("### BM25\nLexical matching across full documents.")
                    baseline = gr.HTML('<div class="empty-result">Enter a question to see the baseline ranking.</div>')
                with gr.Column():
                    gr.Markdown("### FinAlign\nBGE passages with the components selected above.")
                    neural = gr.HTML('<div class="empty-result">The neural ranking will appear here.</div>')
            with gr.Accordion("Inspect this search", open=False):
                quality = gr.Markdown()
                latency = gr.Dataframe(headers=["Stage", "Milliseconds"], datatype=["str", "number"],
                                       interactive=False, label="Measured latency")
                gr.Markdown("**Query variants** · Generated text below is used only to retrieve documents.")
                variants = gr.Textbox(label="Expanded retrieval queries", lines=6, interactive=False)
        with gr.Accordion("About the experiment", open=False):
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
            "Interactive adaptation: Matteo Pietro Di Pilato.", elem_id="footer")
        build_button.click(build_passages, None, build_state, concurrency_limit=1,
                           concurrency_id="search", api_visibility="private").success(
            complete_preparation, build_state, [setup_status, archive_download, search_panel, build_button],
            concurrency_limit=1, concurrency_id="search", api_visibility="private")
        app.load(refresh_page, None, [setup_status, archive_download, search_panel, build_button],
                 queue=False, api_visibility="private")
        state = gr.State()
        inputs = [query, keywords, hyde, lexical, rerank]
        prep_outputs = [state, baseline, neural, status, latency, variants, quality]
        gr.on([submit.click, query.submit], prepare, inputs, prep_outputs,
              concurrency_limit=1, concurrency_id="search", api_visibility="private").success(
                  search, state, [neural, status, latency, variants, quality],
                  concurrency_limit=1, concurrency_id="search", api_visibility="private")
    return app


if __name__ == "__main__":
    make_app().queue(max_size=12).launch(
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"), css=CSS,
        show_error=False, ssr_mode=False)
