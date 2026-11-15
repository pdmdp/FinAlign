# Run the demo

The application uses Gradio and is configured for a Hugging Face ZeroGPU Space. It uses no paid inference API. ZeroGPU has a queue and a daily allowance; a free account can reach its limit during indexing or repeated searches.

## Upload to Hugging Face

Create a public Space with **Gradio**, the **Blank** template and **ZeroGPU** hardware. Use the MIT license for the code. No storage subscription is required.

Upload the **contents of this directory** to the Space root. `app.py`, `requirements.txt`, `packages.txt` and this `README.md` belong at the root; preserve the `finalign/` and `data/` directories. Do not upload the outer delivery ZIP as a single file. The ZIP inside `data/` is an application asset and should stay compressed.

The initial bundle needs `data/fiqa-corpus.zip`. It contains the full corpus, lexical index, passage IDs and model revisions, but no completed dense embeddings. It is generated with:

```bash
python -m pip install -r requirements.txt
python build_index.py --corpus-only --output data/fiqa-corpus
```

A Java JDK, version 11 or newer, is needed for Terrier and Pyjnius. Hugging Face installs `default-jdk-headless` from `packages.txt`; Pyjnius also looks for `javac`, which a JRE-only installation does not provide. Model files are downloaded at startup, so the first launch takes longer than subsequent searches.

## Prepare the dense index once

For a full corpus build, use [the Colab notebook](https://colab.research.google.com/github/pdmdp/FinAlign/blob/main/demo/prepare_index_colab.ipynb) with a free T4 GPU runtime. Run all cells, download `fiqa-index.zip`, and upload it, still compressed, inside the Space's `data/` directory. The notebook uses the same checkpointed encoder code as the app, verifies the downloaded corpus and validates the completed archive. Colab GPU availability is not guaranteed; its limits are separate from ZeroGPU's.

You can also prepare the index inside the Space. This consumes the same ZeroGPU allowance used for searches and may take more than one day's free quota:

1. Open the Space and sign in with its owner account using the login button.
2. Click **Prepare / continue index**. Each call encodes a bounded batch and checkpoints completed passages. Continue until all 66,113 passages are encoded. Calls use the free GPU allowance; if it runs out, wait for its reset and resume. Availability and total preparation time depend on the shared hardware.
3. Download the resulting **fiqa-index.zip**.
4. In the Space’s **Files** tab, open `data/` and upload that ZIP without unpacking it. This commits the completed index, preserving it across rebuilds. Until this upload, progress lives on the Space’s temporary disk and could be lost if the container is replaced.

When `data/fiqa-index.zip` is present, the app starts directly in search mode. Visitors do not run indexing. The corpus ZIP can remain in the repository; the completed index takes precedence.

Only the owner can invoke the preparation callback. OAuth requests basic profile access, not repository write access. No token needs to be pasted into the app or code.

## Build on another GPU

An existing CUDA environment can build the complete archive in one run:

```bash
python build_index.py --device cuda --output data/fiqa-index
```

Upload `data/fiqa-index.zip` with the application to skip Space preparation. A CPU or Apple MPS build is also supported, but full-corpus encoding can take hours on a laptop.

## Local checks

```bash
python -m unittest discover -s tests -v
python app.py
```

`FINALIGN_INDEX_DIR` may point to an unpacked index. `FINALIGN_DEVICE` defaults to CPU locally and CUDA in a Space. `FINALIGN_ENCODER_ONLY=1` skips downloading Qwen and monoT5 for a local BGE/BM25 smoke check; expansion and reranking controls are disabled in that mode.

The Space’s GPU settings determine the cost. The code does not switch hardware or buy credits. Queueing failures leave the BM25 result visible and can be retried later.

For hosting without recurring charges, keep the Space on the eligible free ZeroGPU tier and commit the completed index as a repository file. No storage bucket, paid inference API or dedicated GPU is required. Visitors remain subject to ZeroGPU's usage limits and queues; the free tier does not guarantee uninterrupted access. See the current [ZeroGPU terms](https://huggingface.co/docs/hub/spaces-zerogpu) before changing hardware or purchasing credits.
