# FinAlign  
**Hybrid Financial Question Answering over FiQA (BEIR Benchmark)**  

Matteo Pietro Di Pilato – Lorenzo Cinquemani – Lorenzo Marchioni  
Bachelor’s Degree in Artificial Intelligence  
University of Pavia  

---

## 📌 Overview

**FinAlign** is a multi-stage hybrid Information Retrieval system designed for Financial Question Answering (FiQA) using the BEIR benchmark.

The project addresses a central challenge in financial retrieval:

> Bridging the vocabulary and knowledge gap between non-expert user queries and technical financial documents.

Unlike traditional keyword-based systems, FinAlign integrates:

- Lexical retrieval (BM25)
- Dense semantic retrieval (BGE bi-encoder)
- LLM-based query rewriting
- HyDE-style hypothetical document embeddings
- Document chunking (passage-level retrieval)
- Score fusion strategies (RRF & linear combination)
- Neural re-ranking with monoT5

The results show that instruction-aware dense retrieval combined with chunking and multi-stage re-ranking significantly improves performance over classical IR baselines.

---

## 📂 Dataset

We use the FiQA dataset from the BEIR benchmark.

- ~57,600 documents (after cleaning)
- 648 unique queries
- 1,705 relevance judgments (qrels)
- Heterogeneous corpus including:
  - Financial news
  - Microblogs
  - Report snippets

This is an opinion-based explanatory retrieval task rather than simple fact extraction.

---

## 🏗 System Architecture

FinAlign is implemented as a multi-stage retrieval pipeline:

### 1️⃣ Baseline Systems
- TF-IDF  
- BM25  
- BM25 + RM3 (pseudo-relevance feedback)  

### 2️⃣ Dense Retrieval
- BGE-base bi-encoder  
- Instruction-aware query prefixing:

```
Represent this sentence for searching relevant financial passages:
```

### 3️⃣ Query Expansion
- LLM-based query rewriting  
- HyDE-style hypothetical document embeddings  
- Multi-query fusion  

### 4️⃣ Document Chunking
- Fixed-length overlapping chunks  
- Dense indexing at chunk level  
- Max-pooling aggregation to document level  

### 5️⃣ Fusion Strategies
- Reciprocal Rank Fusion (RRF)  
- Weighted linear score combination  

### 6️⃣ Neural Re-ranking
- monoT5 cross-encoder  
- Applied to a restricted candidate set due to computational constraints  

---

## 📊 Evaluation

Evaluation is performed on the cleaned BEIR/FiQA test set using standard IR metrics:

- MAP  
- Precision@k (1, 5, 10)  
- Recall@k (5, 10)  
- nDCG@5  
- nDCG@10  

All systems are evaluated under identical conditions.

---

## 🚀 Main Results

Key findings:

- Dense retrieval (BGE) significantly outperforms lexical baselines.
- Instruction-aware prefixing improves embedding alignment.
- Document chunking reduces semantic dilution and improves ranking quality.
- Query rewriting works best as diversification (multi-query strategy).
- Hybrid lexical + dense retrieval did not outperform dense retrieval alone.
- monoT5 re-ranking produces the strongest overall results.

The best-performing pipeline combines:

Dense retrieval (instruction-tuned)  
+ LLM query rewriting  
+ Chunk-level retrieval  
+ Linear fusion  
+ monoT5 re-ranking  

For complete numerical results, detailed metric tables, and full experimental analysis, please refer to the project report.

---

## 🛠 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/pdmdp/FinAlign.git
cd FinAlign
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Install monoT5 extension

```bash
pip install --upgrade git+https://github.com/terrierteam/pyterrier_t5.git
```

---

## ⚙️ Implementation Details

- Python  
- PyTerrier  
- Terrier indexing  
- Sentence-Transformers (BGE)  
- FAISS  
- HuggingFace Transformers  
- monoT5 re-ranker  
- Google Colab (T4 GPU)  

Due to hardware constraints, re-ranking is applied only to a limited candidate set.

---

## 📄 Project Report

A complete description of:

- Methodology  
- Experimental setup  
- Detailed metric tables  
- Discussion and limitations  
- Future work  

is available in:

📎 `report/FinalReportIRProject.pdf`

For deeper technical details, please consult the full report.

---

## 🔬 Key Takeaways

This project highlights that:

- Financial retrieval is primarily a semantic matching problem.
- Instruction-aware dense encoders significantly improve alignment.
- Chunk-level retrieval is crucial in heterogeneous financial corpora.
- Multi-stage pipelines outperform single-model systems.
- Careful system-level design is essential for domain-specific retrieval tasks.

---

## 📌 Academic Integrity

AI tools (ChatGPT, Claude, Gemini, GitHub Copilot) were used only for debugging assistance and minor clarity improvements.

All system design, implementation, experiments, and evaluation were independently developed by the authors.
